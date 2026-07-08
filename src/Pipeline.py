"""Function-call resolution via constrained decoding.

Given a natural-language prompt and the set of available functions, this
module first walks the LLM's next-token logits to pick a function name
from a token trie (``get_name``), then, for the chosen function, extracts
a typed value for each parameter by restricting generation to the tokens
allowed for that parameter's type (``get_args``). In both cases the model
can only ever produce output that is structurally guaranteed to be valid:
it can never "hallucinate" a function name outside the known set, nor
produce a string value that would break JSON encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm_sdk import Small_LLM_Model

from src.Parser import FunctionDefinition

Argument = str | float | int | bool


class PipelineError(Exception):
    """Raised when the pipeline cannot resolve a function call."""


@dataclass
class _TrieNode:
    """One node of a token trie built from a closed set of literals."""

    children: dict[int, _TrieNode] = field(default_factory=dict)
    value: str | None = None

    @property
    def is_leaf(self) -> bool:
        """Whether this node marks the end of a complete literal."""
        return self.value is not None


class Pipeline:
    """Turns a natural-language prompt into a typed function call."""

    #: Text used to prompt the model right before it must produce a name.
    #: It must be identical everywhere it is used, so that a candidate
    #: name is tokenized exactly as it would be during real generation.
    #: Deliberately has NO trailing space: BPE tokenizers tend to merge a
    #: trailing space with the word that follows (" fn" is one token, not
    #: a space token followed by "fn"), so a prefix ending in a bare space
    #: would not match the start of "prefix + name" once re-tokenized
    #: together. Ending on ":" keeps the shared prefix stable, and the
    #: space is generated as part of the name itself.
    NAME_PREFIX = "\nFunction name:"

    #: Characters allowed, alone, to make up a numeric-value token.
    _NUMERIC_CHARS = set("0123456789.-")

    def __init__(
        self, llm: Small_LLM_Model, functions: list[FunctionDefinition]
    ) -> None:
        """Build the pipeline: load the vocab and the function-name trie.

        Args:
            llm: The loaded LLM wrapper, used to tokenize text and get
                logits.
            functions: The available function definitions to choose from.
        """
        if not functions:
            raise PipelineError(
                "At least one function definition is required."
            )
        self.llm = llm
        self.functions = functions
        self._vocab = self._load_vocab()
        self._numeric_ids = {
            tid
            for tid, token in self._vocab.items()
            if token and set(token) <= self._NUMERIC_CHARS
        }
        self._string_ids = {
            tid for tid, token in self._vocab.items() if '"' not in token
        }
        self._quote_stop_ids = set(self._encode_ids('"'))
        self._newline_stop_ids = set(self._encode_ids("\n"))
        self._name_trie = self._build_literal_trie(
            self.NAME_PREFIX, [function.name for function in functions]
        )

    # -- low-level helpers -------------------------------------------

    def _encode_ids(self, text: str) -> list[int]:
        """Encode *text* into a flat list of token ids."""
        return [int(t) for t in self.llm.encode(text)[0].tolist()]

    def _load_vocab(self) -> dict[int, str]:
        """Load the token-id -> raw-token-string mapping, once."""
        path = self.llm.get_path_to_vocab_file()
        with open(path, encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
        return {token_id: token for token, token_id in raw.items()}

    # -- closed-set generation: walking a trie of literals ------------

    def _build_literal_trie(
        self, suffix: str, literals: list[str]
    ) -> _TrieNode:
        """Tokenize a closed set of literals into a shared trie.

        Args:
            suffix: The exact text that will precede generation when
                this trie is walked (see NAME_PREFIX for why the exact
                text matters).
            literals: The closed set of allowed values (e.g. function
                names, or "true"/"false").
        """
        root = _TrieNode()
        prefix_len = len(self._encode_ids(suffix))
        for literal in literals:
            token_ids = self._encode_ids(suffix + " " + literal)
            literal_ids = token_ids[prefix_len:]
            if not literal_ids:
                raise PipelineError(
                    f"Could not tokenize literal '{literal}'."
                )
            node = root
            for token_id in literal_ids:
                node = node.children.setdefault(token_id, _TrieNode())
            node.value = literal
        return root

    def _trie_depth(self, node: _TrieNode) -> int:
        """Return the length, in tokens, of the trie's longest branch."""
        if not node.children:
            return 0
        return 1 + max(self._trie_depth(c) for c in node.children.values())

    def _walk_trie(self, prompt_text: str, trie: _TrieNode) -> str:
        """Constrained-decode a value that must be one of *trie*'s leaves."""
        input_ids = self._encode_ids(prompt_text)
        node = trie
        generated_ids: list[int] = []
        for _ in range(self._trie_depth(trie) + 1):
            if node.is_leaf:
                break
            logits = self.llm.get_logits_from_input_ids(
                input_ids + generated_ids
            )
            if not node.children:
                raise PipelineError(
                    "Reached a dead end while walking a trie."
                )
            next_id = max(node.children, key=lambda tid: logits[tid])
            generated_ids.append(next_id)
            node = node.children[next_id]
        if node.value is None:
            raise PipelineError("Failed to resolve a value from the trie.")
        return node.value

    # -- open-ended generation: character-class masking ---------------

    def _generate_constrained(
        self,
        prompt_text: str,
        allowed_ids: set[int],
        stop_ids: set[int],
        max_tokens: int,
    ) -> str:
        """Constrained-decode free-form text within a fixed token set.

        Generation stops as soon as a token from *stop_ids* is chosen
        (it is not included in the returned text), or after
        *max_tokens* tokens as a safety bound.
        """
        input_ids = self._encode_ids(prompt_text)
        candidates = allowed_ids | stop_ids
        if not candidates:
            raise PipelineError("No candidate tokens available.")
        generated: list[int] = []
        for _ in range(max_tokens):
            logits = self.llm.get_logits_from_input_ids(
                input_ids + generated
            )
            next_id = max(candidates, key=lambda tid: logits[tid])
            if next_id in stop_ids:
                break
            generated.append(next_id)
        return self.llm.decode(generated) if generated else ""

    # -- function name --------------------------------------------------

    def _build_system_prompt(self, prompt: str) -> str:
        """Build the natural-language prompt listing the functions."""
        lines = [
            "You must pick exactly one function to answer the request "
            "below.",
            "",
        ]
        for function in self.functions:
            params = ", ".join(function.parameters)
            lines.append(
                f"- {function.name}({params}): {function.description}"
            )
        lines.append("")
        lines.append(f'Request: "{prompt}"')
        return "\n".join(lines) + self.NAME_PREFIX

    def get_name(self, prompt: str) -> str:
        """Select, via constrained decoding, the matching function name.

        Args:
            prompt: The natural-language request to resolve.

        Returns:
            The name of the chosen function, guaranteed to be one of the
            names present in the function definitions used to build the
            pipeline.

        Raises:
            PipelineError: If no valid function name could be resolved.
        """
        text = self._build_system_prompt(prompt)
        return self._walk_trie(text, self._name_trie)

    # -- function arguments -----------------------------------------------

    def _build_value_prompt(
        self,
        prompt: str,
        function: FunctionDefinition,
        param_name: str,
        param_type: str,
        known: dict[str, Argument],
    ) -> str:
        """Build the prompt asking the model for a single parameter's value."""
        lines = [
            f'Request: "{prompt}"',
            f"You are calling {function.name}: {function.description}",
        ]
        if known:
            resolved = ", ".join(f"{k}={v!r}" for k, v in known.items())
            lines.append(f"Already resolved: {resolved}")
        lines.append(
            f'Give the value of parameter "{param_name}" ({param_type}).'
        )
        text = "\n".join(lines) + f"\n{param_name} = "
        if param_type == "string":
            text += '"'
        return text

    def _cast_value(
        self, param_name: str, param_type: str, text: str
    ) -> Argument:
        """Parse the raw generated text into its final Python type."""
        if param_type == "string":
            return text
        if param_type == "boolean":
            return text == "true"
        try:
            if param_type == "integer":
                return int(text)
            return float(text)
        except ValueError as exc:
            raise PipelineError(
                f"Could not parse value {text!r} for parameter "
                f"'{param_name}' as {param_type}."
            ) from exc

    def get_args(
        self, prompt: str, function: FunctionDefinition
    ) -> dict[str, Argument]:
        """Extract, via constrained decoding, one typed value per parameter.

        Args:
            prompt: The natural-language request to resolve.
            function: The function whose parameters must be filled in,
                typically the result of a prior call to ``get_name``.

        Returns:
            A dict mapping each parameter name to its typed value.

        Raises:
            PipelineError: If a value could not be extracted or parsed.
        """
        values: dict[str, Argument] = {}
        for param_name, param in function.parameters.items():
            text_prompt = self._build_value_prompt(
                prompt, function, param_name, param.type, values
            )
            if param.type == "boolean":
                trie = self._build_literal_trie(text_prompt, ["true", "false"])
                raw = self._walk_trie(text_prompt, trie)
            elif param.type == "string":
                raw = self._generate_constrained(
                    text_prompt, self._string_ids, self._quote_stop_ids, 40
                )
            else:
                raw = self._generate_constrained(
                    text_prompt, self._numeric_ids, self._newline_stop_ids, 16
                )
            values[param_name] = self._cast_value(param_name, param.type, raw)
        return values
