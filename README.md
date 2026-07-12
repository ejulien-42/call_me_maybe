*This project has been created as part of the 42 curriculum by ejulien.*

# call me maybe

## Description

Translates natural-language requests into structured function calls using
Qwen3-0.6B. Given *"What is the sum of 40 and 2?"*, the program outputs
`{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}` — not the
answer. **Constrained decoding** guarantees the output is always valid JSON:
the model only makes semantic choices, never structural ones.

## Instructions

```bash
make install    # uv sync
make run        # or: uv run python -m src [--functions_definition F] [--input I] [--output O]
make lint       # flake8 + mypy
```

First run downloads the model weights.

## Algorithm explanation

1. **Function selection** — the prompt lists functions as a numbered
   multiple-choice question and ends on `[`. One logits call; every token
   except the digits `0..N-1` is masked to `-inf`; the surviving argmax is
   always a valid index into our function list.
2. **Parameters** — the JSON syntax (keys, quotes, commas) is *injected* into
   the token stream by our code; the model only fills the value slots:
   - *string*: free generation, stopped when a pair of quotes appears in the
     decoded text; value re-grounded on the exact prompt spelling.
   - *number*: generation stops at the first non-numeric token; if the parsed
     value is absent from the prompt (model "answered" instead of extracting),
     fall back to a number that is present.
   - *boolean*: single logits call, `"true"` score vs `"false"` score.

The output file is written by `json.dump`, so validity never depends on the model.

## Design decisions

- Index-based selection: one token, one LLM call, cannot drift (limit: 10 functions).
- Syntax injected, never generated: absolute structural guarantee.
- Values grounded on the prompt: it is the source of truth for extraction.
- pydantic validation of all input files before any LLM work.

## Performance analysis

11/11 correct calls on the provided test set (and on a harder custom set);
100% valid JSON by construction; full run well under the 5-minute budget.

## Challenges faced

- BPE boundary effects: a trailing space merges into the next token — solved
  by comparing decoded text instead of token ids.
- Model answering instead of extracting (√16 → 4) — solved by the numeric
  grounding fallback.
- Typographic quotes, merged quote+comma tokens, escaped backslashes — solved
  by scanning the decoded text for the closing quote.

## Testing strategy

End-to-end runs on two function sets; parser edge cases (missing file, bad
JSON, missing keys, wrong types) must raise clean errors; `make lint` passes.

## Example usage

```
$ uv run python -m src
Resolving: 'Greet shrek'
  -> {'prompt': 'Greet shrek', 'name': 'fn_greet', 'parameters': {'name': 'shrek'}}
Wrote 11/11 results to data/output/function_calling_results.json
```

## Resources

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) ·
  [Constrained decoding (HF blog)](https://huggingface.co/blog/constrained-beam-search) ·
  [BPE tokenization](https://huggingface.co/learn/nlp-course/chapter6/5) ·
  [pydantic](https://docs.pydantic.dev/)
- **AI usage**: Claude assisted with the input parser, debugging tokenizer and
  quote-handling issues and reviewing the decoding logic. The
  decoding architecture was designed with a peer; all code is understood and owned by the author.
