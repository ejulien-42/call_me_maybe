import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ValidationError, field_validator

ParamType = Literal["string", "number", "integer",
                    "boolean", "float"]


class ParseError(Exception):
    """Raised when a function-definition file cannot be read or is invalid."""


class ParameterDefinition(BaseModel):
    """
    Schema of a single function parameter.
    """

    type: ParamType


class ReturnDefinition(BaseModel):
    """Schema of a function's return value.
    """

    type: ParamType


class FunctionDefinition(BaseModel):
    """
    A single callable function exposed to the LLM.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("name", "description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """
        Reject empty or whitespace-only strings.
        """
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PromptEntry(BaseModel):
    """
    A single natural-language request to resolve into a function call.
    """

    prompt: str

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """
        Reject empty or whitespace-only prompts.
        """
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class Parser:
    """Loads and validates the JSON input files used by the program."""
    def parse_def(self, filename: str) -> list[FunctionDefinition]:
        """
        Load and validate a function-definition JSON file.
        """
        data = self._load_json(filename)
        if not isinstance(data, list):
            raise ParseError(
                f"Expected a JSON array of function definitions in "
                f"'{filename}', got {type(data).__name__}."
            )
        functions: list[FunctionDefinition] = []
        seen_names: set[str] = set()
        for index, entry in enumerate(data):
            try:
                function = FunctionDefinition.model_validate(entry)
            except ValidationError as exc:
                raise ParseError(
                    f"Invalid function definition at index {index} in "
                    f"'{filename}': {exc}"
                ) from exc
            if function.name in seen_names:
                raise ParseError(
                    f"Duplicate function name "
                    f"'{function.name}' in '{filename}'."
                )
            seen_names.add(function.name)
            functions.append(function)
        return functions

    def parse_call(self, filename: str) -> list[PromptEntry]:
        """
        Load and validate a function-calling-tests JSON file.
        """
        data = self._load_json(filename)
        if not isinstance(data, list):
            raise ParseError(
                f"Expected a JSON array of prompts in "
                f"'{filename}', got {type(data).__name__}."
            )
        prompts: list[PromptEntry] = []
        for index, entry in enumerate(data):
            try:
                prompts.append(PromptEntry.model_validate(entry))
            except ValidationError as exc:
                raise ParseError(
                    f"Invalid prompt entry at index {index} in "
                    f"'{filename}': {exc}"
                ) from exc
        return prompts

    @staticmethod
    def _load_json(filename: str) -> object:
        """
        Read and parse a JSON file, wrapping errors in ``ParseError``.
        """
        path = Path(filename)
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise ParseError(f"File not found: '{filename}'.") from exc
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON in '{filename}': {exc}") from exc
        except OSError as exc:
            raise ParseError(f"Could not read '{filename}': {exc}") from exc
        except Exception as e:
            raise ParseError(e)
