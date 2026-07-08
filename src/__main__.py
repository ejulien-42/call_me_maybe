import sys

from llm_sdk import Small_LLM_Model

from src.Parser import ParseError, Parser
from src.Pipeline import Pipeline, PipelineError


def main() -> None:
    """Entry point: resolve each test prompt to a full function call."""
    try:
        parser = Parser()
        functions = parser.parse_def(
            "data/input/functions_definition.json"
        )
        prompts = parser.parse_call(
            "data/input/function_calling_tests.json"
        )
    except ParseError as e:
        print(f"Error: {e}")
        sys.exit(1)

    functions_by_name = {function.name: function for function in functions}

    print("Loading model...")
    llm = Small_LLM_Model()
    pipeline = Pipeline(llm, functions)

    for entry in prompts:
        print(f"\nPrompt: {entry.prompt}")
        try:
            name = pipeline.get_name(entry.prompt)
            args = pipeline.get_args(entry.prompt, functions_by_name[name])
        except PipelineError as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  name: {name}")
        print(f"  parameters: {args}")


if __name__ == "__main__":
    main()
