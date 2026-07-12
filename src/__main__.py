import argparse
import json
import sys
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.GenerationPipeline import GenerationPipeline
from src.Parser import ParseError, Parser


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments, with defaults matching the data/ layout."""
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to write the resolved function calls to.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: resolve each prompt to a function call and write it."""
    try:
        args = parse_args()
        try:
            parser = Parser()
            functions = parser.parse_def(args.functions_definition)
            prompts = parser.parse_call(args.input)
        except ParseError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print("Loading model...")
        llm = Small_LLM_Model()
        results = []
        for entry in prompts:
            print(f"Resolving: {entry.prompt!r}")
            pipeline = GenerationPipeline(
                model=llm,
                prompt=entry,
                functions_definition=functions,
            )
            try:
                pipeline.generate_output()
            except Exception as e:
                print(f"  skipped, could not resolve: {e}")
                continue
            results.append(pipeline.output)
            print(f"  -> {pipeline.output}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)}/{len(prompts)} results to {output_path}")
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
