"""
cli.py — Command-line interface for the RAG Evaluation SDK.

Commands:
  rag-eval run                       Run the full evaluation pipeline
  rag-eval run --chat X --context Y  Run with custom data files
  rag-eval benchmark                 Show the reproducible benchmark commands
  rag-eval --version                 Show version
"""

import argparse
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_run(args):
    """Run the full evaluation pipeline."""
    from .main import run_evaluation_pipeline
    run_evaluation_pipeline(
        chat_path=args.chat,
        context_path=args.context,
        output_path=args.output,
    )


def cmd_benchmark(args):
    """Do not silently run the legacy pseudo-RAGAS benchmark."""
    print(
        "Use `rag-eval-ragtruth --help` for the labelled detector benchmark or "
        "`rag-eval-rgb --help` for the RGB generator stress test. The old synthetic "
        "benchmark was removed; see RESULTS_AUDIT.md for its historical limitations."
    )


def main():
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Auditable, budget-aware RAG verification",
    )
    parser.add_argument(
        "--version", action="version",
        version="rag-eval-sdk v2.0.0a2"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the evaluation pipeline")
    run_parser.add_argument("--chat", default=None, help="Path to chat.json")
    run_parser.add_argument("--context", default=None, help="Path to context.json")
    run_parser.add_argument("--output", default=None, help="Path to save report")
    run_parser.set_defaults(func=cmd_run)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Show v2 benchmark commands")
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
