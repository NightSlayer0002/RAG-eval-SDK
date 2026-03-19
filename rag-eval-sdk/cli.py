"""
cli.py — Command-line interface for the RAG Evaluation SDK.

Commands:
  rag-eval run                       Run the full evaluation pipeline
  rag-eval run --chat X --context Y  Run with custom data files
  rag-eval benchmark                 Run full benchmark (20 scenarios)
  rag-eval benchmark --quick         Run quick benchmark (12 scenarios, ~5 min)
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
    """Run benchmarks comparing Baseline vs RAGAS vs our SDK."""
    sys.path.insert(0, _PROJECT_ROOT)
    from benchmarks.run_benchmarks import run_all_benchmarks
    run_all_benchmarks(quick=args.quick)


def main():
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="RAG Evaluation SDK -- Novel conflict & position-bias detection",
    )
    parser.add_argument(
        "--version", action="version",
        version="rag-eval-sdk v1.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the evaluation pipeline")
    run_parser.add_argument("--chat", default=None, help="Path to chat.json")
    run_parser.add_argument("--context", default=None, help="Path to context.json")
    run_parser.add_argument("--output", default=None, help="Path to save report")
    run_parser.set_defaults(func=cmd_run)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark comparisons")
    bench_parser.add_argument(
        "--quick", action="store_true",
        help="Run quick mode (3 scenarios/category, ~5 minutes)"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
