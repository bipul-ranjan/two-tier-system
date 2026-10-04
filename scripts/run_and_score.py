"""
Runs the main pipeline, then automatically backfills BOTH quality scores -- final-answer
quality (quality_overall, the default backfill mode) and draft quality (quality_draft_overall,
the Tier 1 discarded-draft comparison on escalated rows, via --draft) -- so neither is a
separate step you have to remember to run by hand after every batch.

Each step shells out to the actual script you'd run manually (src.pipeline,
scripts/backfill_quality_scores.py), so there's no duplicated logic to drift out of sync with
those scripts as they change -- this is purely a sequencing wrapper.

Run from the project root, with ANTHROPIC_API_KEY set and Ollama running:
    python scripts/run_and_score.py 100
    python scripts/run_and_score.py 100 --use-semantic-cache
Any arguments after the row count are passed straight through to src.pipeline (--score-quality,
--score-quality-local, --use-semantic-cache, etc. -- see src/pipeline.py's own docstring for
the full list). The two backfill steps always run with no extra filters (--unit/--run/--limit),
scoring whatever was left blank by the run that just happened, plus anything blank from before.
"""
import subprocess
import sys


def run_step(cmd: list, label: str):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n{label} exited with code {result.returncode} -- stopping here rather than "
              f"scoring on top of what might be an incomplete run. Fix the issue above, then "
              f"either re-run this script or run the remaining steps individually.")
        sys.exit(result.returncode)


def main():
    pipeline_args = sys.argv[1:]
    if not pipeline_args or not pipeline_args[0].lstrip("-").isdigit():
        print("Usage: python scripts/run_and_score.py <n> [--use-semantic-cache] [other src.pipeline flags]")
        sys.exit(1)

    run_step(["-m", "src.pipeline"] + pipeline_args, "Step 1/3: Running the pipeline")
    run_step(["scripts/backfill_quality_scores.py"], "Step 2/3: Backfilling answer quality (final answer)")
    run_step(["scripts/backfill_quality_scores.py", "--draft"], "Step 3/3: Backfilling draft quality (Tier 1's discarded draft, escalated rows only)")

    print(f"\n{'=' * 60}\nDone: pipeline run complete, both quality_overall and quality_draft_overall backfilled.\n{'=' * 60}")


if __name__ == "__main__":
    main()
