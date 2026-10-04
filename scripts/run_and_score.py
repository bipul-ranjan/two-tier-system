"""
Runs the pipeline, then runs the two quality backfills afterwards.

`python -m src.pipeline` now evaluates quality by default (every row's final answer -- Tier 1,
cache, or Tier 2 -- gets one Claude judge call after processing), so step 2 below normally
finds nothing blank and costs nothing; it exists as a safety net for any row whose judge call
failed, or any older rows that were never scored. Step 3 is the one that still adds something
new every time: scoring Tier 1's discarded draft on escalated rows (quality_draft_overall),
the same-query comparison against Claude's actual answer.

Each step shells out to the actual script you'd run manually (src.pipeline,
scripts/backfill_quality_scores.py), so there's no duplicated logic to drift out of sync --
this is purely a sequencing wrapper.

Run from the project root, with ANTHROPIC_API_KEY set and Ollama running:
    python scripts/run_and_score.py 100
    python scripts/run_and_score.py 100 --nocache
Any arguments after the row count are passed straight through to src.pipeline (--nocache,
--noquality, --score-quality-local <model>; see src/pipeline.py for the full list).
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
        print("Usage: python scripts/run_and_score.py <n> [--nocache] [--noquality] [other src.pipeline flags]")
        sys.exit(1)

    run_step(["-m", "src.pipeline"] + pipeline_args, "Step 1/3: Running the pipeline")
    run_step(["scripts/backfill_quality_scores.py"], "Step 2/3: Catch-up backfill of final-answer quality (normally nothing left blank)")
    run_step(["scripts/backfill_quality_scores.py", "--draft"], "Step 3/3: Backfilling draft quality (Tier 1's discarded draft, escalated rows only)")

    print(f"\n{'=' * 60}\nDone: pipeline run complete, both quality_overall and quality_draft_overall backfilled.\n{'=' * 60}")


if __name__ == "__main__":
    main()
