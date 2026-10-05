"""
Runs the pipeline, then runs both quality backfills as a safety net.

`python -m src.pipeline` already does all the quality work by default: every row's final answer
is judged, every escalated row's discarded Tier 1 draft is judged, and local rows copy their
final-answer scores onto the draft columns -- so no quality cell is left blank. Steps 2 and 3
below therefore normally find nothing to do and cost nothing; they exist to catch anything a run
couldn't finish (a failed judge call, a Ctrl+C partway through scoring) and any older rows that
were never scored.

Each step shells out to the actual script you'd run manually (src.pipeline,
scripts/backfill_quality_scores.py), so there's no duplicated logic to drift out of sync --
this is purely a sequencing wrapper.

Run from the project root, with ANTHROPIC_API_KEY set and Ollama running:
    python scripts/run_and_score.py 100
    python scripts/run_and_score.py 100 --nocache
Any arguments after the row count are passed straight through to src.pipeline (--nocache,
--noquality, --threshold-router, --score-quality-local <model>; see src/pipeline.py for the full list).
"""
import subprocess
import sys


def run_step(cmd: list, label: str):
    """Run one Python command under a banner, and stop the whole script if it fails, rather than
    scoring on top of a run that did not finish.
    """
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n{label} exited with code {result.returncode} -- stopping here rather than "
              f"scoring on top of what might be an incomplete run. Fix the issue above, then "
              f"either re-run this script or run the remaining steps individually.")
        sys.exit(result.returncode)


def main():
    """Check that the first argument is a number of questions, then run the pipeline followed by the
    two safety-net quality backfills.
    """
    pipeline_args = sys.argv[1:]
    if not pipeline_args or not pipeline_args[0].lstrip("-").isdigit():
        print("Usage: python scripts/run_and_score.py <n> [--nocache] [--noquality] [other src.pipeline flags]")
        sys.exit(1)

    run_step(["-m", "src.pipeline"] + pipeline_args, "Step 1/3: Running the pipeline")
    run_step(["scripts/backfill_quality_scores.py"], "Step 2/3: Safety-net backfill of final-answer quality (normally nothing left blank)")
    run_step(["scripts/backfill_quality_scores.py", "--draft"], "Step 3/3: Safety-net backfill of Tier 1 draft quality (local rows copy, escalated rows judged)")

    print(f"\n{'=' * 60}\nDone: pipeline run complete, both quality_overall and quality_draft_overall backfilled.\n{'=' * 60}")


if __name__ == "__main__":
    main()
