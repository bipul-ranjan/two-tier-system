# results/plots/

Generated chart images — currently just the cost–accuracy trade-off
curve, produced by calling `evaluate.plot_tradeoff()` with results from
running the pipeline at several different confidence thresholds.

## Files

| File | Produced by |
|---|---|
| `tradeoff_curve.png` | `evaluate.plot_tradeoff(threshold_results)` |

This one isn't generated automatically by `python -m src.evaluate` on
its own — you need to run the pipeline multiple times at different
threshold values first (see `THRESHOLD` in `src/router.py`), collect
the accuracy/cost for each run, and pass that dictionary into
`plot_tradeoff()` yourself. This is the one piece of the evaluation
that's a manual sweep rather than a single automated step, since it
requires several full pipeline runs to produce one chart.

## Not tracked by Git

Same reasoning as the other `results/` subfolders — these are
regenerated outputs, not source material. If a specific plot needs to
go in your dissertation, copy it out into your write-up folder rather
than treating this as its permanent home.
