# results/plots/

Chart images. The only one the code can draw is a cost-against-accuracy curve, `tradeoff_curve.png`, made by
`evaluate.plot_tradeoff(threshold_results)` from a dictionary you build yourself (`{pass mark: (accuracy, cost per 1,000 queries)}`).
Nothing calls it automatically, and nothing here is tracked by Git. The live charts are on the dashboard (`dashboard/README.md`).
