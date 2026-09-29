import TrendChart from "./TrendChart";
import { fmtConf, fmtPct, fmtUsd, runLabel } from "../format";

/** Weighted mean of `field` across runs, weighted by `weightField` -- not a plain average
 * of the per-run averages, which would be wrong when runs have different row counts. */
function weightedMean(runs, field, weightField) {
  let num = 0, den = 0;
  for (const r of runs) {
    const v = r.overall[field];
    const w = r.overall[weightField];
    if (v != null && w) {
      num += v * w;
      den += w;
    }
  }
  return den > 0 ? num / den : null;
}

function sum(runs, field) {
  return runs.reduce((acc, r) => acc + (r.overall[field] ?? 0), 0);
}

function Kpi({ label, value, sub }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export default function Overview({ runs }) {
  const totalRows = sum(runs, "rows");
  const totalLocal = sum(runs, "local");
  const totalEscalated = sum(runs, "escalated");
  const totalCost = sum(runs, "tier2_cost_usd");
  const totalClaudeKnown = sum(runs, "claude_conf_known");

  const avgConf = weightedMean(runs, "avg_conf", "rows");
  const avgLocalConf = weightedMean(runs, "avg_local_conf", "local");
  const avgClaudeConf = weightedMean(runs, "avg_claude_conf", "claude_conf_known");

  const first = runs[0], last = runs[runs.length - 1];
  const confDrift = first && last && first !== last && first.overall.avg_conf != null && last.overall.avg_conf != null
    ? last.overall.avg_conf - first.overall.avg_conf
    : null;

  return (
    <>
      <section className="section first">
        <p className="readout">
          Across <strong>{runs.length}</strong> {runs.length === 1 ? "run" : "runs"} and <strong>{totalRows}</strong> total queries,{" "}
          <strong>{fmtPct((totalLocal / totalRows) * 100)}</strong> were answered locally and{" "}
          <strong>{fmtPct((totalEscalated / totalRows) * 100)}</strong> went to Claude.
        </p>
        <p className="readout-sub">
          {avgLocalConf != null ? `Local answers averaged ${fmtConf(avgLocalConf)} confidence` : ""}
          {avgLocalConf != null && avgClaudeConf != null ? ", and " : ""}
          {avgClaudeConf != null ? `Claude self-reported ${fmtConf(avgClaudeConf)} confidence across the ${totalClaudeKnown} queries it answered` : ""}
          {avgLocalConf != null || avgClaudeConf != null ? ". " : ""}
          Total Claude spend so far is {fmtUsd(totalCost)}.
        </p>
      </section>

      <section className="section">
        <div className="kpi-row">
          <Kpi label="Total queries" value={totalRows} sub={`across ${runs.length} ${runs.length === 1 ? "run" : "runs"}`} />
          <Kpi label="Answered locally" value={fmtPct((totalLocal / totalRows) * 100)} sub={`${totalLocal} of ${totalRows}`} />
          <Kpi label="Overall confidence" value={fmtConf(avgConf)} sub="local + escalated combined" />
          <Kpi label="Local confidence" value={fmtConf(avgLocalConf)} sub="answers kept on-device" />
          <Kpi label="Claude confidence" value={fmtConf(avgClaudeConf)} sub={totalClaudeKnown ? `${totalClaudeKnown} self-rated answers` : "not recorded yet"} />
          <Kpi label="Total Claude spend" value={fmtUsd(totalCost)} sub="all runs combined" />
        </div>
        {confDrift != null && (
          <p className="hint">
            Average confidence has {confDrift >= 0 ? "risen" : "fallen"} by {Math.abs(confDrift).toFixed(3)} from the first run ({runLabel(first.run_id)}) to
            the latest ({runLabel(last.run_id)}).
          </p>
        )}
      </section>

      <section className="section">
        <h2>Trend across every run</h2>
        <p className="lede">The same chart as on a run's page, shown on its own here. A dashed marker shows where the local models changed.</p>
        <TrendChart runs={runs} selectedId={last?.run_id ?? null} />
      </section>
    </>
  );
}
