import TrendChart from "./TrendChart";
import ResolutionTrend from "./ResolutionTrend";
import { RunsTable } from "./Tables";
import { fmtConf, fmtPct, fmtQuality, fmtUsd, runLabel } from "../format";

/** Weighted mean + total known across runs for one quality_three_way bucket
 * ("payments_slm" | "retail_slm" | "claude"). */
function threeWaySummary(runs, key) {
  let num = 0, den = 0, known = 0;
  for (const r of runs) {
    const d = r.quality_three_way?.[key];
    if (d?.avg != null && d.known) {
      num += d.avg * d.known;
      den += d.known;
    }
    known += d?.known ?? 0;
  }
  return { avg: den > 0 ? num / den : null, known };
}

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
  // decision stays "ESCALATE" on a cache hit (Tier 1 still wasn't confident enough alone), so
  // `escalated` alone conflates "needed escalation" with "actually called Claude" -- these two
  // split it correctly. claude_calls is a strict correction of escalated (equal to it whenever
  // the cache isn't in use, since cache_hits is then always 0).
  const totalCacheHits = sum(runs, "cache_hits");
  const totalClaudeCalls = sum(runs, "claude_calls");

  const avgConf = weightedMean(runs, "avg_conf", "rows");
  const avgLocalConf = weightedMean(runs, "avg_local_conf", "local");
  const avgClaudeConf = weightedMean(runs, "avg_claude_conf", "claude_conf_known");
  const paymentsSlmQ = threeWaySummary(runs, "payments_slm");
  const retailSlmQ = threeWaySummary(runs, "retail_slm");
  const claudeQ = threeWaySummary(runs, "claude");
  const cacheQ = threeWaySummary(runs, "cache");

  const first = runs[0], last = runs[runs.length - 1];
  const confDrift = first && last && first !== last && first.overall.avg_conf != null && last.overall.avg_conf != null
    ? last.overall.avg_conf - first.overall.avg_conf
    : null;

  return (
    <>
      <section className="section first">
        <p className="readout">
          Across <strong>{runs.length}</strong> {runs.length === 1 ? "run" : "runs"} and <strong>{totalRows}</strong> total queries,{" "}
          <strong>{fmtPct((totalLocal / totalRows) * 100)}</strong> were answered locally
          {totalCacheHits > 0 ? <>, <strong>{fmtPct((totalCacheHits / totalRows) * 100)}</strong> were served from the cache</> : ""} and{" "}
          <strong>{fmtPct((totalClaudeCalls / totalRows) * 100)}</strong> actually went to Claude.
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
          {totalEscalated > 0 && (
            <Kpi label="Cache hit rate" value={fmtPct((totalCacheHits / totalEscalated) * 100)}
                 sub={`${totalCacheHits} of ${totalEscalated} escalation-eligible queries, avg similarity ${weightedMean(runs, "avg_cache_similarity", "cache_hits")?.toFixed(3) ?? "-"}`} />
          )}
        </div>
        <p className="lede" style={{ marginTop: 18 }}>
          Answer quality, split by which source actually produced the answer -- Payments and Retail bank only when they answered locally, Claude only on
          the fresh answers it wrote after an escalation{cacheQ.known > 0 ? ", and the cache on past Claude answers it served again" : ""}.
        </p>
        <div className="kpi-row">
          <Kpi label="Payments SLM quality" value={fmtQuality(paymentsSlmQ.avg)} sub={paymentsSlmQ.known ? `${paymentsSlmQ.known} answers, local only` : "not scored yet"} />
          <Kpi label="Retail bank SLM quality" value={fmtQuality(retailSlmQ.avg)} sub={retailSlmQ.known ? `${retailSlmQ.known} answers, local only` : "not scored yet"} />
          <Kpi label="Claude quality" value={fmtQuality(claudeQ.avg)} sub={claudeQ.known ? `${claudeQ.known} fresh Tier 2 answers` : "not scored yet"} />
          {cacheQ.known > 0 && (
            <Kpi label="Cache quality" value={fmtQuality(cacheQ.avg)} sub={`${cacheQ.known} cached answers re-served`} />
          )}
        </div>
        {confDrift != null && (
          <p className="hint">
            Average confidence has {confDrift >= 0 ? "risen" : "fallen"} by {Math.abs(confDrift).toFixed(3)} from the first run ({runLabel(first.run_id)}) to
            the latest ({runLabel(last.run_id)}).
          </p>
        )}
      </section>

      <section className="section">
        <h2>Local vs. Claude, by assistant</h2>
        <p className="lede">Each assistant's share of answers kept on-device vs. sent to Claude, across every run -- so a shift in either direction is easy to spot per assistant.</p>
        <ResolutionTrend runs={runs} selectedId={last?.run_id ?? null} />
      </section>

      <section className="section">
        <h2>Trend across every run</h2>
        <p className="lede">The same chart as on a run's page, shown on its own here. A dashed marker shows where the local models changed.</p>
        <TrendChart runs={runs} selectedId={last?.run_id ?? null} />
      </section>

      <section className="section">
        <h2>Every run, side by side</h2>
        <p className="lede">Oldest first, so rows line up with the chart above. "new models" flags the first run after a retrain.</p>
        <RunsTable runs={runs} />
      </section>
    </>
  );
}
