import { useState } from "react";
import { useLiveData } from "./useLiveData";
import RunLedger from "./components/RunLedger";
import ConfidenceStrip from "./components/ConfidenceStrip";
import TrendChart from "./components/TrendChart";
import LatencyChart from "./components/LatencyChart";
import Overview from "./components/Overview";
import QualityBreakdown from "./components/QualityBreakdown";
import { UnitTable, ScenarioTable, RecentTable } from "./components/Tables";
import { fmtClock, fmtConf, fmtMs, fmtQuality, fmtUsd, fmtWhen, runLabel, unitLabel } from "./format";

function Message({ title, children }) {
  return (
    <div className="empty">
      <h1>{title}</h1>
      {children}
    </div>
  );
}

function Readout({ k, thresholds, router = "threshold", cutoffs = [] }) {
  const against =
    router === "learned"
      ? cutoffs.length ? `, and the learned router escalated any answer it predicted would score below ${cutoffs.map((t) => t.toFixed(2)).join(" or ")}` : ", decided by the learned router"
      : thresholds.length === 1
      ? ` against a threshold of ${thresholds[0].toFixed(2)}`
      : thresholds.length > 1
      ? ` against thresholds of ${thresholds.map((t) => t.toFixed(2)).join(" and ")}`
      : "";
  const timing = [];
  if (k.median_local_ms != null) timing.push(`Answers that stayed local took a median ${fmtMs(k.median_local_ms)}`);
  if (k.median_escalated_ms != null) timing.push(`queries sent to Claude took ${fmtMs(k.median_escalated_ms)} in total`);
  const split = [];
  if (k.avg_local_conf != null) split.push(`answers kept locally averaged ${fmtConf(k.avg_local_conf)} confidence`);
  if (k.avg_claude_conf != null) split.push(`Claude self-reported ${fmtConf(k.avg_claude_conf)} confidence on the ${k.claude_conf_known} ${k.claude_conf_known === 1 ? "query it" : "queries it"} answered`);
  if (k.avg_quality != null) split.push(`answer quality (Claude-judged) averaged ${fmtQuality(k.avg_quality)} across ${k.quality_known} ${k.quality_known === 1 ? "answer" : "answers"}`);
  if (k.avg_quality_local != null) split.push(`answer quality (local-judged) averaged ${fmtQuality(k.avg_quality_local)} across ${k.quality_local_known} ${k.quality_local_known === 1 ? "answer" : "answers"}`);
  return (
    <>
      <p className="readout">
        <strong>{k.local}</strong> of <strong>{k.rows}</strong> queries were answered by the local models, and <strong>{k.escalated}</strong> were sent
        to Claude. Average confidence was <strong>{fmtConf(k.avg_conf)}</strong>
        {against}.
      </p>
      <p className="readout-sub">
        {split.length ? `${split.join(", and ")}. ` : ""}
        {timing.length ? `${timing.join("; ")}. ` : ""}Claude spend for this run was {fmtUsd(k.tier2_cost_usd)}.
      </p>
    </>
  );
}

export default function App() {
  const runsQ = useLiveData("/api/runs");
  const [picked, setPicked] = useState(null); // null = follow the newest run
  const [overview, setOverview] = useState(true); // land on the cross-run overview first

  const runs = runsQ.data?.runs ?? [];
  const latestId = runs.length ? runs[runs.length - 1].run_id : null;
  const runId = picked && runs.some((r) => r.run_id === picked) ? picked : latestId;
  const detailQ = useLiveData(!overview && runId ? `/api/run?run_id=${encodeURIComponent(runId)}` : null);
  const d = detailQ.data;
  const pick = (id) => {
    setOverview(false);
    setPicked(id === latestId ? null : id);
  };

  if (runsQ.error && !runsQ.data) {
    return (
      <Message title="Cannot reach the dashboard backend">
        <p>Start it from the project root, then reload this page:</p>
        <pre><code>python -m uvicorn dashboard.backend.main:app --port 8000</code></pre>
      </Message>
    );
  }
  if (!runsQ.data) return <Message title="Loading">Reading the run logs.</Message>;
  if (runs.length === 0) {
    return (
      <Message title="No results yet">
        <p>Run the pipeline from the project root and this page fills in by itself:</p>
        <pre><code>python -m src.pipeline 10</code></pre>
        <p className="hint">It reads results/logs/results_history.csv, or results_log_combined.csv if there is no history yet.</p>
      </Message>
    );
  }

  const live = !runsQ.error;
  return (
    <div className="shell">
      <RunLedger runs={runs} selectedId={runId} onPick={pick} onOverview={() => setOverview(true)} overviewSelected={overview} />

      <main className="main">
        <header className="top">
          <div>
            <h1>Two-tier system</h1>
            <p className="tagline">Local models answer first. Claude takes the queries they are unsure about.</p>
          </div>
          <div className={`status ${live ? "is-live" : "is-down"}`} role="status">
            <i />
            {live ? `Live, updated ${runsQ.updated ? fmtClock(runsQ.updated) : ""}` : "Backend not reachable"}
          </div>
        </header>

        {!overview && runsQ.data.source === "latest_only" && (
          <p className="notice">
            Only the latest run is available, so there is nothing to compare yet. Run history builds up from your next pipeline run.
          </p>
        )}

        {overview ? (
          <Overview runs={runs} />
        ) : !d ? (
          <p className="hint">Loading this run.</p>
        ) : (
          <>
            <section className="section first">
              <Readout k={d.kpis} thresholds={d.thresholds} router={d.router_kind} cutoffs={d.cutoffs} />
              <p className="runmeta">
                Run {runLabel(d.run_id)}
                {d.started ? `, started ${fmtWhen(d.started)}` : ""}.{" "}
                {Object.entries(d.models).map(([u, m]) => `${unitLabel(u)} used ${m ?? "an unknown model"}`).join(", ")}.
              </p>
            </section>

            <section className="section">
              <h2>Where each answer landed</h2>
              <p className="lede">
                {d.router_kind === "learned"
                  ? "Each dot is one query, placed by the quality the router predicted for its local answer. Answers that fall left of the cut-point line are escalated: served from the cache when a close past answer exists, otherwise sent to Claude."
                  : "Each dot is one query, placed by the confidence of its local answer. Queries that fall left of the threshold line are escalated: served from the cache when a close past answer exists, otherwise sent to Claude."}
              </p>
              <ConfidenceStrip points={d.points} thresholds={d.thresholds} router={d.router_kind} cutoffs={d.cutoffs} />
              {d.router_kind !== "learned" && d.thresholds.length === 0 && (
                <p className="hint">This run did not record its threshold, so no line is drawn.</p>
              )}
            </section>

            <div className="split">
              <section className="section">
                <h2>Across runs</h2>
                <p className="lede">Compare runs, for example before and after retraining. A dashed marker shows where the local models changed.</p>
                <TrendChart runs={runs} selectedId={runId} />
              </section>
              <section className="section">
                <h2>Time per query</h2>
                <p className="lede">Local answers wait only for the local model. Queries sent to Claude wait for both. Cache hits wait only for the local model and the lookup.</p>
                <LatencyChart latency={d.latency} />
                {d.kpis.load_known > 0 && (
                  <p className="hint">
                    {d.kpis.cold_starts > 0
                      ? `${d.kpis.cold_starts} ${d.kpis.cold_starts === 1 ? "query" : "queries"} in this run waited over a second for a model to load, which inflates those times.`
                      : "No query in this run waited on a model load."}
                  </p>
                )}
              </section>
            </div>

            <section className="section">
              <h2>By business unit</h2>
              <UnitTable rows={d.by_unit} />
            </section>

            {d.quality_three_way && (d.quality_three_way.payments_slm.known > 0 || d.quality_three_way.retail_slm.known > 0 || d.quality_three_way.claude.known > 0 || d.quality_three_way.cache.known > 0) && (
              <section className="section">
                <h2>Answer quality, by model</h2>
                <p className="lede">
                  Each model's own answers only -- SLMs on what they kept local, Claude on what it was escalated. Not an equal-footing comparison: Claude
                  only ever sees the harder queries each SLM wasn't confident about.
                </p>
                <div className="kpi-row">
                  <div className="kpi">
                    <div className="kpi-value">{fmtQuality(d.quality_three_way.payments_slm.avg)}</div>
                    <div className="kpi-label">Payments SLM</div>
                    <div className="kpi-sub">{d.quality_three_way.payments_slm.known} local answers</div>
                  </div>
                  <div className="kpi">
                    <div className="kpi-value">{fmtQuality(d.quality_three_way.retail_slm.avg)}</div>
                    <div className="kpi-label">Retail bank SLM</div>
                    <div className="kpi-sub">{d.quality_three_way.retail_slm.known} local answers</div>
                  </div>
                  <div className="kpi">
                    <div className="kpi-value">{fmtQuality(d.quality_three_way.claude.avg)}</div>
                    <div className="kpi-label">Claude</div>
                    <div className="kpi-sub">{d.quality_three_way.claude.known} fresh Tier 2 answers</div>
                  </div>
                  {d.quality_three_way.cache.known > 0 && (
                    <div className="kpi">
                      <div className="kpi-value">{fmtQuality(d.quality_three_way.cache.avg)}</div>
                      <div className="kpi-label">Cache</div>
                      <div className="kpi-sub">{d.quality_three_way.cache.known} cached answers re-served</div>
                    </div>
                  )}
                </div>
              </section>
            )}

            {(d.kpis.quality_known > 0 || d.kpis.quality_local_known > 0) && (
              <section className="section">
                <h2>Answer quality, by dimension</h2>
                <p className="lede">1-5 per dimension. Claude and local are separate judges, shown side by side -- never averaged together.</p>
                <QualityBreakdown
                  byDim={d.kpis.quality_by_dim} known={d.kpis.quality_known}
                  byDimLocal={d.kpis.quality_local_by_dim} knownLocal={d.kpis.quality_local_known}
                  rows={d.kpis.rows}
                />
              </section>
            )}

            {d.scenarios.length > 0 && (
              <section className="section">
                <h2>Normal and exception queries</h2>
                <p className="lede">Exceptions are the fraud, complaint, bereavement and similar scenarios in the synthetic data.</p>
                <ScenarioTable rows={d.scenarios} />
              </section>
            )}

            <section className="section">
              <h2>Latest queries in this run</h2>
              <RecentTable rows={d.recent} />
            </section>

            <details className="method">
              <summary>How these numbers are calculated</summary>
              <ul>
                <li>Confidence is the geometric mean of the probabilities of the tokens in the local model's answer, taken from Ollama's log-probabilities. It shows how sure the model was of its own wording, not whether the answer is correct.</li>
                <li>The router decides after the local model has answered. The learned router reads the question, the local answer, its confidence and the business unit, predicts how good the answer is (1-5), and escalates it when the prediction is below its cut-point. The older rule escalates when confidence is below a threshold. Either way an escalated query is re-served from the cache if a close enough past Claude answer exists, and only sent to Claude if not.</li>
                <li>Time is measured from the start of the query to the final answer, including the Claude call when there is one.</li>
                <li>Claude spend is estimated from token counts and the prices set in tier2_escalate.py.</li>
                <li>Answer quality (1-5) is a separate, offline measurement from confidence: Claude scores the actual answer text against correctness, completeness, tone, safety and clarity, after the run finishes -- automatically on every run, unless it was started with --noquality. Older rows, and rows from --noquality runs, only have a score once scripts/backfill_quality_scores.py has been run on them.</li>
                <li>The page reads the pipeline logs directly and refreshes every five seconds, so it follows a run while it is in progress.</li>
              </ul>
            </details>
          </>
        )}
      </main>
    </div>
  );
}
