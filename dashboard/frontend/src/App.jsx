import { useState } from "react";
import { useLiveData } from "./useLiveData";
import RunLedger from "./components/RunLedger";
import ConfidenceStrip from "./components/ConfidenceStrip";
import TrendChart from "./components/TrendChart";
import LatencyChart from "./components/LatencyChart";
import Overview from "./components/Overview";
import { UnitTable, ScenarioTable, RecentTable } from "./components/Tables";
import { fmtClock, fmtConf, fmtMs, fmtUsd, fmtWhen, runLabel, unitLabel } from "./format";

function Message({ title, children }) {
  return (
    <div className="empty">
      <h1>{title}</h1>
      {children}
    </div>
  );
}

function Readout({ k, thresholds }) {
  const against =
    thresholds.length === 1
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
              <Readout k={d.kpis} thresholds={d.thresholds} />
              <p className="runmeta">
                Run {runLabel(d.run_id)}
                {d.started ? `, started ${fmtWhen(d.started)}` : ""}.{" "}
                {Object.entries(d.models).map(([u, m]) => `${unitLabel(u)} used ${m ?? "an unknown model"}`).join(", ")}.
              </p>
            </section>

            <section className="section">
              <h2>Where each answer landed</h2>
              <p className="lede">
                Each dot is one query, placed by the confidence of its local answer. Queries that fall left of the threshold line are sent to Claude.
              </p>
              <ConfidenceStrip points={d.points} thresholds={d.thresholds} />
              {d.thresholds.length === 0 && (
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
                <p className="lede">Local answers wait only for the local model. Queries sent to Claude wait for both.</p>
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
                <li>A query is answered locally when its confidence is at or above the threshold. Otherwise it is sent to Claude.</li>
                <li>Time is measured from the start of the query to the final answer, including the Claude call when there is one.</li>
                <li>Claude spend is estimated from token counts and the prices set in tier2_escalate.py.</li>
                <li>The page reads the pipeline logs directly and refreshes every five seconds, so it follows a run while it is in progress.</li>
              </ul>
            </details>
          </>
        )}
      </main>
    </div>
  );
}
