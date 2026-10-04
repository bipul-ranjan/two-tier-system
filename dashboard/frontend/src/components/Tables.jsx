import { fmtConf, fmtMs, fmtPct, fmtUsd, fmtWhen, runLabel, unitLabel } from "../format";

function Meter({ pct }) {
  return (
    <span className="meter-cell">
      <span className="meter" aria-hidden="true">
        <i style={{ width: `${Math.max(0, Math.min(100, pct ?? 0))}%` }} />
      </span>
      {fmtPct(pct)}
    </span>
  );
}

export function UnitTable({ rows }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Business unit</th>
            <th>Local model</th>
            <th className="num">Queries</th>
            <th>Answered locally</th>
            <th className="num">Local conf</th>
            <th className="num">Claude conf</th>
            <th className="num">Median time</th>
            <th className="num">Claude spend</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.unit}>
              <td>{unitLabel(r.unit)}</td>
              <td className="code">{r.model ?? "-"}</td>
              <td className="num">{r.rows}</td>
              <td><Meter pct={r.local_pct} /></td>
              <td className="num">{fmtConf(r.avg_local_conf)}</td>
              <td className="num">{r.claude_conf_known > 0 ? fmtConf(r.avg_claude_conf) : "-"}</td>
              <td className="num">{fmtMs(r.median_total_ms)}</td>
              <td className="num">{fmtUsd(r.tier2_cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ScenarioTable({ rows }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Type of query</th>
            <th className="num">Queries</th>
            <th>Sent to Claude</th>
            <th className="num">Local conf</th>
            <th className="num">Claude conf</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.type}>
              <td>{r.type}</td>
              <td className="num">{r.rows}</td>
              <td><Meter pct={r.claude_pct} /></td>
              <td className="num">{fmtConf(r.avg_local_conf)}</td>
              <td className="num">{r.claude_conf_known > 0 ? fmtConf(r.avg_claude_conf) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunsTable({ runs }) {
  const changed = runs.map((r, i) => i > 0 && JSON.stringify(r.models) !== JSON.stringify(runs[i - 1].models));
  const latestId = runs.length ? runs[runs.length - 1].run_id : null;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Run</th>
            <th className="num">Queries</th>
            <th>Answered locally</th>
            <th className="num">Local conf</th>
            <th className="num">Claude conf</th>
            <th className="num">Median time</th>
            <th className="num">Claude spend</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={r.run_id}>
              <td>
                {runLabel(r.run_id)}
                {r.run_id === latestId && <span className="tag">latest</span>}
                {changed[i] && <span className="tag tag-models">new models</span>}
              </td>
              <td className="num">{r.rows}</td>
              <td><Meter pct={r.overall.local_pct} /></td>
              <td className="num">{fmtConf(r.overall.avg_local_conf)}</td>
              <td className="num">{r.overall.claude_conf_known > 0 ? fmtConf(r.overall.avg_claude_conf) : "-"}</td>
              <td className="num">{fmtMs(r.overall.median_total_ms)}</td>
              <td className="num">{fmtUsd(r.overall.tier2_cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RecentTable({ rows }) {
  return (
    <div className="table-wrap">
      <table className="table recent">
        <thead>
          <tr>
            <th>Time</th>
            <th>Query</th>
            <th>Scenario</th>
            <th className="num">Confidence</th>
            <th>Outcome</th>
            <th className="num">Time taken</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="nowrap">{fmtWhen(r.start)}</td>
              <td className="query" title={r.query}>{r.query}</td>
              <td className="scenario">
                {(r.intent ?? r.category) && <span className="scenario-name">{r.intent ?? r.category}</span>}
                {r.exception && <span className="tag tag-exc">exception</span>}
                <span className="scenario-unit">{unitLabel(r.unit)}</span>
              </td>
              <td className="num">{fmtConf(r.conf)}</td>
              <td>
                <span className={`pill ${r.decision === "LOCAL" ? "pill-local" : r.decision === "CACHE" ? "pill-cache" : "pill-esc"}`}>
                  {r.decision === "LOCAL" ? "answered locally" : r.decision === "CACHE" ? "served from cache" : "sent to Claude"}
                </span>
              </td>
              <td className="num">{fmtMs(r.total_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
