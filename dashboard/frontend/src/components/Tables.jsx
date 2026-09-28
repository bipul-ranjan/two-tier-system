import { fmtConf, fmtMs, fmtPct, fmtUsd, fmtWhen, unitLabel } from "../format";

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
            <th className="num">Avg confidence</th>
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
              <td className="num">{fmtConf(r.avg_conf)}</td>
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
            <th className="num">Avg confidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.type}>
              <td>{r.type}</td>
              <td className="num">{r.rows}</td>
              <td><Meter pct={r.escalated_pct} /></td>
              <td className="num">{fmtConf(r.avg_conf)}</td>
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
                <span className={`pill ${r.decision === "LOCAL" ? "pill-local" : "pill-esc"}`}>
                  {r.decision === "LOCAL" ? "answered locally" : "sent to Claude"}
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
