import { fmtConf, fmtPct, runLabel } from "../format";

// The runs, newest first. Each row shows the run's average confidence so runs can be compared at a glance,
// and flags the first run that used different models (for example after retraining).
export default function RunLedger({ runs, selectedId, onPick, onOverview, overviewSelected }) {
  const latestId = runs.length ? runs[runs.length - 1].run_id : null;
  const changed = runs.map((r, i) => i > 0 && JSON.stringify(r.models) !== JSON.stringify(runs[i - 1].models));
  const routerNew = runs.map((r, i) => i > 0 && r.router !== runs[i - 1].router);   // a retrained router, or a switch of router

  return (
    <nav className="ledger" aria-label="Runs">
      <button className={`overview-link${overviewSelected ? " is-selected" : ""}`} aria-current={overviewSelected} onClick={onOverview}>
        Overview across all runs
      </button>
      <h2 className="ledger-title">Runs</h2>
      <p className="ledger-note">Newest first. The bar is each run's average confidence.</p>
      <ul>
        {runs
          .map((r, i) => ({ r, isChanged: changed[i], isRouterNew: routerNew[i] }))
          .reverse()
          .map(({ r, isChanged, isRouterNew }) => {
            const selected = !overviewSelected && r.run_id === selectedId;
            const conf = r.overall.avg_conf;
            return (
              <li key={r.run_id}>
                <button className={`run${selected ? " is-selected" : ""}`} aria-current={selected} onClick={() => onPick(r.run_id)}>
                  <span className="run-head">
                    <span className="run-name">{runLabel(r.run_id)}</span>
                    {r.run_id === latestId && <span className="tag">latest</span>}
                    {isChanged && <span className="tag tag-models">new models</span>}
                    {isRouterNew && <span className="tag tag-router">new router</span>}
                  </span>
                  <span className="run-meta">
                    {r.rows} queries, {fmtPct(r.overall.local_pct)} answered locally
                  </span>
                  <span className="run-conf">
                    <span className="bar" aria-hidden="true">
                      <i style={{ width: `${Math.max(0, Math.min(1, conf ?? 0)) * 100}%` }} />
                    </span>
                    <b>{fmtConf(conf)}</b>
                  </span>
                </button>
              </li>
            );
          })}
      </ul>
    </nav>
  );
}
