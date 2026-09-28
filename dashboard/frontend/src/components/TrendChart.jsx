import { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend } from "recharts";
import { C, unitColor } from "../theme";
import { fmtConf, fmtMs, fmtPct, runLabel, runTick, unitLabel } from "../format";

const METRICS = [
  { key: "avg_conf", label: "Average confidence", domain: [0, 1], fmt: fmtConf, tick: (v) => v.toFixed(1) },
  { key: "local_pct", label: "Answered locally", domain: [0, 100], fmt: (v) => fmtPct(v, 0), tick: (v) => `${v}%` },
  { key: "median_total_ms", label: "Median time per query", domain: [0, "auto"], fmt: fmtMs, tick: (v) => `${Math.round(v / 1000)} s` },
];

export default function TrendChart({ runs, selectedId }) {
  const [metric, setMetric] = useState(METRICS[0]);
  const units = [...new Set(runs.flatMap((r) => Object.keys(r.units)))].sort();
  const data = runs.map((r, i) => ({
    id: r.run_id,
    overall: r.overall[metric.key],
    ...Object.fromEntries(units.map((u) => [u, r.units[u]?.[metric.key] ?? null])),
    modelsChanged: i > 0 && JSON.stringify(r.models) !== JSON.stringify(runs[i - 1].models),
  }));

  return (
    <div>
      <div className="segmented" role="group" aria-label="Metric to compare">
        {METRICS.map((m) => (
          <button key={m.key} aria-pressed={m.key === metric.key} onClick={() => setMetric(m)}>
            {m.label}
          </button>
        ))}
      </div>
      <div className="chart" style={{ height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 24, right: 30, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="id" padding={{ left: 24, right: 24 }} tickFormatter={runTick} tick={{ fontSize: 11, fill: C.muted }} stroke={C.grid} minTickGap={16} />
            <YAxis domain={metric.domain} tickFormatter={metric.tick} tick={{ fontSize: 12, fill: C.muted }} axisLine={false} tickLine={false} width={48} />
            {selectedId && <ReferenceLine x={selectedId} stroke={C.ink} strokeOpacity={0.1} strokeWidth={22} />}
            {data.filter((d) => d.modelsChanged).map((d) => (
              <ReferenceLine key={d.id} x={d.id} stroke={C.tier2} strokeDasharray="5 4"
                label={{ value: "new models", position: "top", fontSize: 11, fill: C.tier2 }} />
            ))}
            <Tooltip
              isAnimationActive={false}
              labelFormatter={runLabel}
              formatter={(v, name) => [v == null ? "-" : metric.fmt(v), name === "overall" ? "All units" : unitLabel(name)]}
            />
            <Legend formatter={(n) => (n === "overall" ? "All units" : unitLabel(n))} />
            <Line type="monotone" dataKey="overall" stroke={C.ink} strokeWidth={2.6} dot={{ r: 4 }} isAnimationActive={false} connectNulls />
            {units.map((u) => (
              <Line key={u} type="monotone" dataKey={u} stroke={unitColor(u)} strokeWidth={1.6} dot={{ r: 3 }} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {runs.length < 2 && (
        <p className="hint">There is only one run so far. Run the pipeline again, for example after retraining, to see a trend here.</p>
      )}
    </div>
  );
}
