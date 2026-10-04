import { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend } from "recharts";
import { C, unitColor } from "../theme";
import { fmtConf, fmtMs, fmtPct, fmtQuality, runLabel, runTick, unitLabel } from "../format";

// Quality-by-model metrics pull from quality_three_way (a run-level field, not nested under
// r.overall/r.units like the other metrics) and are already specific to one model each, so
// they render as a single line with no further per-business-unit breakdown.
const threeWayGetter = (bucket) => (r) => r.quality_three_way?.[bucket]?.avg ?? null;
const threeWayKnown = (bucket) => (r) => r.quality_three_way?.[bucket]?.known ?? 0;
const threeWayRows = (bucket) => (r) => r.quality_three_way?.[bucket]?.rows ?? 0;

const METRICS = [
  { key: "avg_conf", label: "Average confidence", domain: [0, 1], fmt: fmtConf, tick: (v) => v.toFixed(1) },
  { key: "avg_local_conf", label: "Local confidence", domain: [0, 1], fmt: fmtConf, tick: (v) => v.toFixed(1) },
  { key: "avg_claude_conf", label: "Claude confidence", domain: [0, 1], fmt: fmtConf, tick: (v) => v.toFixed(1) },
  {
    key: "quality_payments_slm", label: "Answer quality (Payment Assistant)", domain: [1, 5], fmt: fmtQuality, tick: (v) => v.toFixed(0),
    get: threeWayGetter("payments_slm"), known: threeWayKnown("payments_slm"), rows: threeWayRows("payments_slm"), noUnitBreakdown: true,
  },
  {
    key: "quality_retail_slm", label: "Answer quality (Retail Assistant)", domain: [1, 5], fmt: fmtQuality, tick: (v) => v.toFixed(0),
    get: threeWayGetter("retail_slm"), known: threeWayKnown("retail_slm"), rows: threeWayRows("retail_slm"), noUnitBreakdown: true,
  },
  {
    key: "quality_claude", label: "Answer quality (Escalate Claude)", domain: [1, 5], fmt: fmtQuality, tick: (v) => v.toFixed(0),
    get: threeWayGetter("claude"), known: threeWayKnown("claude"), rows: threeWayRows("claude"), noUnitBreakdown: true,
  },
  {
    key: "quality_cache", label: "Answer quality (Cache)", domain: [1, 5], fmt: fmtQuality, tick: (v) => v.toFixed(0),
    get: threeWayGetter("cache"), known: threeWayKnown("cache"), rows: threeWayRows("cache"), noUnitBreakdown: true,
    onlyIfData: true,  // hidden until a run has actually served something from the cache
  },
  { key: "local_pct", label: "Answered locally", domain: [0, 100], fmt: (v) => fmtPct(v, 0), tick: (v) => `${v}%` },
  { key: "median_total_ms", label: "Median time per query", domain: [0, "auto"], fmt: fmtMs, tick: (v) => `${Math.round(v / 1000)} s` },
];

const MODEL_LABEL_Y = 10;
const PROMPT_LABEL_Y = 24;

// Anchor a marker label away from the chart edge it's nearest to, so it grows inward instead
// of clipping off the side -- same fix as ResolutionTrend.jsx, needed here too since "new
// models" (and now "new Claude prompt") can land on the very last run, same as before.
function edgeAwareAnchor(index, total) {
  if (index >= total - 1) return "end";
  if (index <= 0) return "start";
  return "middle";
}

function markerLabel(text, color, y, index, total) {
  return (props) => (
    <text x={props.viewBox.x} y={y} textAnchor={edgeAwareAnchor(index, total)} fontSize={11} fill={color}>
      {text}
    </text>
  );
}

export default function TrendChart({ runs, selectedId }) {
  const [metric, setMetric] = useState(METRICS[0]);
  const units = metric.noUnitBreakdown ? [] : [...new Set(runs.flatMap((r) => Object.keys(r.units)))].sort();
  const data = runs.map((r, i) => ({
    id: r.run_id,
    overall: metric.get ? metric.get(r) : r.overall[metric.key],
    ...Object.fromEntries(units.map((u) => [u, r.units[u]?.[metric.key] ?? null])),
    modelsChanged: i > 0 && JSON.stringify(r.models) !== JSON.stringify(runs[i - 1].models),
    promptChanged: i > 0 && r.claude_prompt_version != null && r.claude_prompt_version !== runs[i - 1].claude_prompt_version,
  }));

  const known = metric.known ? runs.reduce((sum, r) => sum + metric.known(r), 0) : null;
  const rows = metric.rows ? runs.reduce((sum, r) => sum + metric.rows(r), 0) : null;

  return (
    <div>
      <div className="segmented" role="group" aria-label="Metric to compare">
        {METRICS.filter((m) => !m.onlyIfData || runs.some((r) => m.known(r) > 0)).map((m) => (
          <button key={m.key} aria-pressed={m.key === metric.key} onClick={() => setMetric(m)}>
            {m.label}
          </button>
        ))}
      </div>
      <div className="chart" style={{ height: 312 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 36, right: 30, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="id" padding={{ left: 24, right: 24 }} tickFormatter={runTick} tick={{ fontSize: 11, fill: C.muted }} stroke={C.grid} minTickGap={16} />
            <YAxis domain={metric.domain} tickFormatter={metric.tick} tick={{ fontSize: 12, fill: C.muted }} axisLine={false} tickLine={false} width={48} />
            {selectedId && <ReferenceLine x={selectedId} stroke={C.ink} strokeOpacity={0.1} strokeWidth={22} />}
            {data.filter((d) => d.modelsChanged).map((d) => (
              <ReferenceLine key={`m-${d.id}`} x={d.id} stroke={C.tier2} strokeDasharray="5 4"
                label={markerLabel("new models", C.tier2, MODEL_LABEL_Y, data.indexOf(d), data.length)} />
            ))}
            {data.filter((d) => d.promptChanged).map((d) => (
              <ReferenceLine key={`p-${d.id}`} x={d.id} stroke={C.claudePrompt} strokeDasharray="6 2"
                label={markerLabel("new Claude prompt", C.claudePrompt, PROMPT_LABEL_Y, data.indexOf(d), data.length)} />
            ))}
            <Tooltip
              isAnimationActive={false}
              labelFormatter={runLabel}
              formatter={(v, name) => [v == null ? "-" : metric.fmt(v), name === "overall" ? (metric.noUnitBreakdown ? metric.label : "All units") : unitLabel(name)]}
            />
            <Legend formatter={(n) => (n === "overall" ? (metric.noUnitBreakdown ? metric.label : "All units") : unitLabel(n))} />
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
      {metric.key === "avg_claude_conf" && runs.every((r) => !r.overall.claude_conf_known) && (
        <p className="hint">No run so far has a recorded Claude confidence. This needs the updated tier2_escalate.py, and only escalated queries get a value.</p>
      )}
      {metric.noUnitBreakdown && known === 0 && (
        <p className="hint">No run so far has answer-quality scores for this model. Run scripts/backfill_quality_scores.py (optionally --draft for the escalated-only comparison) to see this trend.</p>
      )}
      {metric.noUnitBreakdown && known > 0 && rows > 0 && known < rows && (
        <p className="hint">Quality is only scored for a sample of the eligible rows across these runs, not all of them -- treat this line as directional, not exact.</p>
      )}
    </div>
  );
}
