import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend } from "recharts";
import { C } from "../theme";
import { fmtPct, runLabel, runTick, unitLabel } from "../format";

const fmtThreshold = (t) => (t && t.length ? t.map((v) => v.toFixed(2)).join("/") : "-");

const MODEL_LABEL_Y = 10;
const THRESHOLD_LABEL_Y = 24;
const PROMPT_LABEL_Y = 38;

// Anchor a marker label away from the chart edge it's nearest to, so it grows inward instead
// of clipping off the side -- this specifically fixes a label on the last (or first) point
// rendering as e.g. "0.70 \u2192 0" because half the text extended past the plot area.
function edgeAwareAnchor(index, total) {
  if (index >= total - 1) return "end";
  if (index <= 0) return "start";
  return "middle";
}

function markerLabel(text, color, y, index, total) {
  return (props) => (
    <text x={props.viewBox.x} y={y} textAnchor={edgeAwareAnchor(index, total)} fontSize={10} fill={color}>
      {text}
    </text>
  );
}

/** For each business unit, a small chart of % answered locally vs % sent to Claude, across
 * every run, with the threshold actually in effect overlaid as its own step-line (on the
 * same 0-100 scale, so 0.70 lines up with the "70" gridline) -- a step line survives edge
 * clipping and reads at a glance in a way a text label at the last data point cannot. Model
 * changes get their own separate dashed marker, so a shift in the split is never wrongly
 * credited to the model when a threshold change (or vice versa) actually caused it. */
export default function ResolutionTrend({ runs, selectedId }) {
  const units = [...new Set(runs.flatMap((r) => Object.keys(r.units)))].sort();
  if (units.length === 0) return null;

  return (
    <div className="resolution-trend">
      {units.map((u) => {
        const data = runs.map((r, i) => {
          const prevModel = i > 0 ? runs[i - 1].models[u] : undefined;
          const thisModel = r.models[u];
          const thresholds = r.units[u]?.threshold ?? [];
          const thisThreshold = fmtThreshold(thresholds);
          const prevThreshold = i > 0 ? fmtThreshold(runs[i - 1].units[u]?.threshold) : null;
          const prevPromptVersion = i > 0 ? runs[i - 1].claude_prompt_version : undefined;
          const thisPromptVersion = r.claude_prompt_version;
          return {
            id: r.run_id,
            local_pct: r.units[u]?.local_pct ?? null,
            // escalated_pct ("needed escalation") is kept available for the tooltip/debugging
            // but is no longer what's plotted -- decision stays "ESCALATE" on a cache hit, so
            // this alone can't distinguish "served from cache" from "actually called Claude".
            // cache_pct + claude_pct (which sum to escalated_pct) are the two stacked areas now.
            escalated_pct: r.units[u]?.escalated_pct ?? null,
            cache_pct: r.units[u]?.cache_pct ?? null,
            claude_pct: r.units[u]?.claude_pct ?? null,
            cache_hit_rate: r.units[u]?.cache_hit_rate ?? null,
            avg_cache_similarity: r.units[u]?.avg_cache_similarity ?? null,
            model: thisModel,
            threshold: thisThreshold,
            thresholdPct: thresholds.length === 1 ? thresholds[0] * 100 : null,
            promptVersion: thisPromptVersion,
            modelChanged: i > 0 && thisModel !== undefined && thisModel !== prevModel,
            thresholdChanged: i > 0 && thisThreshold !== "-" && thisThreshold !== prevThreshold,
            // Shared across both panels, since Claude is one model either way (not per business
            // unit like modelChanged/thresholdChanged are) -- a Tier 2 prompt change shows up
            // identically on the Payments and Retail bank charts at the same run.
            promptChanged: i > 0 && thisPromptVersion != null && thisPromptVersion !== prevPromptVersion,
          };
        });
        return (
          <div key={u} className="resolution-trend-panel">
            <h3>
              {unitLabel(u)}
              {data.length > 0 && data[data.length - 1].model && <span className="model-tag"> {data[data.length - 1].model}</span>}
            </h3>
            <div className="chart" style={{ height: 256 }}>
              <ResponsiveContainer>
                <ComposedChart data={data} margin={{ top: 48, right: 20, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={C.grid} vertical={false} />
                  <XAxis dataKey="id" padding={{ left: 16, right: 16 }} tickFormatter={runTick} tick={{ fontSize: 11, fill: C.muted }} stroke={C.grid} minTickGap={16} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 12, fill: C.muted }} axisLine={false} tickLine={false} width={40} />
                  {selectedId && <ReferenceLine x={selectedId} stroke={C.ink} strokeOpacity={0.1} strokeWidth={18} />}
                  {data.filter((d) => d.modelChanged).map((d) => (
                    <ReferenceLine key={`m-${d.id}`} x={d.id} stroke={C.tier2} strokeDasharray="5 4"
                      label={markerLabel("new model", C.tier2, MODEL_LABEL_Y, data.indexOf(d), data.length)} />
                  ))}
                  {data.filter((d) => d.thresholdChanged).map((d) => (
                    <ReferenceLine key={`t-${d.id}`} x={d.id} stroke={C.exception} strokeOpacity={0.35} strokeDasharray="2 3"
                      label={markerLabel(d.threshold, C.exception, THRESHOLD_LABEL_Y, data.indexOf(d), data.length)} />
                  ))}
                  {data.filter((d) => d.promptChanged).map((d) => (
                    <ReferenceLine key={`p-${d.id}`} x={d.id} stroke={C.claudePrompt} strokeDasharray="6 2"
                      label={markerLabel("new Claude prompt", C.claudePrompt, PROMPT_LABEL_Y, data.indexOf(d), data.length)} />
                  ))}
                  <Tooltip
                    isAnimationActive={false}
                    labelFormatter={runLabel}
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="chart-tooltip">
                          <div className="chart-tooltip-title">{runLabel(label)}</div>
                          <div>answered locally: {fmtPct(d.local_pct, 0)}</div>
                          <div>served from cache: {fmtPct(d.cache_pct, 0)}</div>
                          <div>sent to Claude: {fmtPct(d.claude_pct, 0)}</div>
                          <div className="chart-tooltip-sub">model: {d.model ?? "-"}</div>
                          <div className="chart-tooltip-sub">threshold: {d.threshold}</div>
                          <div className="chart-tooltip-sub">Claude prompt: {d.promptVersion ?? "-"}</div>
                          {d.cache_hit_rate != null && (
                            <div className="chart-tooltip-sub">cache hit rate (of escalations): {fmtPct(d.cache_hit_rate, 0)}, avg similarity {d.avg_cache_similarity?.toFixed(3) ?? "-"}</div>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Legend formatter={(n) => (n === "local_pct" ? "answered locally" : n === "cache_pct" ? "served from cache" : n === "claude_pct" ? "sent to Claude" : "threshold")} />
                  <Area type="monotone" dataKey="local_pct" stackId="r" stroke={C.local} fill={C.local} fillOpacity={0.55} isAnimationActive={false} connectNulls />
                  <Area type="monotone" dataKey="cache_pct" stackId="r" stroke={C.cache} fill={C.cache} fillOpacity={0.55} isAnimationActive={false} connectNulls />
                  <Area type="monotone" dataKey="claude_pct" stackId="r" stroke={C.escalated} fill={C.escalated} fillOpacity={0.5} isAnimationActive={false} connectNulls />
                  <Line type="stepAfter" dataKey="thresholdPct" stroke={C.exception} strokeWidth={1.75} strokeDasharray="4 3" dot={false} activeDot={false} isAnimationActive={false} connectNulls legendType="line" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
