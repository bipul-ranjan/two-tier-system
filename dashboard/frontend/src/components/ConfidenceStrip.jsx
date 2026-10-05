import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ReferenceArea, ReferenceLine, CartesianGrid } from "recharts";
import { C } from "../theme";
import { fmtConf, unitLabel } from "../format";

const TICKS = Array.from({ length: 11 }, (_, i) => i / 10);
const QUALITY_TICKS = [1, 2, 3, 4, 5];

function StripTip({ active, payload, learned }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const outcome = p.d === "LOCAL" ? "answered locally" : p.d === "CACHE" ? "served from cache" : "sent to Claude";
  return (
    <div className="tip">
      <div className="tip-title">{p.q}</div>
      <div>
        {unitLabel(p.u)}
        {p.i ? `, ${p.i}` : ""}
        {p.x ? ", exception" : ""}
      </div>
      <div>
        {learned ? `Predicted quality ${p.p?.toFixed(2)}` : `Confidence ${fmtConf(p.c)}`}: {outcome}
      </div>
    </div>
  );
}

// One dot per query. Under the confidence-threshold router it is placed by the confidence of its local
// answer and the line is the threshold. Under the learned router it is placed by the quality the router
// PREDICTED for the answer and the line is the router's cut-point: answers predicted below it are escalated.
export default function ConfidenceStrip({ points, thresholds, router = "threshold", cutoffs = [] }) {
  const learned = router === "learned";
  const xKey = learned ? "p" : "c";
  const marks = learned ? cutoffs : thresholds;
  const lo = learned ? 1 : 0;
  const hi = learned ? 5 : 1;
  const units = [...new Set(points.map((p) => p.u))].sort();
  const lane = Object.fromEntries(units.map((u, i) => [u, units.length - i])); // first unit on the top lane
  // spread dots vertically inside their lane so they do not sit on top of each other (same spread every render)
  const data = points.filter((p) => p[xKey] != null).map((p, i) => ({ ...p, y: lane[p.u] + (((i * 7919) % 1000) / 1000 - 0.5) * 0.7 }));
  const normal = data.filter((d) => !d.x);
  const exceptions = data.filter((d) => d.x);
  const single = marks.length === 1 ? marks[0] : null;

  return (
    <div>
      <div className="chart" style={{ height: 110 + units.length * 96 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 30, right: 24, bottom: 6, left: 4 }}>
            {single != null && <ReferenceArea x1={lo} x2={single} fill={C.escalated} fillOpacity={0.12} />}
            {single != null && <ReferenceArea x1={single} x2={hi} fill={C.local} fillOpacity={0.09} />}
            <CartesianGrid stroke={C.grid} horizontal={false} />
            <XAxis type="number" dataKey={xKey} domain={[lo, hi]} ticks={learned ? QUALITY_TICKS : TICKS} tickFormatter={(v) => (learned ? v.toFixed(0) : v.toFixed(1))} tick={{ fontSize: 12, fill: C.muted }} stroke={C.grid} />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0.4, units.length + 0.6]}
              ticks={units.map((_, i) => units.length - i)}
              tickFormatter={(v) => unitLabel(units[units.length - v])}
              width={88}
              tick={{ fontSize: 13, fill: C.ink }}
              axisLine={false}
              tickLine={false}
            />
            <ZAxis range={[50, 50]} />
            {marks.map((t) => (
              <ReferenceLine key={t} x={t} stroke={C.ink} strokeWidth={1.5} strokeDasharray="5 4"
                label={{ value: `${learned ? "cut-point" : "threshold"} ${t.toFixed(2)}`, position: "top", fontSize: 12, fill: C.ink }} />
            ))}
            <Tooltip content={<StripTip learned={learned} />} cursor={{ strokeDasharray: "3 3", stroke: C.muted }} isAnimationActive={false} />
            <Scatter data={normal} fill={C.normal} fillOpacity={0.72} shape="circle" isAnimationActive={false} />
            <Scatter data={exceptions} fill={C.exception} fillOpacity={0.92} shape="diamond" isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        {single != null && (
          <>
            <span><i className="swatch" style={{ background: C.escalated, opacity: 0.5 }} /> escalated (cache or Claude)</span>
            <span><i className="swatch" style={{ background: C.local, opacity: 0.5 }} /> answered locally</span>
          </>
        )}
        <span><i className="dot" style={{ background: C.normal }} /> normal query</span>
        <span><i className="dot diamond" style={{ background: C.exception }} /> exception query</span>
      </div>
    </div>
  );
}
