import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ReferenceArea, ReferenceLine, CartesianGrid } from "recharts";
import { C } from "../theme";
import { fmtConf, unitLabel } from "../format";

const TICKS = Array.from({ length: 11 }, (_, i) => i / 10);

function StripTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tip">
      <div className="tip-title">{p.q}</div>
      <div>
        {unitLabel(p.u)}
        {p.i ? `, ${p.i}` : ""}
        {p.x ? ", exception" : ""}
      </div>
      <div>
        Confidence {fmtConf(p.c)}: {p.d === "LOCAL" ? "answered locally" : "sent to Claude"}
      </div>
    </div>
  );
}

// One dot per query, placed by the confidence of its local answer. The line is the router's threshold.
export default function ConfidenceStrip({ points, thresholds }) {
  const units = [...new Set(points.map((p) => p.u))].sort();
  const lane = Object.fromEntries(units.map((u, i) => [u, units.length - i])); // first unit on the top lane
  // spread dots vertically inside their lane so they do not sit on top of each other (same spread every render)
  const data = points.map((p, i) => ({ ...p, y: lane[p.u] + (((i * 7919) % 1000) / 1000 - 0.5) * 0.7 }));
  const normal = data.filter((d) => !d.x);
  const exceptions = data.filter((d) => d.x);
  const single = thresholds.length === 1 ? thresholds[0] : null;

  return (
    <div>
      <div className="chart" style={{ height: 110 + units.length * 96 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 30, right: 24, bottom: 6, left: 4 }}>
            {single != null && <ReferenceArea x1={0} x2={single} fill={C.escalated} fillOpacity={0.12} />}
            {single != null && <ReferenceArea x1={single} x2={1} fill={C.local} fillOpacity={0.09} />}
            <CartesianGrid stroke={C.grid} horizontal={false} />
            <XAxis type="number" dataKey="c" domain={[0, 1]} ticks={TICKS} tickFormatter={(v) => v.toFixed(1)} tick={{ fontSize: 12, fill: C.muted }} stroke={C.grid} />
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
            {thresholds.map((t) => (
              <ReferenceLine key={t} x={t} stroke={C.ink} strokeWidth={1.5} strokeDasharray="5 4"
                label={{ value: `threshold ${t.toFixed(2)}`, position: "top", fontSize: 12, fill: C.ink }} />
            ))}
            <Tooltip content={<StripTip />} cursor={{ strokeDasharray: "3 3", stroke: C.muted }} isAnimationActive={false} />
            <Scatter data={normal} fill={C.normal} fillOpacity={0.72} shape="circle" isAnimationActive={false} />
            <Scatter data={exceptions} fill={C.exception} fillOpacity={0.92} shape="diamond" isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        {single != null && (
          <>
            <span><i className="swatch" style={{ background: C.escalated, opacity: 0.5 }} /> sent to Claude</span>
            <span><i className="swatch" style={{ background: C.local, opacity: 0.5 }} /> answered locally</span>
          </>
        )}
        <span><i className="dot" style={{ background: C.normal }} /> normal query</span>
        <span><i className="dot diamond" style={{ background: C.exception }} /> exception query</span>
      </div>
    </div>
  );
}
