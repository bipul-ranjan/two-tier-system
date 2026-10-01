import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from "recharts";
import { C } from "../theme";
import { fmtQuality } from "../format";

const DIM_LABELS = { correctness: "Correctness", completeness: "Completeness", tone: "Tone", safety: "Safety", clarity: "Clarity" };

function DimChart({ byDim }) {
  const data = Object.entries(DIM_LABELS)
    .map(([key, label]) => ({ key, label, value: byDim?.[key] ?? null }))
    .filter((d) => d.value != null);
  if (data.length === 0) return null;
  return (
    <div className="chart" style={{ height: 200 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 30, bottom: 4, left: 10 }}>
          <CartesianGrid stroke={C.grid} horizontal={false} />
          <XAxis type="number" domain={[0, 5]} tick={{ fontSize: 11, fill: C.muted }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="label" tick={{ fontSize: 12, fill: C.ink }} axisLine={false} tickLine={false} width={100} />
          <Tooltip isAnimationActive={false} formatter={(v) => [fmtQuality(v), "score"]} cursor={{ fill: C.grid, opacity: 0.4 }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.key} fill={d.value >= 4 ? C.local : d.value >= 3 ? C.escalated : C.exception} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Per-dimension answer quality (1-5) for one run, from BOTH judges shown side by side and
 * clearly labelled -- never blended into one number, since they're different measurements
 * (Claude is the validated judge; local is a cheap, less-validated first pass). Either panel
 * is omitted if that judge scored nothing in this run. */
export default function QualityBreakdown({ byDim, known, byDimLocal, knownLocal, rows }) {
  const hasClaude = known > 0;
  const hasLocal = knownLocal > 0;
  if (!hasClaude && !hasLocal) return null;

  return (
    <div className="resolution-trend">
      {hasClaude && (
        <div className="resolution-trend-panel">
          <h3>Claude judge</h3>
          <p className="lede" style={{ marginTop: 0 }}>{known} of {rows} answers scored.</p>
          <DimChart byDim={byDim} />
        </div>
      )}
      {hasLocal && (
        <div className="resolution-trend-panel">
          <h3>Local judge</h3>
          <p className="lede" style={{ marginTop: 0 }}>{knownLocal} of {rows} answers scored -- cheap first pass, less validated.</p>
          <DimChart byDim={byDimLocal} />
        </div>
      )}
    </div>
  );
}
