import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from "recharts";
import { C } from "../theme";
import { fmtQuality } from "../format";

const DIM_LABELS = { correctness: "Correctness", completeness: "Completeness", tone: "Tone", safety: "Safety", clarity: "Clarity" };

/** Per-dimension answer quality (1-5, Claude-as-judge) for one run -- a bar per dimension,
 * so a specific weakness (e.g. tone on exception queries) is visible, not just one blended
 * "quality" number. */
export default function QualityBreakdown({ byDim }) {
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
