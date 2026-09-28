import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";
import { C } from "../theme";
import { fmtMs } from "../format";

export default function LatencyChart({ latency }) {
  const data = [
    { name: "Answered locally", tier1: latency.local.tier1 ?? 0, tier2: 0 },
    { name: "Sent to Claude", tier1: latency.escalated.tier1 ?? 0, tier2: latency.escalated.tier2 ?? 0 },
  ];
  return (
    <div className="chart" style={{ height: 210 }}>
      <ResponsiveContainer>
        <BarChart layout="vertical" data={data} margin={{ top: 6, right: 24, bottom: 6, left: 4 }} barSize={30}>
          <CartesianGrid stroke={C.grid} horizontal={false} />
          <XAxis type="number" tickFormatter={(v) => `${+(v / 1000).toFixed(1)} s`} tick={{ fontSize: 12, fill: C.muted }} stroke={C.grid} />
          <YAxis type="category" dataKey="name" width={124} tick={{ fontSize: 13, fill: C.ink }} axisLine={false} tickLine={false} />
          <Tooltip isAnimationActive={false} cursor={{ fill: "rgba(20,40,58,0.05)" }} formatter={(v, n) => [fmtMs(v), n === "tier1" ? "Local model" : "Claude"]} />
          <Legend formatter={(n) => (n === "tier1" ? "Local model" : "Claude")} />
          <Bar dataKey="tier1" stackId="t" fill={C.local} isAnimationActive={false} />
          <Bar dataKey="tier2" stackId="t" fill={C.tier2} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
