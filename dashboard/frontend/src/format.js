export const fmtInt = (v) => (v == null ? "-" : Math.round(v).toLocaleString());
export const fmtPct = (v, d = 0) => (v == null ? "-" : `${v.toFixed(d)}%`);
export const fmtConf = (v) => (v == null ? "-" : v.toFixed(3));
export const fmtMs = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
export const fmtUsd = (v) => (v == null ? "-" : v < 0.01 ? `$${v.toFixed(5)}` : `$${v.toFixed(3)}`);

export const unitLabel = (u) => ({ payments: "Payments", retail_bank: "Retail bank" }[u] ?? u);

// "run-28-Sep-26-06:29 PM-2" -> "28-Sep-26 06:29 PM-2"
export const runLabel = (id) => (id ? id.replace(/^run-/, "").replace(/-(\d{2}:\d{2} [AP]M)/, " $1") : "");

// "run-28-Sep-26-06:29 PM-2" -> "28-Sep 06:29 PM-2" (short enough for a chart axis)
export const runTick = (id) => {
  const m = /^run-(\d{2}-[A-Za-z]{3})-\d{2}-(\d{2}:\d{2} [AP]M)(-\d+)?$/.exec(id || "");
  return m ? `${m[1]} ${m[2]}${m[3] ?? ""}` : runLabel(id);
};

export const fmtWhen = (iso) =>
  iso ? new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" }) : "";
export const fmtClock = (d) => d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
