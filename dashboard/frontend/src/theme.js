// One place for the colours the charts use (the CSS variables in App.css mirror these).
export const C = {
  ink: "#14283A",
  muted: "#5C6E7E",
  grid: "#DDE3E8",
  local: "#0F8B8D",      // answered by the local model (Tier 1)
  escalated: "#E3A008",  // sent to Claude (Tier 2)
  tier2: "#3D3B8E",
  normal: "#5B7083",
  exception: "#C2185B",
  payments: "#3D5AD0",
  retail_bank: "#0F8B8D",
  claudePrompt: "#8E44AD",  // distinct from tier2/exception/payments -- Tier 2 system prompt changes specifically
};
export const unitColor = (u) => C[u] ?? "#7A8894";
