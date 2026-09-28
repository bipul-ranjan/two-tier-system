import { useEffect, useState } from "react";
import { getJson } from "./api";

// Fetches `path` now and every `intervalMs`, so the page follows the logs while a pipeline run is in progress.
// Passing a different path clears the old data first; polling the same path keeps it (no flicker).
export function useLiveData(path, intervalMs = 5000) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [updated, setUpdated] = useState(null);

  useEffect(() => {
    setData(null);
    if (!path) return undefined;
    let stopped = false;
    async function tick() {
      try {
        const next = await getJson(path);
        if (stopped) return;
        setData(next);
        setError(null);
        setUpdated(new Date());
      } catch (e) {
        if (!stopped) setError(e.message);
      }
    }
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, [path, intervalMs]);

  return { data, error, updated };
}
