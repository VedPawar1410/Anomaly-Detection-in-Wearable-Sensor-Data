"use client";

import { useEffect, useState } from "react";
import { API_URL } from "@/lib/api";

type Status = "checking" | "online" | "waking" | "offline";

// The free API host sleeps when idle; the first request can take ~1 minute to wake it.
export default function ApiStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;
    const slow = setTimeout(() => !cancelled && setStatus((s) => (s === "checking" ? "waking" : s)), 3000);
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(90_000) })
      .then((r) => !cancelled && setStatus(r.ok ? "online" : "offline"))
      .catch(() => !cancelled && setStatus("offline"))
      .finally(() => clearTimeout(slow));
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);

  const text = {
    checking: "Checking API…",
    online: "API online",
    waking: "Waking the API (free hosting sleeps when idle, up to a minute)…",
    offline: `API unreachable at ${API_URL}`,
  }[status];
  const dot = { checking: "bg-stone-400", online: "bg-green-600", waking: "bg-amber-500", offline: "bg-red-600" }[
    status
  ];
  return (
    <p className="flex items-center gap-2 text-xs text-stone-600" role="status">
      <span className={`inline-block h-2 w-2 rounded-full ${dot}`} aria-hidden />
      {text}
    </p>
  );
}
