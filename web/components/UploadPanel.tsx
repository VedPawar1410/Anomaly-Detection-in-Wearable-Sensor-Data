"use client";

import { useState } from "react";
import { API_URL } from "@/lib/api";

type Flag = { subject: number; timestamp: number; activity_id: number; sensor: string; value: number; score: number };
type Result = {
  filename: string;
  rows: number;
  dropped_rows: number;
  sensors: number;
  flagged: number;
  flagged_pct: number;
  flagged_per_sensor: Record<string, number>;
  returned_flags: number;
  flags: Flag[];
};

export default function UploadPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const r = await fetch(`${API_URL}/detect?max_flags=200`, { method: "POST", body });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail ?? `Request failed (${r.status})`);
      setResult(data as Result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const perSensor = result ? Object.entries(result.flagged_per_sensor).slice(0, 10) : [];
  const maxCount = perSensor[0]?.[1] ?? 1;
  const topFlags = result ? [...result.flags].sort((a, b) => b.score - a.score).slice(0, 25) : [];

  return (
    <section className="space-y-6">
      <form onSubmit={submit} className="flex flex-wrap items-center gap-3 rounded-lg border border-stone-200 bg-white p-4">
        <input
          type="file"
          accept=".dat,.csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-sm text-stone-700 file:mr-3 file:rounded file:border-0 file:bg-stone-100 file:px-3 file:py-1.5"
        />
        <button
          type="submit"
          disabled={!file || busy}
          className="rounded-md bg-stone-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy ? "Detecting…" : "Detect anomalies"}
        </button>
        <a href="/sample_subject108.csv" download className="text-sm text-[#2a78d6] underline">
          Download a sample recording (CSV, 30 s, contains injected faults)
        </a>
      </form>
      <p className="text-xs text-stone-500">
        Accepts a PAMAP2 <code>.dat</code> file or a CSV with the 31 sensor columns, up to 50 MB.
      </p>

      {error && (
        <p role="alert" className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </p>
      )}

      {result && (
        <>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Card label="Rows analysed" value={result.rows.toLocaleString()} hint={`${result.dropped_rows.toLocaleString()} transient rows dropped`} />
            <Card label="Readings flagged" value={result.flagged.toLocaleString()} />
            <Card label="Share flagged" value={`${result.flagged_pct}%`} hint="of all sensor readings" />
            <Card label="Sensors" value={String(result.sensors)} />
          </dl>

          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <h2 className="mb-2 text-sm font-semibold text-stone-900">Flags per sensor (top 10)</h2>
              <ul className="space-y-1.5">
                {perSensor.map(([sensor, n]) => (
                  <li key={sensor} className="grid grid-cols-[9rem_1fr_4rem] items-center gap-2 text-xs">
                    <span className="truncate font-mono text-stone-700">{sensor}</span>
                    <span className="h-3 rounded-r bg-[#2a78d6]" style={{ width: `${(100 * n) / maxCount}%` }} />
                    <span className="text-right tabular-nums text-stone-700">{n.toLocaleString()}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="mb-2 text-sm font-semibold text-stone-900">Most anomalous readings</h2>
              <div className="max-h-80 overflow-auto rounded border border-stone-200">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-stone-50 text-stone-600">
                    <tr>
                      <th className="px-2 py-1.5">Time (s)</th>
                      <th className="px-2 py-1.5">Sensor</th>
                      <th className="px-2 py-1.5 text-right">Value</th>
                      <th className="px-2 py-1.5 text-right">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topFlags.map((f, i) => (
                      <tr key={i} className="border-t border-stone-100">
                        <td className="px-2 py-1 tabular-nums">{f.timestamp.toFixed(2)}</td>
                        <td className="px-2 py-1 font-mono">{f.sensor}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{f.value.toFixed(3)}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{f.score.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-1 text-xs text-stone-500">Score = how far past its threshold (1.0 = at the threshold).</p>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function Card({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white p-3">
      <dt className="text-xs text-stone-600">{label}</dt>
      <dd className="text-xl font-semibold text-stone-900 tabular-nums">{value}</dd>
      {hint && <dd className="text-xs text-stone-500">{hint}</dd>}
    </div>
  );
}
