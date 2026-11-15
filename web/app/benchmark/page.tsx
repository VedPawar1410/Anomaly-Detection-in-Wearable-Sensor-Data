import benchmark from "@/lib/benchmark.json";

const FAULTS = ["spike", "stuck", "offset", "drift", "dropout", "noise"] as const;
const BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const cellColor = (v: number) => BLUE_RAMP[Math.min(BLUE_RAMP.length - 1, Math.floor(v * BLUE_RAMP.length))];

export default function BenchmarkPage() {
  const rows = benchmark.detectors;
  const maxAuc = Math.max(...rows.map((r) => r.pr_auc));
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-stone-900">Benchmark</h1>
        <p className="max-w-3xl text-sm text-stone-600">
          Every detector scored the same data: unseen test subjects {benchmark.test_subjects.join(" and ")} (
          {benchmark.test_rows.toLocaleString()} rows) with {benchmark.injected_events.toLocaleString()} injected faults
          covering {benchmark.injected_reading_pct}% of readings. Thresholds came only from a separate calibration
          subject, targeting 0.5% false alarms.
        </p>
      </div>

      <section className="overflow-x-auto rounded-lg border border-stone-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="bg-stone-50 text-xs text-stone-600">
            <tr>
              <th className="px-3 py-2">Detector</th>
              <th className="px-3 py-2">PR-AUC (ranking quality)</th>
              <th className="px-3 py-2 text-right">F1</th>
              <th className="px-3 py-2 text-right">Precision</th>
              <th className="px-3 py-2 text-right">Recall</th>
              <th className="px-3 py-2 text-right">False alarms on clean data</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="border-t border-stone-100">
                <td className="px-3 py-2 font-mono text-xs">{r.name}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2" title={`PR-AUC ${r.pr_auc}`}>
                    <span className="h-3 rounded-r bg-[#2a78d6]" style={{ width: `${(100 * r.pr_auc) / maxAuc}%`, maxWidth: "10rem" }} />
                    <span className="tabular-nums">{r.pr_auc.toFixed(3)}</span>
                  </div>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{r.f1.toFixed(3)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{r.precision.toFixed(3)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{r.recall.toFixed(3)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{r.clean_false_alarm_pct.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold text-stone-900">Share of injected faults caught, by fault type</h2>
        <p className="max-w-3xl text-sm text-stone-600">
          A fault counts as caught if any of its readings is flagged. Recall alone rewards flagging everything, so each
          row also shows how much that detector flags overall.
        </p>
        <div className="overflow-x-auto rounded-lg border border-stone-200 bg-white">
          <table className="w-full border-separate border-spacing-0.5 text-center text-xs">
            <thead className="text-stone-600">
              <tr>
                <th className="px-2 py-2 text-left">Detector</th>
                {FAULTS.map((f) => (
                  <th key={f} className="px-2 py-2">
                    {f}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td className="px-2 py-1.5 text-left">
                    <div className="font-mono">{r.name}</div>
                    <div className="text-stone-500">flags {r.flagged_pct.toFixed(1)}% of readings</div>
                  </td>
                  {FAULTS.map((f) => {
                    const v = r.recall_by_type[f];
                    return (
                      <td
                        key={f}
                        className="rounded px-2 py-1.5 tabular-nums"
                        style={{ background: cellColor(v), color: v >= 0.55 ? "white" : "#0b0b0b" }}
                        title={`${r.name}: ${Math.round(100 * v)}% of ${f} faults caught`}
                      >
                        {Math.round(100 * v)}%
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="max-w-3xl space-y-2 text-sm text-stone-700">
        <h2 className="text-lg font-semibold text-stone-900">What this shows</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            The simplest detector wins: comparing each sensor with its own recent past beats predicting it from the
            other sensors, and it is the only one whose false-alarm rate carried over to new people.
          </li>
          <li>
            The detectors fail in opposite places. The rolling z-score catches frozen sensors and dropouts but misses
            slow drifts; the per-sensor models and the autoencoder catch drifts but miss frozen sensors.
          </li>
          <li>
            The 2024 rule flags most of all readings, so its high recall means nothing; its PR-AUC is near random.
          </li>
        </ul>
      </section>
    </div>
  );
}
