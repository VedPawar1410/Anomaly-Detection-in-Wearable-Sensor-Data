"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SensorChart from "@/components/SensorChart";
import { ACTIVITY_COLORS, WS_URL } from "@/lib/api";

const WINDOW = 1000; // samples on screen: 10 s at 100 Hz
const DEFAULT_SENSORS = ["hand_acc16_x", "ankle_gyro_y", "heart_rate", "chest_mag_x"];

type Fault = { type: string; sensor: string; start: number; length: number };
type Meta = {
  sensors: string[];
  sample_rate_hz: number;
  batch_size: number;
  rows: number;
  activities: Record<string, string>;
  faults: Fault[];
};
type Batch = {
  seq: number;
  activity_id: number;
  values: number[][];
  flags: [number, number, number][];
  faults: [number, number, number][];
  compute_ms: number;
};
type Status = "idle" | "connecting" | "live" | "ended" | "error";

// The last WINDOW samples per sensor. Rebuilt as new arrays every batch (never mutated in
// place) so React and the charts see each update: 93 arrays of <=1000 numbers, 10x a second.
type Buffers = { x: number[]; y: number[][]; flagged: (number | null)[][]; injected: number[][] };
const emptyBuffers = (n: number): Buffers => ({
  x: [],
  y: Array.from({ length: n }, () => []),
  flagged: Array.from({ length: n }, () => []),
  injected: Array.from({ length: n }, () => []),
});

export default function LiveStream() {
  const [status, setStatus] = useState<Status>("idle");
  const [speed, setSpeed] = useState(1);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [selected, setSelected] = useState<string[]>(DEFAULT_SENSORS);
  const [activity, setActivity] = useState("");
  const [stats, setStats] = useState({ seen: 0, caught: 0, falseAlarms: 0, readings: 0, computeMs: 0 });
  const [view, setView] = useState<Buffers>(() => emptyBuffers(31));

  const ws = useRef<WebSocket | null>(null);
  const buf = useRef<Buffers>(emptyBuffers(31));
  const eventOf = useRef<Map<string, number>>(new Map()); // "row:sensor" -> fault index
  const caught = useRef<Set<number>>(new Set());
  const counters = useRef({ falseAlarms: 0, readings: 0, compute: [] as number[] });

  const stop = useCallback(() => {
    ws.current?.close();
    ws.current = null;
  }, []);
  useEffect(() => stop, [stop]);

  const start = () => {
    stop();
    setStatus("connecting");
    buf.current = emptyBuffers(31);
    setView(buf.current);
    caught.current = new Set();
    counters.current = { falseAlarms: 0, readings: 0, compute: [] };
    const sock = new WebSocket(`${WS_URL}/ws/stream?speed=${speed}`);
    ws.current = sock;
    let m: Meta | null = null;

    sock.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "meta") {
        m = msg as Meta;
        setMeta(m);
        const map = new Map<string, number>();
        m.faults.forEach((f, i) => {
          const s = m!.sensors.indexOf(f.sensor);
          for (let r = f.start; r < f.start + f.length; r++) map.set(`${r}:${s}`, i);
        });
        eventOf.current = map;
        setStatus("live");
        return;
      }
      if (msg.type === "end") {
        setStatus("ended");
        return;
      }
      if (!m) return;
      const b = msg as Batch;
      const base = b.seq * m.batch_size;
      const n = m.sensors.length;
      const prev = buf.current;
      const flagSet = new Set(b.flags.map(([r, c]) => `${r}:${c}`));
      const injSet = new Set(b.faults.map(([r, c]) => `${r}:${c}`));
      const keep = <T,>(old: T[], add: T[]) => old.concat(add).slice(-WINDOW);
      const next: Buffers = {
        x: keep(
          prev.x,
          b.values.map((_, r) => (base + r) / m!.sample_rate_hz)
        ),
        y: prev.y.map((old, c) => keep(old, b.values.map((row) => row[c]))),
        flagged: prev.flagged.map((old, c) =>
          keep(old, b.values.map((row, r) => (flagSet.has(`${r}:${c}`) ? row[c] : null)))
        ),
        injected: prev.injected.map((old, c) =>
          keep(old, b.values.map((_, r) => (injSet.has(`${r}:${c}`) ? 1 : 0)))
        ),
      };
      buf.current = next;
      setView(next);
      // event-level scoring: a fault counts as caught if any of its readings is flagged
      for (const [r, c] of b.flags) {
        const ev = eventOf.current.get(`${base + r}:${c}`);
        if (ev === undefined) counters.current.falseAlarms++;
        else caught.current.add(ev);
      }
      counters.current.readings += b.values.length * n;
      counters.current.compute.push(b.compute_ms);
      const now = base + b.values.length;
      const seen = m.faults.filter((f) => f.start < now).length;
      const comp = [...counters.current.compute].slice(-50).sort((a, z) => a - z);
      setStats({
        seen,
        caught: caught.current.size,
        falseAlarms: counters.current.falseAlarms,
        readings: counters.current.readings,
        computeMs: comp[Math.floor(comp.length / 2)] ?? 0,
      });
      setActivity(m.activities[String(b.activity_id)] ?? "");
    };
    sock.onerror = () => setStatus("error");
    sock.onclose = () => setStatus((s) => (s === "live" ? "ended" : s));
  };

  const sensors = meta?.sensors ?? [];
  const toggle = (s: string) =>
    setSelected((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s].slice(-6)));
  const faultSensors = new Set(meta?.faults.map((f) => f.sensor));
  const live = status === "live" || status === "connecting";

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={live ? stop : start}
          className="rounded-md bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-700"
        >
          {live ? "Stop" : status === "idle" ? "Start live stream" : "Restart"}
        </button>
        <label className="flex items-center gap-2 text-sm text-stone-700">
          Speed
          <select
            value={speed}
            disabled={live}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="rounded border border-stone-300 bg-white px-2 py-1"
          >
            {[1, 2, 5, 10].map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <span className="text-sm text-stone-600">
          {status === "connecting" && "Connecting…"}
          {status === "error" && "Couldn't connect to the stream. Is the API running?"}
          {status === "ended" && "Stream finished."}
          {status === "live" && activity && (
            <>
              Activity: <strong className="text-stone-900">{activity}</strong>
            </>
          )}
        </span>
      </div>

      {meta && (
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Injected faults caught" value={`${stats.caught} / ${stats.seen}`} />
          <Stat
            label="False-alarm readings"
            value={`${stats.falseAlarms}`}
            hint={stats.readings ? `${((100 * stats.falseAlarms) / stats.readings).toFixed(2)}% of readings` : ""}
          />
          <Stat label="Server compute per batch" value={`${stats.computeMs.toFixed(1)} ms`} hint="median, 10 samples" />
          <Stat label="Sample rate" value={`${meta.sample_rate_hz} Hz`} hint={`${speed}× replay`} />
        </dl>
      )}

      {meta && (
        <div className="flex flex-wrap gap-1.5" aria-label="Sensors to show">
          {sensors.map((s) => (
            <button
              key={s}
              onClick={() => toggle(s)}
              aria-pressed={selected.includes(s)}
              className={`rounded-full border px-2.5 py-0.5 font-mono text-xs ${
                selected.includes(s)
                  ? "border-stone-900 bg-stone-900 text-white"
                  : "border-stone-300 bg-white text-stone-700 hover:border-stone-500"
              }`}
              title={faultSensors.has(s) ? "Faults are injected into this sensor" : "No injected faults"}
            >
              {s}
              {faultSensors.has(s) && " •"}
            </button>
          ))}
        </div>
      )}

      {meta && (
        <p className="flex flex-wrap items-center gap-4 text-xs text-stone-600">
          <Legend swatch={<span className="inline-block h-0.5 w-4" style={{ background: ACTIVITY_COLORS.series }} />}>
            sensor reading
          </Legend>
          <Legend
            swatch={<span className="inline-block h-2 w-2 rounded-full" style={{ background: ACTIVITY_COLORS.flagged }} />}
          >
            flagged by the detector
          </Legend>
          <Legend swatch={<span className="inline-block h-3 w-4" style={{ background: ACTIVITY_COLORS.injected }} />}>
            fault actually injected
          </Legend>
          <span>• = sensor has injected faults</span>
        </p>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {selected
          .filter((s) => sensors.includes(s))
          .map((s) => {
            const c = sensors.indexOf(s);
            return (
              <SensorChart
                key={s}
                title={s}
                x={view.x}
                y={view.y[c]}
                flagged={view.flagged[c]}
                injected={view.injected[c]}
              />
            );
          })}
      </div>
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white p-3">
      <dt className="text-xs text-stone-600">{label}</dt>
      <dd className="text-xl font-semibold text-stone-900 tabular-nums">{value}</dd>
      {hint && <dd className="text-xs text-stone-500">{hint}</dd>}
    </div>
  );
}

function Legend({ swatch, children }: { swatch: React.ReactNode; children: React.ReactNode }) {
  return (
    <span className="flex items-center gap-1.5">
      {swatch}
      {children}
    </span>
  );
}
