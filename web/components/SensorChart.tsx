"use client";

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { ACTIVITY_COLORS } from "@/lib/api";

type Props = {
  title: string;
  x: number[]; // seconds
  y: number[];
  flagged: (number | null)[]; // value where the detector flagged, else null
  injected: number[]; // 1 where a fault was really injected, else 0
  height?: number;
};

// Shade every run of injected-fault samples behind the line.
function shadeInjected(injected: number[]) {
  return (u: uPlot) => {
    const xs = u.data[0] as number[];
    const { ctx, bbox } = u;
    ctx.save();
    ctx.fillStyle = ACTIVITY_COLORS.injected;
    let start = -1;
    for (let i = 0; i <= injected.length; i++) {
      const on = i < injected.length && injected[i] === 1;
      if (on && start < 0) start = i;
      if (!on && start >= 0) {
        const x0 = u.valToPos(xs[start], "x", true);
        const x1 = u.valToPos(xs[i - 1], "x", true);
        ctx.fillRect(x0, bbox.top, Math.max(x1 - x0, 2), bbox.height);
        start = -1;
      }
    }
    ctx.restore();
  };
}

export default function SensorChart({ title, x, y, flagged, injected, height = 150 }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const injectedRef = useRef(injected); // read by the draw hook; updated in the data effect below

  // Create the chart once; resize with its container.
  useEffect(() => {
    if (!box.current) return;
    const axis = { stroke: "#52514e", grid: { stroke: "#ebeae6", width: 1 }, ticks: { show: false } };
    const opts: uPlot.Options = {
      width: box.current.clientWidth,
      height,
      legend: { show: false },
      cursor: { drag: { x: false, y: false } },
      scales: { x: { time: false } },
      axes: [{ ...axis, values: (_u, ticks) => ticks.map((t) => `${t.toFixed(0)}s`) }, { ...axis, size: 50 }],
      series: [
        {},
        { label: "signal", stroke: ACTIVITY_COLORS.series, width: 1.5, points: { show: false } },
        {
          label: "flagged",
          stroke: ACTIVITY_COLORS.flagged,
          paths: () => null,
          points: { show: true, size: 7, fill: ACTIVITY_COLORS.flagged, stroke: "#fcfcfb", width: 2 },
        },
      ],
      hooks: { drawClear: [(u) => shadeInjected(injectedRef.current)(u)] },
    };
    plot.current = new uPlot(opts, [[], [], []], box.current);
    const ro = new ResizeObserver(() => {
      if (box.current && plot.current) plot.current.setSize({ width: box.current.clientWidth, height });
    });
    ro.observe(box.current);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, [height]);

  // Push new data on every update (10 times a second while streaming).
  useEffect(() => {
    injectedRef.current = injected;
    plot.current?.setData([x, y, flagged] as uPlot.AlignedData);
  }, [x, y, flagged, injected]);

  return (
    <figure className="rounded-lg border border-stone-200 bg-[#fcfcfb] p-2">
      <figcaption className="px-1 pb-1 font-mono text-xs text-stone-700">{title}</figcaption>
      <div ref={box} />
    </figure>
  );
}
