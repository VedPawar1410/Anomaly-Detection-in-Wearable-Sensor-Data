import ApiStatus from "@/components/ApiStatus";
import LiveStream from "@/components/LiveStream";

export default function LivePage() {
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-stone-900">Live sensor stream</h1>
        <p className="max-w-3xl text-sm text-stone-600">
          Replays 140 s of a person the model never saw (PAMAP2 subject 108) at 100 samples a second, with faults
          injected into 8 sensors: spikes, frozen readings, offsets, drifts, dropouts and noise bursts. The detector
          scores every reading as it arrives, using only the past. Shaded bands are the real faults, red dots are what
          it flagged.
        </p>
        <p className="max-w-3xl text-xs text-stone-500">
          Plays in real time (1×). The detector runs at about 15× real time on a laptop, so faster replays work there;
          this demo runs on a free hosting tier whose CPU limit only keeps up with real time.
        </p>
        <ApiStatus />
      </div>
      <LiveStream />
    </div>
  );
}
