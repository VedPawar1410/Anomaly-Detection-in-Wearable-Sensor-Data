import numpy as np
import pandas as pd

from wearable_anomaly import faults


def _clean(n=3000):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, dtype=np.float32)
    df.insert(0, "subject", np.repeat([1, 2], n // 2))
    return df


def test_inject_labels_every_changed_reading_and_nothing_else():
    df = _clean()
    std = df[["a", "b"]].std()
    corrupted, event_ids, events = faults.inject(df, std, ["a", "b"])
    changed = corrupted[["a", "b"]].to_numpy() != df[["a", "b"]].to_numpy()
    labelled = event_ids > 0
    assert not (changed & ~labelled).any()  # no unlabelled corruption
    # stuck windows can repeat an identical value, so only spikes/offsets must all change
    moving = np.isin(event_ids, events[events.type != "stuck"].id)
    assert changed[moving].all()
    assert set(events.type) == set(faults.FAULT_TYPES)


def test_events_stay_within_one_subject_and_do_not_overlap():
    df = _clean()
    _, event_ids, events = faults.inject(df, df[["a", "b"]].std(), ["a", "b"])
    for ev in events.itertuples():
        span = df["subject"].iloc[ev.start : ev.start + ev.length]
        assert span.nunique() == 1
    # every event id occupies exactly `length` cells, so no event overwrote another
    counts = pd.Series(event_ids[event_ids > 0]).value_counts().sort_index()
    assert (counts.to_numpy() == events.set_index("id").length.to_numpy()).all()
