import numpy as np
import pandas as pd
import pytest

from wearable_anomaly import detectors

pytest.importorskip("torch")  # training extra; skipped where only requirements-dev is installed

from wearable_anomaly.detectors.autoencoder import ConvAutoencoder


def _signal(n, seed):
    """Two correlated sensors: b follows a with a lag, as body-worn sensors do."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 100
    a = np.sin(2 * np.pi * 0.8 * t)
    df = pd.DataFrame(
        {"a": a + rng.normal(0, 0.05, n), "b": np.roll(a, 10) + rng.normal(0, 0.05, n)}
    )
    df.insert(0, "subject", 1)
    return df


def _detector():
    # The bottleneck must be smaller than the input or the network learns to copy it, faults
    # included. Here: 2 sensors x 100 samples (200 values) -> 1 x 25 (25 values), 8x compression.
    # (Real data: 31 x 100 = 3,100 -> 8 x 25 = 200, about 15x.)
    # Also: enough optimizer steps to converge (10 epochs x ~31 batches). With ~100 steps the
    # loss barely moved (1.05 -> 0.73) and faults were indistinguishable from normal data.
    return ConvAutoencoder(
        ["a", "b"],
        false_alarm_rate=0.01,
        epochs=10,
        windows_per_epoch=8000,
        lr=5e-3,
        hidden=8,
        code=1,
    )


def test_autoencoder_flags_frozen_sensor_and_keeps_false_alarms_low():
    det = _detector().fit(_signal(20000, 0)).calibrate(_signal(10000, 1))
    test = _signal(3000, 2)
    test.loc[1000:1199, "b"] = test.loc[999, "b"]  # frozen for 2 s
    flags = det.score(test) > 1
    assert flags.loc[1000:1199, "b"].mean() > 0.3
    assert (det.score(_signal(10000, 3)) > 1).to_numpy().mean() < 0.05


def test_scores_cover_every_row_including_the_tail():
    det = _detector().fit(_signal(5000, 0)).calibrate(_signal(5000, 1))
    test = _signal(1234, 2)  # not a multiple of the 100-sample window
    scores = det.score(test)
    assert scores.shape == (1234, 2)
    assert (scores.iloc[-50:] > 0).all().all()  # the tail window was scored


def test_benchmark_includes_autoencoder_only_when_torch_installed():
    assert "conv_autoencoder" in detectors.available()
