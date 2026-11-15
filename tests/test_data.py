import numpy as np
import pytest

from wearable_anomaly import config
from wearable_anomaly.data import load_subject, read_dat


def _write_dat(path, n_rows):
    rows = np.random.default_rng(0).normal(size=(n_rows, 54))
    rows[:, 1] = 1  # activity id
    rows[::2, 2] = np.nan  # heart rate NaN, as in the real files
    np.savetxt(path, rows, fmt="%.4f")


def test_read_dat_parses_54_named_columns(tmp_path):
    f = tmp_path / "subject999.dat"
    _write_dat(f, 20)
    df = read_dat(f, expected_rows=20)
    assert list(df.columns) == config.RAW_COLUMNS
    assert df["heart_rate"].isna().sum() == 10


def test_read_dat_fails_loudly_on_truncated_file(tmp_path):
    f = tmp_path / "subject999.dat"
    _write_dat(f, 15)
    with pytest.raises(ValueError, match="expected 20"):
        read_dat(f, expected_rows=20)


@pytest.mark.skipif(
    not (config.PROTOCOL_DIR / "subject109.dat").exists(), reason="dataset not downloaded"
)
def test_real_subject_loads_with_official_row_count():
    df = load_subject(109, use_cache=False)
    assert len(df) == config.EXPECTED_ROWS[109]
    assert (df["subject"] == 109).all()
