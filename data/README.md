# Data: PAMAP2 Physical Activity Monitoring

Everything in `data/` except this file is git-ignored. Download the data with:

```bash
python scripts/download_data.py
```

The script uses only the Python standard library. It downloads the UCI archive (~690 MB zip), unpacks the
nested zip and leaves about 1.6 GB in `data/raw/PAMAP2_Dataset/`. Re-running it is safe.

## Source

- UCI Machine Learning Repository, dataset 231:
  <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>
- License: CC BY 4.0
- Citation: A. Reiss and D. Stricker, *Introducing a New Benchmarked Dataset for Activity Monitoring*,
  ISWC 2012.

## What's in it

9 subjects wore 3 IMUs (inertial measurement units: wrist, chest and ankle, each sampled 100 times a second)
and a heart-rate monitor (sampled about 9 times a second) while following a 12-activity protocol.

| Folder | Files | Rows |
|---|---|---|
| `Protocol/` | `subject101.dat` to `subject109.dat` | 2,872,533 |
| `Optional/` | 5 subjects, extra activities | 977,972 |

Rows per protocol file (checked after download, Nov 2026):

| 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---|---|---|---|---|---|---|---|
| 376,417 | 447,000 | 252,833 | 329,576 | 374,783 | 361,817 | 313,599 | 408,031 | 8,477 |

## File format

Space-separated text with no header and 54 columns per row:

| Columns | Content |
|---|---|
| 1 | timestamp (s, since recording start) |
| 2 | activityID |
| 3 | heart rate (bpm) |
| 4-20 | IMU hand |
| 21-37 | IMU chest |
| 38-54 | IMU ankle |

Each IMU block of 17 values:

| In-block index | Content |
|---|---|
| 1 | temperature (°C) |
| 2-4 | 3D acceleration, ±16g (m/s²), the recommended accelerometer |
| 5-7 | 3D acceleration, ±6g (m/s²), saturates during high-impact movement |
| 8-10 | 3D gyroscope (rad/s) |
| 11-13 | 3D magnetometer (μT) |
| 14-17 | orientation (marked invalid by the dataset authors) |

Missing sensor values are `NaN`. Heart rate is `NaN` in about 91% of rows only because it's sampled about
9 times a second against 100 for the IMUs. Those gaps are not data loss.

## Gotchas this project handles

- **activityID 0** is transient time between activities. The dataset authors say to discard it.
- **Orientation columns** are invalid and are dropped.
- **The ±6g accelerometer** duplicates the ±16g one (and saturates), so it's dropped. Keeping it lets a
  model "predict" one accelerometer from the other almost perfectly (R² 0.9995), which teaches nothing.
- **Heart rate** is forward-filled within each subject (the last reading holds until the next one), not
  mean-imputed.

## History note

The 2024 version of this project used a copy in which every subject file had been cut off after its first
1-4 minutes: 131,329 rows in total, 4.6% of the protocol data, covering only 3 activities. The download script
replaces that copy with the full dataset.
