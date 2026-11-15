from wearable_anomaly import config


def test_raw_layout_matches_dataset_readme():
    assert len(config.RAW_COLUMNS) == 54
    assert len(set(config.RAW_COLUMNS)) == 54


def test_sensor_columns_exclude_invalid_and_duplicate_channels():
    assert len(config.SENSOR_COLUMNS) == 31  # heart rate + 3 IMUs x (temp + 3 x 3 axes)
    assert not any("orient" in c or "acc6" in c for c in config.SENSOR_COLUMNS)
    assert set(config.SENSOR_COLUMNS) <= set(config.RAW_COLUMNS)


def test_subject_split_is_disjoint():
    train, calib, test = map(
        set, (config.TRAIN_SUBJECTS, config.CALIB_SUBJECTS, config.TEST_SUBJECTS)
    )
    assert not (train & calib or train & test or calib & test)
    assert train | calib | test <= set(config.EXPECTED_ROWS)
