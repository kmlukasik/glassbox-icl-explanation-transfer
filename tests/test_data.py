
import pandas as pd
import pytest

from xai_transfer.data import (
    clean_features,
    encode_target,
    make_split,
    save_split,
)


def test_clean_features():
    X = pd.DataFrame({
        "age": [20, 30, 40],
        "occupation": ["worker", "?", "manager"],
    })

    cleaned = clean_features(X)

    assert cleaned["occupation"].isna().sum() == 0
    assert isinstance(
        cleaned["occupation"].dtype,
        pd.CategoricalDtype,
    )


def test_encode_target():
    y = pd.Series(["good", "bad", "good"])

    encoded = encode_target(
        y,
        positive_class="good",
        negative_class="bad",
    )

    assert encoded.tolist() == [1, 0, 1]


def test_encode_target_invalid_classes():
    y = pd.Series(["good", "unknown"])

    with pytest.raises(ValueError):
        encode_target(y, "good", "bad")


def test_make_split():
    X = pd.DataFrame({
        "feature": range(100),
    })
    y = pd.Series([0, 1] * 50)

    split = make_split(
        X=X,
        y=y,
        test_size=0.2,
        seed=42,
    )

    assert len(split.X_train) == 80
    assert len(split.X_test) == 20

    assert split.y_train.mean() == 0.5
    assert split.y_test.mean() == 0.5


def test_make_split_reproducibility():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0, 1] * 50)

    first = make_split(X, y, 0.2, 42)
    second = make_split(X, y, 0.2, 42)

    pd.testing.assert_frame_equal(
        first.X_train,
        second.X_train,
    )

    pd.testing.assert_series_equal(
        first.y_test,
        second.y_test,
    )
