import numpy as np
import pandas as pd

from xai_transfer.metrics import summarize
from xai_transfer.perturb import shift_feature


def test_shift_clips_to_train_range():
    Xtr = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    Xp = shift_feature(Xtr, Xtr, "a", shift_std=100, clip=True)
    assert Xp["a"].max() == 3.0


def test_shift_does_not_mutate_input():
    Xtr = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    X = Xtr.copy()
    shift_feature(X, Xtr, "a", 1.0)
    pd.testing.assert_frame_equal(X, Xtr)


def test_summarize_keys():
    rng = np.random.default_rng(0)
    d = rng.normal(size=50)
    res = summarize(d, d + rng.normal(scale=0.1, size=50), n_resamples=100)
    assert res["spearman"] > 0.8
