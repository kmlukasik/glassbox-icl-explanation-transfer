import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=300) > 0).astype(int))
    return X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:]


def test_cuda_available():
    assert torch.cuda.is_available()


def test_tabpfn_fit_predict_shape_and_range(toy):
    from xai_transfer.models import fit_tabpfn, proba

    X_tr, y_tr, X_te, _ = toy
    model = fit_tabpfn(X_tr, y_tr, seed=0, device="cuda")
    p = proba(model, X_te)
    assert p.shape == (len(X_te),)
    assert ((p >= 0) & (p <= 1)).all()


def test_tabpfn_reproducible(toy):
    from xai_transfer.models import fit_tabpfn, proba

    X_tr, y_tr, X_te, _ = toy
    p1 = proba(fit_tabpfn(X_tr, y_tr, seed=0, device="cuda"), X_te)
    p2 = proba(fit_tabpfn(X_tr, y_tr, seed=0, device="cuda"), X_te)
    np.testing.assert_allclose(p1, p2, atol=1e-4)


def test_tabpfn_responds_to_informative_feature(toy):
    from xai_transfer.models import fit_tabpfn, proba
    from xai_transfer.perturb import shift_feature

    X_tr, y_tr, X_te, _ = toy
    model = fit_tabpfn(X_tr, y_tr, seed=0, device="cuda")
    Xp = shift_feature(X_te, X_tr, "a", shift_std=1.0, clip=False)
    assert proba(model, Xp).mean() > proba(model, X_te).mean()
