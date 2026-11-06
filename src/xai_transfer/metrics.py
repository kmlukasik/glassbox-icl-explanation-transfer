import numpy as np
from scipy.stats import bootstrap, spearmanr


def summarize(d_ebm, d_pfn, n_resamples=1000, seed=0) -> dict:
    """Podsumowanie różnic w predykcjach po perturbacji cechy"""
    d_ebm, d_pfn = np.asarray(d_ebm), np.asarray(d_pfn)

    match = (np.sign(d_ebm) == np.sign(d_pfn)).astype(float)
    ci = bootstrap(
        (match,), np.mean,
        n_resamples=n_resamples, random_state=seed, method="percentile",
    ).confidence_interval

    rho, p = spearmanr(d_ebm, d_pfn)

    return {
        "sign_agreement": float(match.mean()),
        "sign_ci_low": float(ci.low),
        "sign_ci_high": float(ci.high),
        "spearman": float(rho),
        "spearman_p": float(p),
        "mean_d_ebm": float(d_ebm.mean()),
        "mean_d_pfn": float(d_pfn.mean()),
    }
