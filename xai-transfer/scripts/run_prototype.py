import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from xai_transfer.data import load_openml
from xai_transfer.explain import pick_features
from xai_transfer.metrics import summarize
from xai_transfer.models import fit_tabpfn, proba, train_ebm
from xai_transfer.perturb import shift_feature
from xai_transfer.io import save_results


def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    seed = cfg["seed"]

    s = load_openml(cfg["dataset"], cfg["positive_class"],
                    cfg["test_size"], seed=seed)
    ebm = train_ebm(s.X_train, s.y_train, seed=seed, **cfg["ebm"])
    pfn = fit_tabpfn(s.X_train, s.y_train, seed=seed,
                     **cfg["tabpfn"]["device"])

    print("EBM AUC:", roc_auc_score(s.y_test, proba(ebm, s.X_test)))
    print("TabPFN AUC:", roc_auc_score(s.y_test, proba(pfn, s.X_test)))

    top, irrelevant = pick_features(ebm, s.X_train)
    p = cfg["perturbation"]
    base_ebm, base_pfn = proba(ebm, s.X_test), proba(pfn, s.X_test)

    rows = []
    for role, feat in [("top", top), ("control", irrelevant)]:
        Xp = shift_feature(s.X_test, s.X_train, feat,
                           p["shift_std"], p["clip_to_train_range"])
        d_ebm = proba(ebm, Xp) - base_ebm
        d_pfn = proba(pfn, Xp) - base_pfn
        res = summarize(
            d_ebm, d_pfn, cfg["bootstrap"]["n_resamples"], seed=seed)
        res.update(role=role, feature=feat, mean_abs_d_ebm=float(
            np.abs(d_ebm).mean()), mean_abs_d_pfn=float(np.abs(d_pfn).mean()))
        rows.append(res)
    out = pd.DataFrame(rows)
    print(out.T)
    save_results(out, cfg, cfg_path, name="prototype")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.yaml")
    args = parser.parse_args()
    main(args.config)
