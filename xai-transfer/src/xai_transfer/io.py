import shutil
from pathlib import Path

import pandas as pd


def save_results(
    df: pd.DataFrame,
    cfg: dict,
    cfg_path: str | Path,
    name: str,
    out_dir: str | Path = "results",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["dataset"] = cfg["dataset"]
    df["seed"] = cfg["seed"]
    df["shift_std"] = cfg["perturbation"]["shift_std"]

    parquet_path = out_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    shutil.copy(cfg_path, out_dir / f"{name}.config.yaml")
    return parquet_path
