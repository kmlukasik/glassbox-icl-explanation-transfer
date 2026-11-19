from dataclasses import dataclass
from importlib.resources import path
from pathlib import Path    
import json

import openml
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def load_openml(dataset_name: str, dataset_id: int, target: str, output_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.Series]:

    raw_dir = Path(output_dir) / str(dataset_id)
    raw_file = raw_dir / "data.parquet"

    if raw_file.exists():
        print(f"Znaleziono dane niezmienione w {raw_file}. Wczytywanie...")
        raw = pd.read_parquet(raw_file)
        if target not in raw.columns:
            raise ValueError(f"Brak kolumny docelowej '{target}' w danych niezmienionych.")
        y = raw.pop(target)
        X = raw

        return X, y

    print(f"Pobieranie zbioru danych {dataset_id} z OpenML...")

    ds = openml.datasets.get_dataset(dataset_id)
    actual_target = ds.default_target_attribute
    if actual_target != target:
        raise ValueError(f"Zbiór danych {dataset_id} ma inną kolumnę docelową: {actual_target}. Oczekiwano: {target}")
    X, y, _, _ = ds.get_data(target=target, dataset_format="dataframe")

    if X is None or y is None:
        raise ValueError(f"OpenML zwrócił puste dane dla zbioru {dataset_id}")

    X = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)

    if X.columns.duplicated().any():
        raise ValueError(f"Zbiór danych {dataset_id} zawiera zduplikowane nazwy kolumn: {X.columns[X.columns.duplicated()].tolist()}")

    y.name = target

    save_raw(X, y, dataset_name, output_dir)

    return X, y

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.columns = X.columns.astype(str)

    for col in X.columns:
        series = X[col]

        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            series = (
                series.astype("string")
                .str.strip()
                .replace(r"^\s*\?\s*$", pd.NA, regex=True)
            )

            if series.isna().any():
                print(f"Kolumna '{col}' zawiera {series.isna().sum()} brakujących wartości. Zastępowanie brakujących wartości kategorią 'not-specified'.")
                series = series.fillna("not-specified")

            X[col] = series.astype("category")

    return X

def encode_target(
    y: pd.Series,
    positive_class: str,
    negative_class: str,
) -> pd.Series:

    y = y.astype("string").str.strip()

    if y.isna().any():
        raise ValueError("Brakujące wartości w kolumnie docelowej")

    classes = set(y.unique())
    expected = {positive_class, negative_class}

    if classes != expected:
        raise ValueError(
            f"Nieprawidłowe klasy w kolumnie docelowej: {classes}. Oczekiwano: {expected}."
        )

    encoded = (y == positive_class).astype("int8")
    encoded.name = "target"

    return encoded

def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
) -> Split:
    """Create reproducible stratified train/test split."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return Split(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )

def save_raw(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    output_dir: str | Path = "data/raw",
) -> Path:

    path = Path(output_dir) / dataset_name
    path.mkdir(parents=True, exist_ok=True)

    raw = X.copy()
    raw[y.name] = y
    raw.to_parquet(path / "data.parquet", index=False)

    return path

def save_interim(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    output_dir: str | Path = "data/interim",
) -> Path:

    path = Path(output_dir) / dataset_name
    path.mkdir(parents=True, exist_ok=True)

    interim = X.copy()
    interim[y.name] = y

    interim.to_parquet(path / "data.parquet", index=False)

    return path


def save_split(
    split: Split,
    dataset_name: str,
    metadata: dict,
    output_dir: str | Path = "data/processed",
) -> Path:

    path = Path(output_dir) / dataset_name
    path.mkdir(parents=True, exist_ok=True)

    split.X_train.to_parquet(
        path / "X_train.parquet", index=False
    )
    split.X_test.to_parquet(
        path / "X_test.parquet", index=False
    )
    split.y_train.to_frame().to_parquet(
        path / "y_train.parquet", index=False
    )
    split.y_test.to_frame().to_parquet(
        path / "y_test.parquet", index=False
    )

    with (path / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=4)

    return path