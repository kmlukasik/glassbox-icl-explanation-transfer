from dataclasses import dataclass

import openml
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_openml(name: str, positive_class: str, test_size: float, seed: int) -> Split:
    ds = openml.datasets.get_dataset(name)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    y = (y.astype(str) == positive_class).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return Split(X_tr, X_te, y_tr, y_te)