import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from tabpfn import TabPFNClassifier


def train_ebm(X: pd.DataFrame, y: pd.Series, seed: int, **kwargs):
    return ExplainableBoostingClassifier(random_state=seed, **kwargs).fit(X, y)


def fit_tabpfn(X: pd.DataFrame, y: pd.Series, seed: int, device: str = "auto"):
    return TabPFNClassifier(device=device, random_state=seed).fit(X, y)


def proba(model, X: pd.DataFrame):
    return model.predict_proba(X)[:, 1]
