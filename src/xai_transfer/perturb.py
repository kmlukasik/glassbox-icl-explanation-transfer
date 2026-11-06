import pandas as pd


def shift_feature(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    feature: str,
    shift_std: float,
    clip: bool = True,
) -> pd.DataFrame:
    """Przesunięcie cechy w X o shift_std * std(X_train[feature])"""
    Xp = X.copy()
    col = X_train[feature].astype(float)
    new = Xp[feature].astype(float) + shift_std * col.std()
    if clip:
        new = new.clip(col.min(), col.max())
    if pd.api.types.is_integer_dtype(X[feature]):
        new = new.round()
    Xp[feature] = new.astype(X[feature].dtype)
    return Xp
