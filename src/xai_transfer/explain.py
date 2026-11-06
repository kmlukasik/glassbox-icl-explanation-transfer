import pandas as pd


def global_importance(ebm) -> pd.Series:
    """Ważność termów EBM posortowana malejąco"""
    return pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)


def pick_features(ebm, X: pd.DataFrame, min_unique: int = 10):
    """Zwraca najważniejszą cechę numeryczną oraz najmniej ważną cechę
    numeryczną z co najmniej min_unique różnymi wartościami (kontrola)"""
    imp = global_importance(ebm)
    num = set(X.select_dtypes("number").columns)
    imp = imp[[n for n in imp.index if n in num]]
    wide = [n for n in imp.index if X[n].nunique() >= min_unique]
    return imp.index[0], wide[-1]