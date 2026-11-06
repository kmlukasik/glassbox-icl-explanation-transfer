import pandas as pd


def global_importance(ebm) -> pd.Series:
    """Ważność termów EBM posortowana malejąco"""
    return pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)


def pick_features(ebm, X: pd.DataFrame):
    """Zwraca najbardziej ważną i najmniej ważną cechę numeryczną"""
    imp = global_importance(ebm)
    num = set(X.select_dtypes("number").columns)
    imp = imp[[n for n in imp.index if n in num]]
    return imp.index[0], imp.index[-1]
