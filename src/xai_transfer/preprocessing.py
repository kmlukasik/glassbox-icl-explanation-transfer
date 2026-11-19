import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



def preprocess_ebm(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    X_train = X_train.copy()
    X_test = X_test.copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Wartosci kolumn w zbiorze treningowym i testowym nie są zgodne.")

    for col in X_train.columns:
        if (
            isinstance(X_train[col].dtype, pd.CategoricalDtype)
            and not isinstance(X_test[col].dtype, pd.CategoricalDtype)
        ):
            X_test[col] = X_test[col].astype("category")

    return X_train, X_test


def preprocess_xgboost(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer | None]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Wartosci kolumn w zbiorze treningowym i testowym nie są zgodne.")

    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if not categorical_cols:
        return X_train, X_test, None

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,)
        ],
        remainder="passthrough",
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(X_train_array, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_array, columns=feature_names, index=X_test.index)
    
    return X_train_processed, X_test_processed, preprocessor

def preprocess_tabpfn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Wartosci kolumn w zbiorze treningowym i testowym nie są zgodne.")

    return X_train, X_test