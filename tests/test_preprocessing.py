import pandas as pd

from xai_transfer.preprocessing import *


#EBM preprocessing tests

def test_preprocess_ebm_preserves_numeric_and_categorical_features():
    X_train = pd.DataFrame({
        "age": [20, 30, 40],
        "job": pd.Series(["admin", "technician", "admin"], dtype="category"),
    })

    X_test = pd.DataFrame({
        "age": [25, 35],
        "job": pd.Series(["admin", "services"], dtype="category"),
    })

    X_train_processed, X_test_processed = preprocess_ebm(
        X_train, X_test
    )

    assert X_train_processed["age"].dtype == X_train["age"].dtype
    assert isinstance(
        X_train_processed["job"].dtype,
        pd.CategoricalDtype,
    )

    assert X_test_processed["age"].dtype == X_test["age"].dtype
    assert isinstance(
        X_test_processed["job"].dtype,
        pd.CategoricalDtype,
    )


def test_preprocess_ebm_preserves_column_order():
    X_train = pd.DataFrame({
        "age": [20, 30],
        "income": [1000, 2000],
    })

    X_test = pd.DataFrame({
        "age": [25],
        "income": [1500],
    })

    X_train_processed, X_test_processed = preprocess_ebm(
        X_train, X_test
    )

    assert list(X_train_processed.columns) == ["age", "income"]
    assert list(X_test_processed.columns) == ["age", "income"]


# XGBoost preprocessing tests

def test_preprocess_xgboost_encodes_categorical_features():
    X_train = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "job": pd.Series(
            ["admin", "worker", "admin", "teacher"],
            dtype="category",
        ),
    })

    X_test = pd.DataFrame({
        "age": [25, 35],
        "job": pd.Series(
            ["worker", "unknown"],
            dtype="category",
        ),
    })

    X_train_processed, X_test_processed, preprocessor = (
        preprocess_xgboost(X_train, X_test)
    )

    assert preprocessor is not None

    assert "remainder__age" in X_train_processed.columns
    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    assert X_train_processed.select_dtypes(
        include=["object", "category"]
    ).empty

    assert X_test_processed.select_dtypes(
        include=["object", "category"]
    ).empty


def test_preprocess_xgboost_handles_numeric_only_data():
    X_train = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [1000, 2000, 3000],
    })

    X_test = pd.DataFrame({
        "age": [25, 35],
        "income": [1500, 2500],
    })

    X_train_processed, X_test_processed, preprocessor = (
        preprocess_xgboost(X_train, X_test)
    )

    assert preprocessor is None
    pd.testing.assert_frame_equal(X_train_processed, X_train)
    pd.testing.assert_frame_equal(X_test_processed, X_test)



# TabPFN preprocessing tests

def test_preprocess_tabpfn_preserves_feature_types():
    X_train = pd.DataFrame({
        "age": [20, 30, 40],
        "job": pd.Series(
            ["admin", "worker", "teacher"],
            dtype="category",
        ),
    })

    X_test = pd.DataFrame({
        "age": [25, 35],
        "job": pd.Series(
            ["worker", "admin"],
            dtype="category",
        ),
    })

    X_train_processed, X_test_processed = preprocess_tabpfn(
        X_train, X_test
    )

    assert X_train_processed["age"].dtype == X_train["age"].dtype
    assert isinstance(
        X_train_processed["job"].dtype,
        pd.CategoricalDtype,
    )
    assert isinstance(
        X_test_processed["job"].dtype,
        pd.CategoricalDtype,
    )


def test_preprocess_tabpfn_preserves_missing_values():
    X_train = pd.DataFrame({
        "age": [20, 30, 40],
        "job": pd.Series(
            ["admin", None, "worker"],
            dtype="category",
        ),
    })

    X_test = pd.DataFrame({
        "age": [25],
        "job": pd.Series(["admin"], dtype="category"),
    })

    X_train_processed, X_test_processed = preprocess_tabpfn(
        X_train, X_test
    )

    assert X_train_processed["job"].isna().sum() == 1
    assert X_test_processed["job"].isna().sum() == 0