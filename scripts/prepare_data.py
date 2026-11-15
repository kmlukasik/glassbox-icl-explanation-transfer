
from logging import config
from pathlib import Path
from unicodedata import name

import yaml

from xai_transfer.data import *

def prepare_dataset(
    name: str,
    config: dict,
    seed: int,
    test_size: float,
) -> Split:

    print(f"\Pobieranie {name}...")

    X_raw, y_raw = load_openml(
        dataset_name=name,
        dataset_id=config["openml_id"],
        target=config["target"],
    )

    print(f"Wymiary danych niezmienionych: {X_raw.shape}")

    print("Czyszczenie cech...")

    X = clean_features(X_raw)

    save_interim(X, y_raw, name)

    y = encode_target(
        y_raw,
        positive_class=str(config["positive_class"]),
        negative_class=str(config["negative_class"]),
    )

    print("Tworzenie podziału na zbiór treningowy i testowy...")

    split = make_split(
        X=X,
        y=y,
        test_size=test_size,
        seed=seed,
    )

    metadata = {
        "dataset": name,
        "openml_id": config["openml_id"],
        "target": config["target"],
        "positive_class": config["positive_class"],
        "negative_class": config["negative_class"],
        "seed": seed,
        "test_size": test_size,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "train_samples": len(split.X_train),
        "test_samples": len(split.X_test),
        "train_positive": int(split.y_train.sum()),
        "test_positive": int(split.y_test.sum()),
        "missing_values": int(X.isna().sum().sum()),
        "feature_types": {
            str(col): str(dtype)
            for col, dtype in X.dtypes.items()
        },
    }

    save_split(split, name, metadata)

    print(f"Przygotowano {name}:")
    print(f"  Liczba cech: {X.shape[1]}")
    print(f"  Train: {len(split.X_train)}")
    print(f"  Test: {len(split.X_test)}")
    print(f"  Brakujące wartości: {metadata['missing_values']}")
    print(f"  Zapisano do data/processed/{name}")

    return split


def main():
    config_path = Path("configs/datasets.yaml")

    with config_path.open() as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    test_size = config["test_size"]

    for name, dataset_config in config["datasets"].items():
        prepare_dataset(
            name=name,
            config=dataset_config,
            seed=seed,
            test_size=test_size,
        )

    print("\nWszystkie zbiory danych przygotowane pomyślnie!")


if __name__ == "__main__":
    main()
