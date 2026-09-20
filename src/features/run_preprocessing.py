"""Run the Step 03 preprocessing smoke test without training a classifier."""

from pathlib import Path
import pandas as pd

from .preprocessing import validate_pipeline


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "data" / "processed" / "train_clean.csv")
    test = pd.read_csv(root / "data" / "processed" / "test_clean.csv")
    result = validate_pipeline(train, test)
    print(
        "Preprocessing validation passed | "
        f"train={result['train_shape']} validation={result['valid_shape']} test={result['test_shape']} | "
        f"train rate={result['train_rate']:.4f} validation rate={result['valid_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
