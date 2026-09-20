"""Run Step 04 baseline experiments and write measured artifacts."""

from pathlib import Path
import pandas as pd

from .train import create_model_charts, run_experiments, write_results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "data" / "processed" / "train_clean.csv")
    results = run_experiments(train)
    charts = create_model_charts(results, root / "web" / "assets" / "generated")
    write_results(results, root / "MODEL_RESULTS.md")
    print(results["validation"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Cross-validation")
    print(results["cross_validation"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Charts:", ", ".join(charts))


if __name__ == "__main__":
    main()
