"""Run focused Step 05 evaluation and final artifact generation."""

from pathlib import Path
import pandas as pd

from .evaluate import append_final_evaluation, create_evaluation_charts, evaluate_candidate, save_production_artifact, write_model_card


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "data" / "processed" / "train_clean.csv")
    result = evaluate_candidate(train)
    charts = create_evaluation_charts(result, root / "web" / "assets" / "generated")
    artifact, metadata = save_production_artifact(train, root / "model")
    append_final_evaluation(result, root / "MODEL_RESULTS.md", artifact)
    write_model_card(result, root / "docs" / "MODEL_CARD.md", artifact)
    print("Candidate: Gradient Boosting")
    print("Validation metrics:", {key: round(value, 4) for key, value in result["metrics"].items()})
    print("Artifact:", artifact)
    print("Metadata:", metadata)
    print("Charts:", ", ".join(charts))


if __name__ == "__main__":
    main()
