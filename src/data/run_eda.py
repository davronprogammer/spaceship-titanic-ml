"""Run Step 02 reproducible EDA from the repository root."""
from pathlib import Path
from .eda import create_eda_figures, load_processed_data, write_eda_report

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train, test = load_processed_data(root)
    created = create_eda_figures(train, root / "web" / "assets" / "generated")
    write_eda_report(train, test, root / "EDA_INSIGHTS.md")
    print(f"Created {len(created)} charts: {', '.join(created)}")

if __name__ == "__main__":
    main()
