"""Run the repeatable Spaceship Titanic data audit from the repository root."""

from __future__ import annotations

from pathlib import Path

from .cleaning import (
    audit_dataset,
    clean_dataset,
    create_audit_figures,
    load_data,
    save_processed_data,
    validate_dataset,
    write_audit_report,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw = load_data(root / "data" / "raw")
    audits = {name: audit_dataset(frame, name) for name, frame in raw.items()}
    cleaned, changes = {}, {}
    for name, frame in raw.items():
        cleaned[name], changes[name] = clean_dataset(frame)
        validate_dataset(
            frame,
            cleaned[name],
            is_train=name == "train",
            is_submission=name == "sample_submission",
        )
    outputs = save_processed_data(cleaned, root / "data" / "processed")
    reloaded = {name: __import__("pandas").read_csv(path) for name, path in outputs.items()}
    for name, frame in reloaded.items():
        validate_dataset(
            cleaned[name],
            frame,
            is_train=name == "train",
            is_submission=name == "sample_submission",
        )
    figures = create_audit_figures(cleaned["train"], root / "web" / "assets" / "generated")
    write_audit_report(audits, raw, root / "data" / "raw", root / "docs" / "DATA_AUDIT.md", changes)
    print("Audit complete")
    for name, audit in audits.items():
        print(f"{name}: {audit['shape'][0]} rows x {audit['shape'][1]} columns")
    print("Cleaning changes:", {name: value or ["none"] for name, value in changes.items()})
    print("Processed:", ", ".join(str(path.relative_to(root)) for path in outputs.values()))
    print("Figures:", ", ".join(str(path.relative_to(root)) for path in figures))


if __name__ == "__main__":
    main()
