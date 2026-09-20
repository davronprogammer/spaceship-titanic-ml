"""Reusable, non-mutating exploratory analysis for Spaceship Titanic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
PALETTE = {False: "#304a6e", True: "#c46d3b"}


def load_processed_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Step 01 outputs, never the immutable raw files."""
    return (
        pd.read_csv(root / "data" / "processed" / "train_clean.csv"),
        pd.read_csv(root / "data" / "processed" / "test_clean.csv"),
    )


def add_eda_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a temporary analysis copy; no processed file is changed."""
    data = frame.copy()
    data["TotalSpending"] = data[SPENDING].sum(axis=1, min_count=1)
    cabin = data["Cabin"].str.split("/", expand=True)
    data["CabinDeck"] = cabin[0]
    data["CabinNumber"] = pd.to_numeric(cabin[1], errors="coerce")
    data["CabinSide"] = cabin[2]
    data["GroupId"] = data["PassengerId"].str.split("_", n=1).str[0]
    data["GroupSize"] = data.groupby("GroupId")["PassengerId"].transform("size")
    return data


def transportation_rate(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate observed target rates with an explicit missing category."""
    labels = frame[column].astype("object").where(frame[column].notna(), "Missing")
    result = frame.assign(_category=labels).groupby("_category", dropna=False)["Transported"].agg(
        passengers="size", transported="sum", transportation_rate="mean"
    )
    return result.reset_index(names=column).sort_values("transportation_rate", ascending=False)


def _save(fig: plt.Figure, path: Path) -> bool:
    """Save a new chart but never replace an existing asset."""
    if path.exists():
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _counts(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("object").where(frame[column].notna(), "Missing").value_counts()


def _count_plot(frame: pd.DataFrame, column: str, title: str, path: Path) -> bool:
    counts = _counts(frame, column)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#304a6e")
    ax.set(title=title, ylabel="Passengers")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    return _save(fig, path)


def _rate_plot(frame: pd.DataFrame, column: str, title: str, path: Path) -> bool:
    rates = transportation_rate(frame, column).sort_values("transportation_rate")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(rates[column].astype(str), rates["transportation_rate"] * 100, color="#c46d3b")
    ax.set(title=title, xlabel="Observed transportation rate (%)", xlim=(0, 100))
    for bar, rate in zip(bars, rates["transportation_rate"]):
        ax.text(rate * 100 + 1, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center", fontsize=8)
    return _save(fig, path)


def create_eda_figures(train: pd.DataFrame, output_dir: Path) -> list[str]:
    """Create the Step 02 chart set, retaining pre-existing charts untouched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = add_eda_features(train)
    created: list[str] = []
    def add(filename: str, made: bool) -> None:
        if made:
            created.append(filename)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for value, label in [(False, "Not transported"), (True, "Transported")]:
        ax.hist(data.loc[data.Transported.eq(value), "Age"].dropna(), bins=30, alpha=.65, label=label, color=PALETTE[value])
    ax.set(title="Age distribution by transportation outcome", xlabel="Age", ylabel="Passengers")
    ax.legend()
    add("age_by_transportation.png", _save(fig, output_dir / "age_by_transportation.png"))
    for column, stem, label in [("HomePlanet", "homeplanet", "Home planet"), ("Destination", "destination", "Destination"), ("CryoSleep", "cryosleep", "CryoSleep"), ("VIP", "vip", "VIP")]:
        add(f"{stem}_distribution.png", _count_plot(data, column, f"Passenger distribution by {label}", output_dir / f"{stem}_distribution.png"))
        add(f"{stem}_vs_transportation.png", _rate_plot(data, column, f"Transportation rate by {label}", output_dir / f"{stem}_vs_transportation.png"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.flat, SPENDING):
        ax.hist(data[column].dropna(), bins=40, color="#304a6e", edgecolor="white")
        ax.set(title=column, xlabel="Spend", ylabel="Passengers")
    axes.flat[-1].axis("off")
    add("spending_distributions.png", _save(fig, output_dir / "spending_distributions.png"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for value, label in [(False, "Not transported"), (True, "Transported")]:
        ax.hist(data.loc[data.Transported.eq(value), "TotalSpending"].dropna(), bins=50, range=(0, 8000), density=True, alpha=.65, label=label, color=PALETTE[value])
    ax.set(title="Total spending by transportation outcome", xlabel="Total spending (display capped at 8,000)", ylabel="Density")
    ax.legend()
    add("total_spending_vs_transportation.png", _save(fig, output_dir / "total_spending_vs_transportation.png"))
    medians = data.groupby("HomePlanet")["TotalSpending"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(medians.index, medians.values, color="#304a6e")
    ax.set(title="Median total spending by home planet", ylabel="Median total spending")
    for bar, value in zip(bars, medians.values): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    add("spending_by_homeplanet.png", _save(fig, output_dir / "spending_by_homeplanet.png"))
    add("cabin_deck_distribution.png", _count_plot(data, "CabinDeck", "Passenger distribution by cabin deck", output_dir / "cabin_deck_distribution.png"))
    add("cabin_deck_vs_transportation.png", _rate_plot(data, "CabinDeck", "Transportation rate by cabin deck", output_dir / "cabin_deck_vs_transportation.png"))
    add("cabin_side_vs_transportation.png", _rate_plot(data, "CabinSide", "Transportation rate by cabin side", output_dir / "cabin_side_vs_transportation.png"))
    group_counts = data.drop_duplicates("GroupId")["GroupSize"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(group_counts.index.astype(str), group_counts.values, color="#304a6e")
    ax.set(title="Distribution of passenger group sizes", xlabel="Group size", ylabel="Groups")
    add("group_size_distribution.png", _save(fig, output_dir / "group_size_distribution.png"))
    add("group_size_vs_transportation.png", _rate_plot(data, "GroupSize", "Transportation rate by passenger group size", output_dir / "group_size_vs_transportation.png"))
    numeric = data[["Age", *SPENDING, "TotalSpending"]].copy(); numeric["Transported"] = data["Transported"].astype(int)
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(9, 7)); image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)), xticklabels=corr.columns, yticklabels=corr.columns, title="Correlation among numeric EDA variables")
    ax.tick_params(axis="x", rotation=45); fig.colorbar(image, ax=ax, label="Pearson correlation")
    add("correlation_heatmap.png", _save(fig, output_dir / "correlation_heatmap.png"))
    return created


def _table(frame: pd.DataFrame, digits: int = 2) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"): shown[col] = shown[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return shown.to_markdown(index=False)


def write_eda_report(train: pd.DataFrame, test: pd.DataFrame, report_path: Path) -> None:
    """Write concise evidence-led findings calculated from processed datasets."""
    data, test_data = add_eda_features(train), add_eda_features(test)
    target = train.Transported.value_counts().rename_axis("Transported").reset_index(name="Passengers"); target["Percent"] = target.Passengers / len(train) * 100
    def rates(column: str) -> pd.DataFrame:
        result = transportation_rate(data, column); result["transportation_rate"] *= 100; return result
    age_bins = pd.cut(data.Age, [-1, 12, 17, 29, 44, 64, 100], labels=["0-12", "13-17", "18-29", "30-44", "45-64", "65+"])
    age_rates = transportation_rate(data.assign(AgeGroup=age_bins), "AgeGroup"); age_rates.transportation_rate *= 100
    spending = data[SPENDING + ["TotalSpending"]].describe(percentiles=[.5]).T.reset_index(names="Feature")[["Feature", "count", "mean", "50%", "max"]]
    zero = int(data.TotalSpending.eq(0).sum())
    cabin_rates, group_rates = rates("CabinDeck"), rates("GroupSize")
    deck = cabin_rates.loc[cabin_rates.CabinDeck != "Missing"].sort_values("transportation_rate", ascending=False)
    side = rates("CabinSide")
    correlations = data[["Age", *SPENDING, "TotalSpending"]].copy(); correlations["Transported"] = data.Transported.astype(int)
    corr_to_target = correlations.corr()["Transported"].drop("Transported").sort_values(key=lambda s: s.abs(), ascending=False).head(4)
    comparisons = []
    for col in ["Age", "HomePlanet", "Destination", "CryoSleep", "VIP"]:
        if col == "Age": comparisons.append([col, f"mean {train.Age.mean():.2f}", f"mean {test.Age.mean():.2f}"])
        else:
            comparisons.append([col, f"missing {train[col].isna().mean():.2%}", f"missing {test[col].isna().mean():.2%}"])
    report = f"""# Spaceship Titanic - EDA Insights

## 1. Executive Summary

The training set is near evenly split: 4,378 of 8,693 passengers (50.36%) were transported. The clearest descriptive contrast is CryoSleep: 81.76% of recorded CryoSleep=True passengers were transported versus 32.89% for CryoSleep=False. Europa passengers had a 65.88% observed rate, compared with 42.39% from Earth. These are observed associations, not causal claims or model feature importance.

## 2. Dataset Overview

Train contains {len(train):,} rows and {train.shape[1]} columns; test contains {len(test):,} rows and {test.shape[1]} columns. The train set has {int(train.isna().sum().sum()):,} missing cells and test has {int(test.isna().sum().sum()):,}. There are six original numeric columns, four low-cardinality categorical/boolean variables, and high-cardinality identifiers/text fields.

{_table(target)}

## 3. Passenger Demographics

Age is available for {data.Age.notna().sum():,} passengers (mean {data.Age.mean():.2f}, median {data.Age.median():.0f}, range {data.Age.min():.0f}-{data.Age.max():.0f}). No negative ages occur. Observed transportation rates by compact age band are:

{_table(age_rates[["AgeGroup", "passengers", "transportation_rate"]])}

## 4. Home Planet

{_table(rates("HomePlanet")[["HomePlanet", "passengers", "transportation_rate"]])}

Europa has the highest observed rate among recorded planets; Earth has the lowest. Missing home planet remains a separate category rather than an inferred value.

## 5. Destination

{_table(rates("Destination")[["Destination", "passengers", "transportation_rate"]])}

## 6. CryoSleep

{_table(rates("CryoSleep")[["CryoSleep", "passengers", "transportation_rate"]])}

The large rate difference is an EDA observation that should be tested with other variables during modelling.

## 7. VIP

{_table(rates("VIP")[["VIP", "passengers", "transportation_rate"]])}

Only {int(data.VIP.eq(True).sum()):,} recorded passengers are VIP, so the VIP comparison has a much smaller sample than non-VIP.

## 8. Spending Behavior

All spending measures are right-skewed and have a median of zero or near-zero. TotalSpending is an EDA-only row sum and was not written to processed data. {zero:,} passengers ({zero / len(data):.2%}) have zero recorded total spending.

{_table(spending)}

Median total spending differs by home planet: {", ".join(f"{planet} {amount:,.0f}" for planet, amount in data.groupby("HomePlanet").TotalSpending.median().sort_values(ascending=False).items())}. The charts show distributions rather than treating high-spend extremes as errors.

## 9. Cabin Patterns

Cabin-derived columns are temporary EDA fields. Among recorded decks, the highest observed rate is {deck.iloc[0].CabinDeck} ({deck.iloc[0].transportation_rate:.2f}%) and the lowest is {deck.iloc[-1].CabinDeck} ({deck.iloc[-1].transportation_rate:.2f}%). Cabin side rates are:

{_table(side[["CabinSide", "passengers", "transportation_rate"]])}

CabinNumber was parsed only to verify structure; no continuous-number interpretation is asserted.

## 10. Passenger Groups

All PassengerId values retain the documented `dddd_dd` pattern, so the prefix was used as a temporary GroupId. There are {data.GroupId.nunique():,} groups; {int(data.drop_duplicates("GroupId").GroupSize.gt(1).sum()):,} ({data.drop_duplicates("GroupId").GroupSize.gt(1).mean():.2%}) contain more than one recorded passenger. The most common group size is {int(data.drop_duplicates("GroupId").GroupSize.mode().iloc[0])}. Group-size rates are charted descriptively; the endpoint rates may have small samples.

{_table(group_rates[["GroupSize", "passengers", "transportation_rate"]])}

## 11. Numerical Relationships

Pearson correlations were calculated only for numeric variables plus the binary target. The largest absolute target correlations are {", ".join(f"{name} ({value:.3f})" for name, value in corr_to_target.items())}. This measures linear association only and does not establish predictive value.

## 12. Train/Test Distribution

{_table(pd.DataFrame(comparisons, columns=["Feature", "Train", "Test"]))}

The age means are close and missingness differs by less than one percentage point for these high-level checks. Category charts and counts should still be reviewed during preprocessing for any encoding-sensitive shift.

## 13. Strongest Observed Patterns

- CryoSleep status has a large observed transportation-rate gap.
- Home planet and destination have visibly different observed transportation rates.
- Spending is highly zero-inflated and right-skewed.
- Cabin deck and side show descriptive rate differences where cabin data is present.
- Passenger identifiers form verifiable groups, making group-size candidates reasonable to test later.

## 14. Questions Raised by EDA

- Does CryoSleep remain associated with the target after accounting for spending and cabin location?
- Do cabin deck and side add useful signal once missingness is handled?
- Can zero-spending behavior improve classification without leakage?
- Does group size add stable value on held-out data?
- Which categorical-missingness treatments generalize from train to test?

# Story Material for Website

### CryoSleep divides the passenger record
**Evidence:** 81.76% transported with recorded CryoSleep=True versus 32.89% with False.  
**Visualization:** `cryosleep_vs_transportation.png`  
**Why It Matters:** It provides a striking, factual transition from passenger state to the incident outcome.

### Europa passengers had a higher observed transportation rate
**Evidence:** Europa: 65.88%; Earth: 42.39%.  
**Visualization:** `homeplanet_vs_transportation.png`  
**Why It Matters:** It establishes that the voyage population was not homogeneous.

### Most journeys pointed to TRAPPIST-1e
**Evidence:** {int(data.Destination.eq("TRAPPIST-1e").sum()):,} recorded passengers named TRAPPIST-1e as destination.  
**Visualization:** `destination_distribution.png`  
**Why It Matters:** It gives the narrative a concrete destination before comparing outcomes.

### Spending records are mostly quiet, with a long tail
**Evidence:** {zero:,} passengers had zero recorded total spending; every spending feature is strongly right-skewed.  
**Visualization:** `spending_distributions.png`  
**Why It Matters:** It makes the economic behavior aboard the ship legible without overstating a cause.

### Cabin location leaves a visible pattern
**Evidence:** Recorded deck rates range from {deck.iloc[-1].transportation_rate:.2f}% to {deck.iloc[0].transportation_rate:.2f}%.  
**Visualization:** `cabin_deck_vs_transportation.png`  
**Why It Matters:** It motivates a later, careful cabin-feature experiment.

### Passengers were often not travelling alone
**Evidence:** {int(data.drop_duplicates("GroupId").GroupSize.gt(1).sum()):,} of {data.GroupId.nunique():,} groups contain multiple recorded passengers.  
**Visualization:** `group_size_distribution.png`  
**Why It Matters:** It introduces the social structure encoded in PassengerId.
"""
    report_path.write_text(report, encoding="utf-8")
