# Spaceship Titanic - EDA Insights

## 1. Executive Summary

The training set is near evenly split: 4,378 of 8,693 passengers (50.36%) were transported. The clearest descriptive contrast is CryoSleep: 81.76% of recorded CryoSleep=True passengers were transported versus 32.89% for CryoSleep=False. Europa passengers had a 65.88% observed rate, compared with 42.39% from Earth. These are observed associations, not causal claims or model feature importance.

## 2. Dataset Overview

Train contains 8,693 rows and 14 columns; test contains 4,277 rows and 13 columns. The train set has 2,324 missing cells and test has 1,117. There are six original numeric columns, four low-cardinality categorical/boolean variables, and high-cardinality identifiers/text fields.

| Transported   |   Passengers |   Percent |
|:--------------|-------------:|----------:|
| True          |         4378 |     50.36 |
| False         |         4315 |     49.64 |

## 3. Passenger Demographics

Age is available for 8,514 passengers (mean 28.83, median 27, range 0-79). No negative ages occur. Observed transportation rates by compact age band are:

| AgeGroup   |   passengers |   transportation_rate |
|:-----------|-------------:|----------------------:|
| 0-12       |          806 |                 69.98 |
| 13-17      |          739 |                 55.35 |
| Missing    |          179 |                 50.28 |
| 45-64      |         1133 |                 49.16 |
| 30-44      |         2354 |                 47.45 |
| 18-29      |         3375 |                 47.23 |
| 65+        |          107 |                 43.93 |

## 4. Home Planet

| HomePlanet   |   passengers |   transportation_rate |
|:-------------|-------------:|----------------------:|
| Europa       |         2131 |                 65.88 |
| Mars         |         1759 |                 52.3  |
| Missing      |          201 |                 51.24 |
| Earth        |         4602 |                 42.39 |

Europa has the highest observed rate among recorded planets; Earth has the lowest. Missing home planet remains a separate category rather than an inferred value.

## 5. Destination

| Destination   |   passengers |   transportation_rate |
|:--------------|-------------:|----------------------:|
| 55 Cancri e   |         1800 |                 61    |
| Missing       |          182 |                 50.55 |
| PSO J318.5-22 |          796 |                 50.38 |
| TRAPPIST-1e   |         5915 |                 47.12 |

## 6. CryoSleep

| CryoSleep   |   passengers |   transportation_rate |
|:------------|-------------:|----------------------:|
| True        |         3037 |                 81.76 |
| Missing     |          217 |                 48.85 |
| False       |         5439 |                 32.89 |

The large rate difference is an EDA observation that should be tested with other variables during modelling.

## 7. VIP

| VIP     |   passengers |   transportation_rate |
|:--------|-------------:|----------------------:|
| Missing |          203 |                 51.23 |
| False   |         8291 |                 50.63 |
| True    |          199 |                 38.19 |

Only 199 recorded passengers are VIP, so the VIP comparison has a much smaller sample than non-VIP.

## 8. Spending Behavior

All spending measures are right-skewed and have a median of zero or near-zero. TotalSpending is an EDA-only row sum and was not written to processed data. 3,653 passengers (42.02%) have zero recorded total spending.

| Feature       |   count |    mean |   50% |   max |
|:--------------|--------:|--------:|------:|------:|
| RoomService   |    8512 |  224.69 |     0 | 14327 |
| FoodCourt     |    8510 |  458.08 |     0 | 29813 |
| ShoppingMall  |    8485 |  173.73 |     0 | 23492 |
| Spa           |    8510 |  311.14 |     0 | 22408 |
| VRDeck        |    8505 |  304.85 |     0 | 24133 |
| TotalSpending |    8693 | 1440.87 |   716 | 35987 |

Median total spending differs by home planet: Europa 1,901, Mars 946, Earth 704. The charts show distributions rather than treating high-spend extremes as errors.

## 9. Cabin Patterns

Cabin-derived columns are temporary EDA fields. Among recorded decks, the highest observed rate is B (73.43%) and the lowest is T (20.00%). Cabin side rates are:

| CabinSide   |   passengers |   transportation_rate |
|:------------|-------------:|----------------------:|
| S           |         4288 |                 55.5  |
| Missing     |          199 |                 50.25 |
| P           |         4206 |                 45.13 |

CabinNumber was parsed only to verify structure; no continuous-number interpretation is asserted.

## 10. Passenger Groups

All PassengerId values retain the documented `dddd_dd` pattern, so the prefix was used as a temporary GroupId. There are 6,217 groups; 1,412 (22.71%) contain more than one recorded passenger. The most common group size is 1. Group-size rates are charted descriptively; the endpoint rates may have small samples.

|   GroupSize |   passengers |   transportation_rate |
|------------:|-------------:|----------------------:|
|           4 |          412 |                 64.08 |
|           6 |          174 |                 61.49 |
|           3 |         1020 |                 59.31 |
|           5 |          265 |                 59.25 |
|           7 |          231 |                 54.11 |
|           2 |         1682 |                 53.8  |
|           1 |         4805 |                 45.24 |
|           8 |          104 |                 39.42 |

## 11. Numerical Relationships

Pearson correlations were calculated only for numeric variables plus the binary target. The largest absolute target correlations are RoomService (-0.245), Spa (-0.221), VRDeck (-0.207), TotalSpending (-0.200). This measures linear association only and does not establish predictive value.

## 12. Train/Test Distribution

| Feature     | Train         | Test          |
|:------------|:--------------|:--------------|
| Age         | mean 28.83    | mean 28.66    |
| HomePlanet  | missing 2.31% | missing 2.03% |
| Destination | missing 2.09% | missing 2.15% |
| CryoSleep   | missing 2.50% | missing 2.17% |
| VIP         | missing 2.34% | missing 2.17% |

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
**Evidence:** 5,915 recorded passengers named TRAPPIST-1e as destination.  
**Visualization:** `destination_distribution.png`  
**Why It Matters:** It gives the narrative a concrete destination before comparing outcomes.

### Spending records are mostly quiet, with a long tail
**Evidence:** 3,653 passengers had zero recorded total spending; every spending feature is strongly right-skewed.  
**Visualization:** `spending_distributions.png`  
**Why It Matters:** It makes the economic behavior aboard the ship legible without overstating a cause.

### Cabin location leaves a visible pattern
**Evidence:** Recorded deck rates range from 20.00% to 73.43%.  
**Visualization:** `cabin_deck_vs_transportation.png`  
**Why It Matters:** It motivates a later, careful cabin-feature experiment.

### Passengers were often not travelling alone
**Evidence:** 1,412 of 6,217 groups contain multiple recorded passengers.  
**Visualization:** `group_size_distribution.png`  
**Why It Matters:** It introduces the social structure encoded in PassengerId.
