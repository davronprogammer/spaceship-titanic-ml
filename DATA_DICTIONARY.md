# Data Dictionary

This dictionary reflects the official Kaggle Spaceship Titanic dataset description. Actual data types, completeness, and distributions will be verified during the data audit.

## Dataset Files

| File | Purpose |
| --- | --- |
| `train.csv` | Labelled passenger records used for training and validation. |
| `test.csv` | Unlabelled passenger records for which `Transported` is predicted. |
| `sample_submission.csv` | Example submission with `PassengerId` and `Transported`. |

## Columns

| Column | Present in | Meaning |
| --- | --- | --- |
| `PassengerId` | train, test, submission | Unique passenger identifier in the form `gggg_pp`, where `gggg` is the travelling group and `pp` is the passenger number within that group. Groups are often, but not always, families. |
| `HomePlanet` | train, test | Planet from which the passenger departed; typically their planet of permanent residence. |
| `CryoSleep` | train, test | Whether the passenger elected suspended animation for the voyage. Passengers in cryosleep are confined to their cabins. |
| `Cabin` | train, test | Cabin identifier in the form `deck/num/side`; `side` is `P` (Port) or `S` (Starboard). |
| `Destination` | train, test | Planet where the passenger will disembark. |
| `Age` | train, test | Passenger age. |
| `VIP` | train, test | Whether the passenger paid for special VIP service during the voyage. |
| `RoomService` | train, test | Amount billed for room-service amenities. |
| `FoodCourt` | train, test | Amount billed for food-court amenities. |
| `ShoppingMall` | train, test | Amount billed for shopping-mall amenities. |
| `Spa` | train, test | Amount billed for spa amenities. |
| `VRDeck` | train, test | Amount billed for VR-deck amenities. |
| `Name` | train, test | Passenger first and last names. |
| `Transported` | train, submission | Target: whether the passenger was transported to another dimension. The sample-submission prediction is `True` or `False`. |

Source: [Kaggle — Spaceship Titanic dataset description](https://www.kaggle.com/competitions/spaceship-titanic/data).
