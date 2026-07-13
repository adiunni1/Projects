# Projects
A compilation of projects that I've done. 

# Stock Price Movement Predictor

An end-to-end machine learning pipeline that predicts **whether a stock will go UP or DOWN the next trading day** — framed as a binary classification problem, not a price-prediction problem (which is far more tractable and honest). Built with XGBoost, time-aware validation (including walk-forward validation and a trading-strategy backtest against a SPY buy-and-hold benchmark), and deployed as a serverless REST API on AWS (S3, Lambda, API Gateway).

**Full write-up, code, and deployment guide:** [`stock-predictor/`](stock-predictor/README.md)


# CAHOOTS Welfare Check Analyzer — Cloud ML Pipeline

## What This Is

This project was my first experiment with AWS pipelines, and lifts an existing 11-year welfare check call analysis into a full cloud-native ML pipeline culminating in a live REST API that predicts whether a 911 welfare check call will result in direct assistance based on the esponding agency, year, and call priority.

The underlying analysis is comparing outcomes between the Eugene Police Department
(EPD) and CAHOOTS, Eugene's community-based crisis response program, which was originally
informed a $2.2M civic budget decision. This project re-implements that analysis
as a production-ready cloud data pipeline.

Live API endpoint:

POST https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict


Note: Opening this URL directly in a browser will show {"message":"Not Found"} —
this is expected behavior. The endpoint only accepts POST requests with a JSON body.
Check out the **Try It** Live section below for how to call it.

**Try It Live!**

You can test the live API in three ways, and there's no installation required for options 1 and 2.

Option 1 — Hoppscotch (browser-based, easiest)


Go to hoppscotch.io
Set the method to POST
Paste this URL:


   https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict


Click the Headers tab and confirm content-type is set to application/json
Click the Body tab, set Content Type to application/json, and paste exactly this into the Raw Request Body field:


{"agency": "CAHOOTS", "year": 2022, "priority": 2}


Important: Make sure the body starts with { and not with the word json. If you see json{...} in the body field, delete the json prefix.


Fix the network error: Hoppscotch's default "Browser" mode is blocked by browser security restrictions. When you see "Network Error: Unknown cause", look for the Interceptor panel and switch from Browser to Proxy. Then click Send again.
You should see the prediction response appear on the right side of the screen.


Try swapping "CAHOOTS" for "EPD" in the body to see the contrast in predicted outcomes.

Expected responses:

CAHOOTS call:

json{
  "prediction": "Assisted",
  "confidence": 0.678,
  "inputs": {"agency": "CAHOOTS", "year": 2022, "priority": 2}
}

EPD call (same year, same priority):

json{
  "prediction": "Not Assisted",
  "confidence": 0.878,
  "inputs": {"agency": "EPD", "year": 2022, "priority": 2}
}


Option 2 — curl (terminal)

bash# CAHOOTS call
curl -X POST \
  https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"agency": "CAHOOTS", "year": 2022, "priority": 2}'

# EPD call (same year, same priority — different outcome)
curl -X POST \
  https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"agency": "EPD", "year": 2022, "priority": 2}'


Option 3 — Python

pythonimport requests

response = requests.post(
    "https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict",
    json={"agency": "CAHOOTS", "year": 2022, "priority": 2}
)
print(response.json())


API Parameters

ParameterTypeValuesDefaultagencystring"CAHOOTS" or "EPD""EPD"yearinteger2015 to 20252023priorityinteger1 to 93

## Architecture

```
Raw CSVs (11 files, 1.4M rows)
        |
        v
   Amazon S3
   (raw/)
        |
        v
 clean_cahoots.py
 (Python + boto3 + pandas)
 - Filter welfare check calls
 - Identify CAHOOTS via J-pattern + agency field
 - Recode 40+ outcomes into 8 categories
        |
        v
   Amazon S3
   (clean/wc_clean.csv — 93,189 rows)
        |
        +----------------+
        |                |
        v                v
  Amazon Athena     train_local.py
  (SQL queries      (scikit-learn
   on S3 data)       logistic regression)
                         |
                         v
                    Amazon S3
                    (model-output/model.joblib)
                         |
                         v
               precompute_predictions.py
               (198 predictions cached)
                         |
                         v
                    Amazon S3
                    (model-output/predictions_lookup.json)
                         |
                         v
                 AWS Lambda Function
                 (cahoots-predictor)
                         |
                         v
                 Amazon API Gateway
                 (public REST endpoint)
```

**AWS Services used:** S3, IAM, Athena, Lambda, API Gateway, CloudWatch

---

## Key Findings (reproduced in SQL on cloud data)

| Agency | Assisted Rate | Arrests |
|--------|--------------|---------|
| CAHOOTS | 48.1% | 0 |
| EPD | 9.2% | 875 |

Chi-square test: p < 0.001 across 93,189 welfare check calls, 2015-2025.

The ML model confirmed agency as the strongest predictor of outcome by far
(coefficient: 2.71), more than 13x stronger than priority and 340x stronger
than year.

---

## Live API Usage

**Request:**
```bash
curl -X POST \
  https://91lnkl8d42.execute-api.us-west-2.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"agency": "CAHOOTS", "year": 2022, "priority": 2}'
```

**Response:**
```json
{
  "prediction": "Assisted",
  "confidence": 0.678,
  "inputs": {
    "agency": "CAHOOTS",
    "year": 2022,
    "priority": 2
  }
}
```

**Parameters:**
| Parameter | Type | Values |
|-----------|------|--------|
| agency | string | "CAHOOTS" or "EPD" |
| year | integer | 2015 to 2025 |
| priority | integer | 1 to 9 |

---

## Project Structure

```
cahoots-aws-pipeline/
├── clean_cahoots.py          # Stage 2: reads raw CSVs from S3, cleans, writes back
├── train_local.py            # Stage 3: trains logistic regression, saves model to S3
├── precompute_predictions.py # Stage 3: precomputes all 198 predictions, saves to S3
├── launch_training.py        # Stage 3: SageMaker training job launcher (alt. approach)
├── lambda_package/
│   └── lambda_function.py   # Stage 4: Lambda handler, loads lookup table from S3
├── training/
│   └── train.py             # Stage 3: SageMaker training script
└── notes.txt                # API URL and project notes
```

**S3 bucket structure:**
```
cahoots-pipeline-adiunni/
├── raw/                          # 11 original CSVs (1.4M rows)
├── clean/wc_clean.csv            # Cleaned dataset (93,189 rows)
├── model-output/model.joblib     # Trained logistic regression model
├── model-output/predictions_lookup.json  # Precomputed predictions
└── athena-results/               # Athena SQL query outputs
```

---

## Reproducing This Project

### Prerequisites
- AWS account with IAM user configured via `aws configure`
- Python 3.12 with boto3, pandas, scikit-learn, joblib installed
- S3 bucket created in us-west-2

### Step 1 — Upload raw data
```bash
aws s3 cp /path/to/data/ s3://your-bucket/raw/ --recursive \
  --include "EugeneCAD*noloc.csv"
```

### Step 2 — Run cleaning pipeline
```bash
python clean_cahoots.py
```
This will reads 11 CSVs from S3, filter for welfare checks, identify CAHOOTS calls,
recode outcomes, and write wc_clean.csv back to S3.

### Step 3 — Train model
```bash
python train_local.py
```
Loads wc_clean.csv from S3, trains logistic regression (agency, year, priority
as features, Assisted/Not Assisted as target), saves model.joblib to S3.
Accuracy: 73.7% on held-out test set (15% split).

### Step 4 — Precompute predictions
```bash
python precompute_predictions.py
```
Generates predictions for all 198 combinations of agency (2) x year (11) x
priority (9), saves lookup JSON to S3.

### Step 5 — Deploy Lambda + API Gateway
See AWS console setup instructions in deployment notes. Lambda function loads
the predictions lookup from S3 on first invocation and caches it in memory.

---

## Athena Queries

Connect to the cahoots_db database in AWS Athena (us-west-2) to run SQL
directly against the cleaned S3 data.

**Outcome proportions by agency:**
```sql
SELECT
  agency_clean,
  outcome,
  COUNT(*) as n,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY agency_clean), 1) as pct
FROM cahoots_db.welfare_checks
GROUP BY agency_clean, outcome
ORDER BY agency_clean, pct DESC;
```

**Call volume by year:**
```sql
SELECT yr, agency_clean, COUNT(*) as n
FROM cahoots_db.welfare_checks
GROUP BY yr, agency_clean
ORDER BY yr, agency_clean;
```

**Zero arrest confirmation:**
```sql
SELECT agency_clean, COUNT(*) as total_arrests
FROM cahoots_db.welfare_checks
WHERE outcome = 'Arrest'
GROUP BY agency_clean;
```

---

## ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression (scikit-learn) |
| Features | agency_binary, yr_numeric, priority_numeric |
| Target | 1 = Assisted, 0 = Not Assisted |
| Train/test split | 85% / 15% (stratified) |
| Accuracy | 73.7% |
| Strongest predictor | agency_binary (coefficient: 2.71) |
| CAHOOTS zero arrests | Confirmed across 47,872 calls |

---

## Cost

This project runs almost entirely within the AWS free tier:
- S3: ~$0.00 (well under 5GB free tier limit)
- Athena: ~$0.00 (13MB file, $5/TB scanned)
- Lambda + API Gateway: free tier covers 1M requests/month
- Local training: free (runs on your laptop)

Total estimated cost: under $0.05

---

## Related Project

The original R-based analysis that this pipeline is built on:
[github.com/adiunni1/cahoots-welfare-check-analysis](https://github.com/adiunni1/cahoots-welfare-check-analysis)

---

## Dependencies

```
boto3
pandas
scikit-learn
joblib
numpy
sagemaker==2.232.2
```

Install:
```bash
pip install boto3 pandas scikit-learn joblib numpy sagemaker==2.232.2
```

---

## Author

Adi Unni — github.com/adiunni1


# Appointment Scheduler

## What This Is

This is a a Python implementation of an appointment scheduling system that tracks a list of appointments and automatically detects any **scheduling conflicts**, particularly where cases where two appointments overlap in time.

I made the project is built around two classes: `Appt` (a single appointment) and `Agenda` (a collection of appointments), and demonstrates object-oriented design using Python's special methods to make the objects work naturally with comparison operators and built-in functions.

---

## How It Works

### `Appt` — A Single Appointment

Each appointment has a start time, an end time, and a description. The class defines comparison operators so appointments can be compared and sorted intuitively:

| Operator | Meaning |
|----------|---------|
| `appt1 < appt2` | `appt1` ends before or exactly when `appt2` starts (no overlap) |
| `appt1 > appt2` | `appt1` starts after or exactly when `appt2` ends (no overlap) |
| `appt1 == appt2` | Both appointments cover the exact same time period |

Two helper methods build on these:

- **`overlaps(other)`** — returns `True` if there is any time overlap between the two appointments
- **`intersect(other)`** — returns a new `Appt` representing the overlapping time window, or `None` if there is no overlap

### `Agenda` — A Collection of Appointments

An `Agenda` holds a list of `Appt` objects and provides two key operations:

- **`sort()`** — sorts all appointments by start time
- **`conflicts()`** — returns a new `Agenda` containing only the overlapping portions of any conflicting appointments

The conflict detection algorithm sorts appointments first, then uses a nested loop with an early exit: once it finds an appointment that starts after the current one ends, it can skip the rest (since all subsequent appointments will start even later).

---

## Example

```python
from datetime import datetime
from appointments import Appt, Agenda

appt1 = Appt(datetime(2024, 3, 15, 13, 30), datetime(2024, 3, 15, 15, 30), "Early afternoon nap")
appt2 = Appt(datetime(2024, 3, 15, 15, 0),  datetime(2024, 3, 15, 16, 0),  "Coffee break")

agenda = Agenda()
agenda.append(appt1)
agenda.append(appt2)

conflicts = agenda.conflicts()
print(conflicts)
# 2024-03-15 15:00 15:30 | Overlap
```

---

## Key Design Decisions

- **Comparison operators** (`__lt__`, `__gt__`, `__eq__`) are defined in terms of the *time period*, not the description — two appointments at the same time are "equal" regardless of what they're called
- **`overlaps` is derived from `__lt__` and `__gt__`** rather than implementing its own logic, keeping the definition clean and consistent
- **Conflict detection uses an early exit** — after sorting, once a non-overlapping later appointment is found, the inner loop breaks, making the algorithm more efficient than a naive double loop

---

## Dependencies

- Python 3.10+
- Standard library only (`datetime`)

---

## Files

- **`appointments.py`** — contains both the `Appt` and `Agenda` classes, plus a short demo in the `__main__` block


# Movie Genre Classifier

## What This Is

This project builds a machine learning classifier that predicts whether a movie is a **comedy or thriller** based purely on the words used in its screenplay. I built it on the intuition that a movie would have word-frequency leaning towards the genre it is, and the classifier finds the most similar movies in the training set and uses their genres to make a prediction.

The dataset contains around 5,000 stemmed word features extracted from movie scripts, along with metadata like title, year, and rating.

---

## How It Works

### The Core Idea: K-Nearest Neighbors (k-NN)

Rather than learning explicit rules about what makes a movie a comedy or thriller, k-NN works by similarity: given a new movie, find the *k* most similar movies in the training set and take a majority vote on their genres.

"Similarity" here means **Euclidean distance** across all word-frequency features. Two movies that use words like "laugh" and "marri" (stemmed form of "marry") frequently will be close together in feature space; movies heavy on "dead" and "cop" will cluster differently.

### Pipeline

1. **Data Loading** — load `movies.csv` (one row per movie, columns for each stemmed word) and `stem.csv` (mapping stems back to their original words)
2. **Exploratory Analysis** — visualize word correlations and genre distributions across the dataset
3. **Train/Test Split** — 85% training, 15% test (no shuffling; sequential split)
4. **Feature Selection** — experiment with different word feature subsets to find the most predictive ones
5. **Classification** — for each test movie, compute distances to all training movies and take a majority vote among the *k* nearest neighbors
6. **Evaluation** — measure proportion of correct predictions on the held-out test set

---

## Key Functions

### `distance(features_array1, features_array2)`
Computes Euclidean distance between two movies represented as arrays of word-frequency values. Works across any number of features.

### `classify(test_row, train_rows, train_labels, k)`
Core classifier. Given a test movie's feature vector, finds the *k* nearest neighbors in the training set and returns the most common genre among them.

```python
classify(test_features, train_features, train_labels, k=13)
```

### `classify_feature_row(row)`
Wrapper around `classify` for use with `DataFrame.apply()` — runs the classifier on every row of the test set in one call.

### `most_common(label, table)`
Returns the most frequent value in a column of a DataFrame. Used to tally genre votes among nearest neighbors.

### `su(array)`
Standardizes an array to zero mean and unit standard deviation. Used when computing correlations between word features.

---

## Feature Sets Explored

Two feature sets were tested and compared:

| Feature Set | Words |
|-------------|-------|
| Common words | `i`, `the`, `to`, `a`, `it`, `and`, `that`, `of`, `your`, `what` |
| Genre-specific words | `laugh`, `marri`, `dead`, `heart`, `cop` |

The genre-specific word set captures more meaningful signal for distinguishing comedies from thrillers.

---

## Results

- The k-NN classifier achieved **85% accuracy** on the held-out test set
- Best results used k=13 neighbors and the genre-specific feature set
- Accuracy was measured as the proportion of test movies whose predicted genre matched the actual genre

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn
```

---

## Files

- **`mov_gen_classifier.py`** — main script with all analysis and classification code
- **`movies.csv`** — dataset of movies with stemmed word frequencies and genre labels (not included in repo; required to run)
- **`stem.csv`** — mapping from stemmed words back to their original forms (not included in repo; required to run)

**Duck Machine Assembler, Phase 1**

What This Is

Writing code directly in machine language (raw numbers the computer understands) is painful and error-prone. Assembly language is a human-readable alternative: instead of calculating numeric instruction values by hand, you write things like ADD or STORE, and an assembler translates that into the binary the machine actually runs.

This project is Phase 1 of a two-phase assembler for the Duck Machine, a simulated CPU architecture used in CIS 211 at the University of Oregon. In my mind, Phase 1's job is to take "shorthand" assembly code and resolve it into a fully explicit form that Phase 2 can then convert into machine code.

Specifically, Phase 1 handles two things that raw Duck Machine assembly doesn't support:


Symbolic labels. Instead of calculating memory addresses by hand (e.g., "jump 3 instructions back"), you label a line and refer to it by name
JUMP pseudo-instructions — a cleaner way to write conditional and unconditional branches, which get translated into the actual ADD r15,... instructions the CPU understands


Example

Instead of writing:

     ADD    r1,r1,r0[2]
     STORE  r1,r0,r0[511]
     SUB    r0,r1,r0[10]
     ADD/P  r15,r0,r15[-3]

You can write:

again: ADD    r1,r1,r0[2]
       STORE  r1,r0,r0[511]
       SUB    r0,r1,r0[10]
       JUMP/P again

Phase 1 resolves again to its actual address and rewrites JUMP/P again as ADD/P r15,r0,r15[-3], which Phase 2 can then encode into machine code.


How It Works

Two-Pass Algorithm

Because a label might be used before it's defined (e.g., jumping forward to a label that appears later in the file), the assembler makes two passes through the source code:


Pass 1 (resolve) — reads every line and builds a dictionary mapping each label name to its memory address
Pass 2 (transform) — goes through the lines again and rewrites any instruction that references a label, replacing it with a PC-relative address


PC-relative means the offset is calculated as target_address - current_address, so the code works regardless of where in memory it's loaded.

Line Types

Each line of assembly is matched against one of four patterns:

KindExampleWhat happensCOMMENT# this is a comment or a blank linePassed through unchanged; does not count as a memory addressDATAx: DATA 42Passed through unchanged; counts as one memory wordFULLADD r1,r1,r0[2]Passed through unchanged; already fully resolvedJUMPJUMP/P againRewritten as ADD/P r15,r0,r15[offset] #again

Labels on any line type are recorded during the first pass.


Files


assembler_phase1.py — this file; Phase 1 assembler
assembler_phase2.py (provided) — takes fully resolved assembly and produces object code (machine instructions as integers)
run/asmgo.py (provided) — convenience script that chains Phase 1, Phase 2, and the Duck Machine simulator together in one command



How to Run

Basic usage

bashpython3 assembler_phase1.py input.asm output.asm


input.asm — your assembly source file (can use labels and JUMP)
output.asm — the resolved output, ready for Phase 2


If no files are specified, it reads from stdin and writes to stdout.

Full pipeline (Phase 1 + Phase 2 + run)

bashpython3 run/asmgo.py programs/asm/your_program.asm


Key Functions

resolve(lines) → dict[str, int]

First pass. Scans all lines and maps each label to its memory address. Comment lines don't count toward addresses; all other line types take up one memory word.

transform(lines) → list[str]

Second pass. Calls resolve first to get the label table, then rewrites each line. JUMP instructions become ADD r15,... instructions with a computed PC-relative offset. Lines that don't need changes are passed through as-is.

parse_line(line) → dict

Tries each regex pattern against a line and returns a dictionary of named fields (label, opcode, predicate, target, offset, comment, etc.) plus a kind field indicating which pattern matched. Raises SyntaxError if nothing matches.

resolve_labels(fields, labels, address)

Given parsed fields, the label table, and the current address, computes the PC-relative offset for a label reference.


Error Handling


Syntax errors, unknown labels, and unexpected exceptions are printed to stderr with the offending line number
The assembler stops after 5 errors to avoid flooding output



Dependencies


Python 3.10+
Standard library only (re, argparse, sys, logging)

