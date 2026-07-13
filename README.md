# Projects
A compilation of projects that I've done. 

# Stock Price Movement Predictor

I decided to build an end-to-end machine learning pipeline that predicts **whether a stock will go UP or DOWN the next trading day**  framed as a binary classification problem, not a price-prediction problem (which is far more tractable and honest).

My goal was to practice building time-series feature engineering, XGBoost classification, and deploying ML models as serverless REST APIs on AWS.

**Live API:**
```
POST https://<your-api-gateway-url>/prod/predict
{ "ticker": "AAPL" }
```

---

## Why Classification, Not Regression?

I knew that predicting exact stock prices is notoriously unreliable but redicting *direction* (up or down) is a more tractable problem. It transforms an unbounded regression target into a binary label, and the question "will this go up tomorrow?" maps cleanly to real trading decisions.

---

## Architecture

```
Yahoo Finance API
      │
      ▼
data_pipeline.py        ← Feature engineering + S3 upload
      │
      ▼
  Amazon S3             ← Stores raw features CSV + trained model
      │
      ▼
train_model.py          ← XGBoost, time-aware split, evaluation artifacts
      │
      ▼
  Amazon S3             ← Stores serialised model (.joblib)
      │
      ▼
lambda_function.py      ← Loads model, fetches latest data, returns prediction
      │
      ▼
API Gateway             ← Live REST endpoint
```

**AWS Services:** S3, IAM, Lambda, API Gateway, CloudWatch

---

## Features Engineered

All features are computed from raw OHLCV (Open, High, Low, Close, Volume) data:

| Feature | Description |
|---|---|
| `sma_5/10/20/50` | Simple moving averages — trend direction |
| `ema_5/10/20/50` | Exponential moving averages — recent-weighted trend |
| `price_to_sma20/50` | Close price relative to MA — momentum signal |
| `rsi_14` | Relative Strength Index — overbought/oversold |
| `macd`, `macd_signal`, `macd_hist` | MACD — trend/momentum crossover |
| `bb_width`, `bb_position` | Bollinger Bands — volatility + price position |
| `volume_ratio` | Volume vs. 20-day average — conviction behind moves |
| `daily_return` | Today's percentage return |
| `return_lag_1..5` | Returns from the previous 5 trading days |
| `volatility_10` | 10-day rolling standard deviation of returns |
| `vix_close` | CBOE Volatility Index close — market-wide "fear gauge" |
| `spy_return` | S&P 500 (SPY) daily return — broad market direction/context |

**Target:** `1` if next day's close > today's close, else `0`

---

## Key Design Decision: Time-Aware Train/Test Split

Stock data is sequential. Randomly shuffling before splitting would allow the model to "see the future" during training, a form of data leakage that inflates test accuracy but fails in production.

```
────────────────────────────────────────────────────────
  2015              TRAIN (80%)              2023 │ TEST (20%) │ 2025
────────────────────────────────────────────────────────
```

The model is always trained on the **past** and evaluated on the **future**.

---

## Model: XGBoost Classifier

XGBoost was chosen over alternatives (LSTM, logistic regression) because:
- Handles tabular time-series data with engineered features better than recurrent networks out of the box
- Produces interpretable feature importances — critical for understanding *why* a prediction was made
- Fast to train and tune; no GPU required
- Strong industry adoption in quantitative finance

**Hyperparameters:** 300 estimators, max depth 4, learning rate 0.05, subsample 0.8

---

## Results (Test Set, 2015–2025)

After all four rounds of improvement (oversampling, VIX/SPY features, threshold tuning), actual test-set performance across all 5 tickers:

| Ticker | Accuracy | UP Recall |
|---|---|---|
| AAPL | 44% | 7% |
| MSFT | 50% | 21% |
| GOOGL | 48% | 44% |
| AMZN | 49% | 54% |
| NVDA | 45% | 5% |

> **Note:** A naive "always predict UP" baseline lands around ~53% accuracy (markets go up more than they go down long-term) — none of these tickers beat it. AMZN and GOOGL showed the strongest signal, with UP recall in a more usable 44–54% range. AAPL and NVDA are the weakest: NVDA's 2023–2024 AI-driven rally was such a sustained, structurally-different regime that historical technical patterns from earlier years no longer transferred, and AAPL similarly struggled to find real signal in this window. These numbers are a reminder that the earlier "Model Development Journey" fixes addressed real bugs (the always-predict-DOWN behavior, the miscalibrated threshold) without guaranteeing genuine predictive edge — imbalance and threshold fixes make a model's predictions *sane*, not necessarily *accurate*.

---

## Walk-Forward Validation

A single 80/20 train/test split only shows performance on one slice of history. To check whether that performance holds up across different market conditions, `train_model.py` also supports **walk-forward validation**: train on a rolling 3-year window, test on the following 6 months, then step forward 6 months and repeat — from 2018 through 2025.

```bash
python train_model.py --ticker AAPL --bucket your-bucket-name --walkforward
```

Results are printed as a summary table and saved to `s3://your-bucket-name/stock-predictor/{ticker}_walkforward.csv`.

### Results (AAPL, 8 windows, 2018–2025)

| Window | Train Period | Test Period | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | 2018-01 → 2021-01 | 2021-01 → 2021-07 | 0.50 | 48.4% | 48.4% | 23.8% | 0.319 |
| 2 | 2018-07 → 2021-07 | 2021-07 → 2022-01 | 0.50 | 51.6% | 72.7% | 22.2% | 0.340 |
| 3 | 2019-01 → 2022-01 | 2022-01 → 2022-07 | 0.50 | 48.4% | 43.5% | 34.5% | 0.385 |
| 4 | 2019-07 → 2022-07 | 2022-07 → 2023-01 | 0.50 | 50.4% | 46.6% | 45.8% | 0.462 |
| 5 | 2020-01 → 2023-01 | 2023-01 → 2023-07 | 0.45 | 48.4% | 55.6% | 49.3% | 0.522 |
| 6 | 2020-07 → 2023-07 | 2023-07 → 2024-01 | 0.50 | 50.8% | 54.8% | 58.0% | 0.563 |
| 7 | 2021-01 → 2024-01 | 2024-01 → 2024-07 | 0.50 | 46.0% | 48.0% | 56.3% | 0.518 |
| 8 | 2021-07 → 2024-07 | 2024-07 → 2025-01 | 0.50 | 39.1% | 50.0% | 1.3% | 0.025 |
| **Average** | | | | **47.9%** | **52.4%** | **36.4%** | **0.392** |

> **Key finding — performance is regime-dependent, not stable.** UP recall rises from 23.8% in window 1 to a peak in windows 5–7 (Jan 2023–Jul 2024), where it holds in the 49–58% range — the model's strongest, most usable stretch. Window 8 (Jul–Dec 2024) then collapses to near-zero recall (1.3%), missing almost every actual up day in that period despite accuracy only dropping to 39%. That combination — recall falling off a cliff while accuracy stays in a plausible range — is a signature of the model reverting to predicting the majority class once the market regime it was trained on stops matching the test period.
>
> The broadly steady 1→6 climb also suggests **recent training data carries more signal than older data**: each window's 3-year train set slides forward in time, and performance improves as older, less-relevant history rolls out of it. That motivates trying a **shorter training window** (e.g. 1–2 years instead of 3) in a future iteration — dropping stale data faster may help the model adapt to regime shifts like the one that broke window 8, instead of diluting recent patterns with years-old ones.

### Cross-Ticker Comparison (All 5 Tickers, 8 Windows Each)

Running the same walk-forward validation on all 5 tickers shows how much AAPL's numbers generalize:

| Ticker | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 | Window 8 UP Recall |
|---|---|---|---|---|---|
| AAPL | 47.9% | 52.4% | 36.4% | 0.392 | 1.3% |
| MSFT | 50.0% | 52.0% | 49.8% | 0.487 | 66.2% |
| GOOGL | 51.2% | 59.9% | 53.7% | 0.509 | 51.4% |
| AMZN | 47.8% | 50.4% | 44.7% | 0.420 | 17.9% |
| NVDA | 49.4% | 56.0% | 45.8% | 0.454 | 77.3% |

> **Key finding — the window 8 collapse is stock-specific to AAPL, not a market-wide regime shift.** If Jul–Dec 2024 had broken the model for every ticker, that would point to a broad macro shock the model couldn't handle. Instead, only AAPL's UP recall collapsed to near-zero (1.3%) in window 8 — MSFT (66.2%), GOOGL (51.4%), and NVDA (77.3%) all saw window-8 recall *above* their own averages, and AMZN dipped (17.9% vs. a 44.7% average) but nowhere near AAPL's near-total miss. That rules out "the whole market changed regime" as the explanation and points instead to something AAPL-specific in that stretch — consistent with the earlier note in [Results](#results-test-set-20152025) that AAPL's technical indicators struggled to find signal in this window even in the single-split evaluation.

---

## Backtesting

Walk-forward validation shows the model's predictive edge is modest and regime-dependent — the next question is whether that edge is worth anything in dollar terms. `backtest.py` simulates a simple trading strategy on each ticker's existing test period (Jan 2023–Dec 2024): starting with $10,000, hold the stock from close to close whenever the model predicts UP, sit in cash whenever it predicts DOWN, and compare the result against a SPY buy-and-hold benchmark starting with the same $10,000.

```bash
python backtest.py --ticker AAPL --bucket your-bucket-name
```

### Results (Jan 2023 – Dec 2024)

![GOOGL Model Strategy vs. SPY Buy & Hold](GOOGL_backtest.png)

| Ticker | Model Strategy Return | Final Value | Gap vs. SPY |
|---|---|---|---|
| AAPL | -0.27% | $9,972.61 | -51.5 pts |
| MSFT | +30.07% | $13,007.24 | -21.2 pts |
| GOOGL | +42.85% | $14,284.68 | -8.4 pts |
| AMZN | +28.11% | $12,810.82 | -23.1 pts |
| NVDA | +32.49% | $13,249.32 | -18.7 pts |
| **SPY Buy & Hold** | **+51.22%** | **$15,122.20** | — |

> **Key finding — four of five strategies were profitable, but none beat SPY.** MSFT, GOOGL, AMZN, and NVDA all ended the two-year period with real gains (+28% to +43%); AAPL was the outlier, finishing essentially flat with a small loss (-0.27%), consistent with its weak walk-forward and single-split results elsewhere in this README. GOOGL came closest to the benchmark, trailing SPY by only 8.4 percentage points — still a meaningful gap, but the smallest of the five, which is why its chart is shown above.
>
> This isn't surprising given the model's ~44–52% test-set accuracy reported earlier: Jan 2023–Dec 2024 was a strong, sustained bull market (SPY +51% in two years), and beating buy-and-hold through a rally like that is an extremely high bar — professional active managers routinely fail to clear it too. A more revealing test of whether this model earns its keep would be **risk-adjusted returns** (Sharpe ratio, max drawdown) rather than raw returns, and/or performance during **bear or sideways markets**, where a model that can rotate to cash on predicted-DOWN days has more room to add value over a benchmark that can only fall with the market. That's a natural next analysis rather than something this backtest currently measures.

---

## Model Development Journey

The model went through several rounds of debugging and iteration after the initial baseline underperformed badly on recent data:

1. **Baseline XGBoost — ~44% accuracy.** The first version, trained on technical indicators alone, barely beat a coin flip and was worse than the "always predict UP" naive baseline. Digging into the confusion matrix showed the model was defaulting to DOWN almost every time — UP recall was only ~9%.

2. **Class balancing (oversampling).** The label distribution is mildly imbalanced (~53% UP / 47% DOWN), but that alone didn't explain a model that predicted DOWN 90%+ of the time. `scale_pos_weight` was tried first and didn't move the needle enough, so the approach switched to `RandomOverSampler` (imbalanced-learn), duplicating minority-class rows in the *train* split only (never the test split, to avoid leaking duplicated test-like rows into evaluation) until it was 50/50. Model capacity was also reduced (`max_depth=3`, `min_child_weight=5`, fewer estimators) to fight overfitting on the noisier training signal.

3. **Market-context features (VIX + SPY).** Technical indicators derived purely from AAPL's own OHLCV data describe the stock in isolation — they say nothing about the broader market regime. Two features were added: `vix_close` (CBOE Volatility Index — the market's "fear gauge") and `spy_return` (S&P 500 daily return — broad market direction), both merged onto the AAPL date index and forward-filled. The idea: a stock's next-day move is influenced by whether the whole market is risk-on or risk-off, not just its own recent price action.

4. **Decision threshold tuning.** Even with balanced training data, the default 0.5 classification threshold turned out to be a poor cutoff for this problem — it swept in a train-set threshold search (0.30–0.50 in steps of 0.05, maximizing F1) and applied the winning threshold to test-set predictions. This threshold is persisted to the metrics CSV in S3 and read by `lambda_function.py` at inference time, so the live API and the offline evaluation stay consistent.

5. **News sentiment (Finnhub + VADER) — no improvement.** A `sentiment_score` feature was added by pulling daily headlines per ticker from the Finnhub API, scoring each with NLTK's VADER sentiment analyzer, and merging the daily mean compound score onto the feature set (forward-filled, defaulting to 0.0 on days with no news). The hypothesis was that market-wide VIX/SPY features miss stock-specific news events that move a single name independent of the broader market. In practice it **didn't help and hurt one ticker**: AMZN's average UP recall dropped from 54% to 31%, while the other four tickers were essentially unchanged. Three likely reasons:
   - **VADER is tuned for social media, not financial news.** It was built and validated on tweets and short informal text, where sentiment cues (punctuation, capitalization, slang) work differently than in headline-style financial reporting, where "growth slows" and "misses estimates" carry strong negative signal VADER's lexicon isn't tuned to catch.
   - **Finnhub's free tier has sparse historical coverage.** Company-news history is thin the further back you query, so a large fraction of training rows fall back to the neutral 0.0 default rather than a real sentiment signal — diluting whatever signal exists in the days that do have coverage.
   - **Daily aggregation loses intraday timing.** Averaging all of a day's headlines into one score discards *when* during the day a headline landed relative to market open/close, and mixes pre-market catalysts with after-hours noise into a single number.

   This is a **known limitation**, not a bug — the feature is live in the pipeline (`sentiment_score` in `FEATURE_COLS`) but isn't currently pulling its weight. A finance-specific language model like **FinBERT** (pretrained on financial text rather than general/social-media text) is a more promising direction than swapping sentiment sources within the same VADER-based approach.

**Honest caveat:** even after five rounds of improvement, technical indicators — even augmented with VIX/SPY context and news sentiment — have a limited ceiling on 2023–2024 AAPL data. That period included unusually concentrated conditions (the AI/mega-cap rally, a fast Fed hiking cycle, and a handful of outsized single-day moves around earnings) that don't resemble the more "normal" price action technical indicators are typically evaluated against. The model's edge over a naive baseline in this window is real but modest, and shouldn't be mistaken for a robust trading signal.

---

## What I Learned

- **A model that "looks" broken (always predicting one class) is usually a class-imbalance or threshold problem before it's an architecture problem.** Reaching for a fancier model before checking the confusion matrix would have wasted time — the fix here was data balancing and threshold calibration, not a different algorithm.
- **`scale_pos_weight` and oversampling are not interchangeable, even though both "address imbalance."** `scale_pos_weight` reweights the loss function but leaves the data untouched, which can be too weak a signal for gradient-boosted trees with shallow depth. Oversampling changes what the trees actually split on. Worth trying both rather than assuming the textbook answer works.
- **The default 0.5 threshold is an assumption, not a law.** It's only optimal when classes are balanced and false positives/negatives are equally costly — neither is guaranteed to hold, and tuning it on the train set (never the test set) was a cheap, high-leverage fix.
- **A stock's own technical indicators are an incomplete picture.** Two features describing the *entire market's* mood (VIX, SPY) meaningfully diversified the feature set beyond "what has AAPL's price been doing."
- **Time-aware splitting matters even more once you start balancing classes.** It would have been easy to oversample before splitting and leak duplicated rows across train/test — worth double-checking the split boundary every time the pipeline changes.
- **Backtest-period selection matters.** 2023–2024 was an unusually distinctive market regime for AAPL; results here shouldn't be assumed to generalize to calmer periods or to other tickers without re-validation.

---

## How to Run

### 1. Set up AWS credentials
```bash
aws configure
# Enter your Access Key ID, Secret Access Key, region (e.g. us-west-2)
```

### 2. Create an S3 bucket
```bash
aws s3 mb s3://your-bucket-name
```

### 3. Run the data pipeline
```bash
pip install yfinance xgboost scikit-learn pandas numpy matplotlib seaborn boto3 joblib
python data_pipeline.py --ticker AAPL --bucket your-bucket-name
```

### 4. Train the model
```bash
python train_model.py --ticker AAPL --bucket your-bucket-name
```

### 5. Deploy Lambda + API Gateway
See [deploy.md](deploy.md) for the full step-by-step CLI walkthrough — packaging dependencies, IAM role + policy JSON, environment variables, and the HTTP API Gateway route.

### 6. Call the live API
```bash
curl -X POST https://<your-api-gateway-url>/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**Response:**
```json
{
  "ticker": "AAPL",
  "prediction": "UP",
  "confidence": 0.6134,
  "latest_close": 189.42,
  "latest_date": "2025-01-10",
  "model_version": "xgb_v1"
}
```

---

## Project Files

| File | Purpose |
|---|---|
| `data_pipeline.py` | Fetch data, engineer features (incl. sentiment), upload to S3 |
| `train_model.py` | Time-aware split, walk-forward validation, XGBoost training, evaluation + plots |
| `backtest.py` | Simulates the model's trading strategy vs. a SPY buy-and-hold benchmark |
| `lambda_function.py` | Serverless inference handler for API Gateway |
| `run_all.py` | Runs the pipeline + training for a batch of tickers |
| `deploy.md` | Step-by-step Lambda + API Gateway deployment guide |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Next Steps

- [ ] **Compare XGBoost vs. LSTM** on the same feature set — worth checking whether a sequence model captures temporal patterns (e.g. multi-day momentum regimes) that flattened, per-row technical indicators miss.
- [ ] **Replace VADER with FinBERT for sentiment scoring** — the current Finnhub + VADER `sentiment_score` feature (see [Model Development Journey](#model-development-journey)) didn't move the needle and hurt AMZN recall; a finance-specific language model trained on financial text rather than social media is a more promising source of sentiment signal.
- [ ] Add more tickers (portfolio-level signals)
- [ ] Add EventBridge to retrain the model weekly on fresh data
- [ ] Backtest a simple trading strategy using the model's signals

---

## Tech Stack

**Languages:** Python  
**ML:** XGBoost, scikit-learn  
**Data:** yfinance (Yahoo Finance API), pandas, NumPy  
**Visualisation:** Matplotlib, Seaborn  
**AWS:** S3, IAM, Lambda, API Gateway, CloudWatch  
**Serialisation:** joblib


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

