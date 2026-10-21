# Client Insider: fraud detection on card transactions

In this project I built a fraud detection model on the Client Insider payment data. I explored the data on Databricks, created new features from each client's history, trained and compared several models with a time-based split, picked a decision threshold, and scored every transaction into a new Delta table (`fraud_scores`). The results are shown in Power BI.

Tools: Databricks (PySpark, Delta Lake), pandas, scikit-learn, XGBoost, LightGBM, matplotlib, Power BI.

**Main result:** the final model is XGBoost with a threshold of 0.9. On the test period it catches 154 of 460 frauds with 392 false alarms, and its PR-AUC (0.2748) is about twice the Logistic Regression baseline.

## Repository structure

```
├── README.md
├── Fraud_Analysis_Official_steps.ipynb      # Databricks notebook, steps 1 to 8, with outputs
├── Fraud_Risk_Analysis_final_version.pbit   # Client Insider report (4 dashboards)
├── Fraud_Score.pbit                         # Fraud Risk report (fraud_scores dashboard)
└── images/                                  # charts from the notebook (steps 6 and 7)
```

## Rules I followed

- The existing tables (`pay_transactions`, `pay_labels`, `pay_users`, `pay_cards`) are only read. The only table I write is the new `fraud_scores`.
- No CSV exports. All the data stays on Databricks and Power BI reads it through the Databricks connector.
- No data leakage. Features only use each client's past, and the model is tested on transactions that happen after the training period.

## Contents

1. [Finding and exploring the data](#1-finding-and-exploring-the-data)
2. [Feature engineering](#2-feature-engineering)
3. [Time-based train / test split](#3-time-based-train--test-split)
4. [Baseline: Logistic Regression](#4-baseline-logistic-regression)
5. [Stronger models](#5-stronger-models)
6. [Comparing the models](#6-comparing-the-models)
7. [Final model and threshold](#7-final-model-and-threshold)
8. [Scoring and Power BI dashboards](#8-scoring-and-power-bi-dashboards)
9. [Limitations](#9-limitations)

---

## 1. Finding and exploring the data

The data is in the `payments.default` schema on Databricks. These are the tables I used:

| Table | Content | Used for |
|---|---|---|
| `pay_transactions` | Card transactions | Features, scoring, dashboards |
| `pay_labels` | `transaction_id`, `is_fraud` (Yes/No) | Fraud label |
| `pay_users` | Client profile (income, debt, credit score...) | Dashboards |
| `pay_cards` | Card profile (brand, type, limit...) | Dashboards |

`pay_transactions` has 1,240,880 rows and 12 columns, all from 2010. The `id` column is a unique transaction ID: 1,240,880 distinct values, no nulls and no duplicates.

A few things I noticed while looking at the data:

- 1,106,058 transactions are swipe and 134,822 are online.
- `amount` goes from -500 (refunds) to 6,820.20, with a mean of 43.70.
- `merchant_state` (134,822) and `zip` (142,042) are empty for online and foreign transactions.
- Only 19,383 transactions have something in `errors`.

**Columns**

| Column | Meaning |
|---|---|
| `id` | Unique transaction ID, used to join the scores in step 8 |
| `date` | Date and time of the transaction |
| `client_id` | Client who made the transaction (links to `pay_users.id`) |
| `card_id` | Card used (links to `pay_cards.id`) |
| `amount` | Amount in USD, negative values are refunds |
| `use_chip` | Channel: Swipe Transaction or Online Transaction |
| `merchant_id` | Merchant ID |
| `merchant_city` | Merchant city (`ONLINE` for online transactions) |
| `merchant_state` | US state, or country for foreign transactions |
| `zip` | Merchant ZIP code |
| `mcc` | Merchant Category Code (type of business) |
| `errors` | Error on the transaction (Insufficient Balance, Bad PIN, Bad CVV...) |

**Is there a fraud label?**

Not inside `pay_transactions`. The labels are in a separate table, `pay_labels` (8,914,963 rows, 13,332 of them fraud), which I joined on `pay_transactions.id = pay_labels.transaction_id`.

| | Transactions |
|---|---|
| In `pay_transactions` | 1,240,880 |
| With a label | 831,529 (67.01%) |
| Labelled as fraud | 2,573 |
| Fraud % on labelled transactions | **0.309%** |
| Fraud % on all transactions | 0.207% |

Note: `pay_users.fraud_risk_level` (Critical / Suspicious / Normal) looks like a label, but it's a rule based on debt and income. I didn't use it as the target.

---

## 2. Feature engineering

I added 11 columns using Spark window functions ordered by time. Each window only looks at the current or earlier transactions of the same client, so nothing comes from the future.

| Feature | What it is |
|---|---|
| `tx_hour` | Hour of the day (0-23) |
| `tx_weekday` | Day of the week (0 = Monday, 6 = Sunday) |
| `is_night` | 1 if the transaction is between 00:00 and 05:59 |
| `client_prev_tx_count` | How many transactions the client made before this one |
| `amount_vs_client_mean` | Amount divided by the client's average amount on previous transactions |
| `tx_count_last_1h` | Other transactions by the same client in the previous hour |
| `tx_count_last_24h` | Other transactions by the same client in the previous 24 hours |
| `seconds_since_prev_tx` | Seconds since the client's previous transaction |
| `is_first_client_merchant` | 1 if it's the first time the client pays this merchant |
| `is_first_client_place` | 1 if it's the first time the client pays in this state / country (or online) |
| `has_error` | 1 if the transaction had an error |

To check for leakage I looked at the first transaction of each of the 1,137 clients: all the "past" features were empty there, which is what should happen.

I also compared the averages between fraud and normal transactions (labelled data only):

| Feature | Not fraud | Fraud |
|---|---|---|
| `is_first_client_merchant` | 8.5% | 62.5% |
| `amount_vs_client_mean` | 1.02 | 3.03 |
| `tx_count_last_24h` | 3.84 | 6.06 |
| `seconds_since_prev_tx` | 28,256 | 14,193 |
| `is_night` | 5.8% | 13.2% |
| `tx_hour` | 12.4 | 9.4 |
| `has_error` | 1.6% | 4.4% |
| `is_first_client_place` | 0.8% | 5.8% |

Fraud happens much more often at a new merchant, with an amount around 3 times the client's usual one, in bursts of transactions and earlier in the day.

---

## 3. Time-based train / test split

I sorted the labelled transactions by date and used the oldest 80% for training and the newest 20% for testing. I didn't shuffle randomly, because in real life the model is trained on the past and used on the future, and a random split would give results that look better than they really are.

| Part | Rows | Date range | Fraud cases | Fraud % |
|---|---|---|---|---|
| Train (oldest ~80%) | 665,223 | 2010-01-01 00:01 to 2010-10-21 10:00 | 2,113 | 0.318% |
| Test (newest ~20%) | 166,306 | 2010-10-21 10:01 to 2010-12-31 23:54 | 460 | 0.277% |
| Total | 831,529 | 2010-01-01 to 2010-12-31 | 2,573 | 0.309% |

The cutoff is 2010-10-21 10:01:00, so every training transaction is earlier than every test transaction. Both parts have fraud cases and the fraud rate is similar (the test rate is 0.87 times the train rate), so there's no big drift in the label.

Since there are only about 3 frauds every 1,000 transactions, accuracy would be misleading. I used Precision, Recall and PR-AUC instead.

---

## 4. Baseline: Logistic Regression

The baseline is a pipeline fitted on the training set only: median imputation (with missing-value flags), `StandardScaler`, then `LogisticRegression(class_weight='balanced')`.

| Model | Precision | Recall | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (balanced), threshold 0.5 | 0.0406 | 0.6826 | 0.1405 | 0.0766 | 0.9203 |
| Random model (reference) | - | - | 0.0028 | - | - |

It catches 314 of the 460 frauds (68%), but it also raises 7,425 false alarms, so roughly 24 alerts for every real fraud. Its PR-AUC is about 50 times better than random, and this is the number the other models have to beat.

The biggest coefficients were the hour of the day (-1.01, fraud happens earlier), the number of transactions in the last 24 hours (+0.81), paying a merchant for the first time (+0.80) and a short gap since the previous transaction (-0.53).

---

## 5. Stronger models

All these models use the same `X_train / y_train` as the baseline and are scored on the same test set.

| Model | Imbalance handling | PR-AUC | ROC-AUC | Precision@0.5 | Recall@0.5 | vs baseline |
|---|---|---|---|---|---|---|
| XGBoost | `scale_pos_weight` = 313.8 | **0.2748** | 0.9563 | 0.1029 | 0.6109 | 1.96x |
| LightGBM | `class_weight='balanced'` | 0.2570 | 0.9551 | 0.1386 | 0.5522 | 1.83x |
| Random Forest | `class_weight='balanced'` | 0.2378 | 0.8644 | 1.0000 | 0.0065 | 1.69x |
| Logistic Regression (baseline) | `class_weight='balanced'` | 0.1405 | 0.9203 | 0.0406 | 0.6826 | 1.00x |
| Isolation Forest (unsupervised) | doesn't use the label | 0.0131 | 0.7874 | - | - | 0.09x |

XGBoost and LightGBM almost double the baseline PR-AUC. Random Forest ranks the frauds quite well, but its probabilities rarely go above 0.5, so at that threshold it only flags 3 transactions. Isolation Forest does much worse than the baseline: without the label, these frauds just don't look like outliers.

---

## 6. Comparing the models

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| XGBoost | 0.1029 | 0.6109 | **0.2748** |
| LightGBM | 0.1386 | 0.5522 | 0.2570 |
| Random Forest | 1.0000 | 0.0065 | 0.2378 |
| Logistic Regression (baseline) | 0.0406 | 0.6826 | 0.1405 |
| Isolation Forest | 0.0227 | 0.0261 | 0.0131 |

Precision and Recall are at the 0.5 threshold. For Isolation Forest I flagged the 0.32% most anomalous transactions, the same share as the fraud rate in train.

![Precision-Recall curves on the test set](images/step6_precision_recall_curves.png)

XGBoost is the best model: it has the highest PR-AUC (0.2748, about twice the baseline), ahead of LightGBM (0.2570), and its curve is above the others at almost every level of recall.

---

## 7. Final model and threshold

I tried several thresholds on the XGBoost probabilities to see how Precision and Recall change:

| Threshold | Precision | Recall | F1 | Alerts | Frauds caught | False alarms | Frauds missed |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.0299 | 0.8239 | 0.0577 | 12,686 | 379 | 12,307 | 81 |
| 0.2 | 0.0484 | 0.7630 | 0.0911 | 7,250 | 351 | 6,899 | 109 |
| 0.3 | 0.0666 | 0.7152 | 0.1218 | 4,941 | 329 | 4,612 | 131 |
| 0.4 | 0.0830 | 0.6543 | 0.1474 | 3,625 | 301 | 3,324 | 159 |
| 0.5 | 0.1029 | 0.6109 | 0.1761 | 2,731 | 281 | 2,450 | 179 |
| 0.6 | 0.1232 | 0.5696 | 0.2026 | 2,127 | 262 | 1,865 | 198 |
| 0.7 | 0.1544 | 0.5326 | 0.2394 | 1,587 | 245 | 1,342 | 215 |
| 0.8 | 0.2004 | 0.4696 | 0.2809 | 1,078 | 216 | 862 | 244 |
| **0.9** | **0.2821** | **0.3348** | **0.3062** | **546** | **154** | **392** | **306** |

![Precision, Recall and F1 by threshold](images/step7_threshold_precision_recall.png)

I chose **0.9**. It has the best F1 of the values I tried, and it keeps the number of alerts manageable for a fraud team: 546 alerts in about 10 weeks, so around 8 a day.

Confusion matrix at threshold 0.9:

| | Predicted not fraud | Predicted fraud |
|---|---|---|
| Actual not fraud | 165,454 | 392 |
| Actual fraud | 306 | 154 |

![Confusion matrix at threshold 0.9](images/step7_confusion_matrix.png)

Top 10 features by XGBoost importance:

| # | Feature | Importance |
|---|---|---|
| 1 | `is_first_client_merchant` | 0.380 |
| 2 | `tx_count_last_24h` | 0.093 |
| 3 | `amount` | 0.082 |
| 4 | `tx_hour` | 0.074 |
| 5 | `tx_weekday` | 0.064 |
| 6 | `is_night` | 0.050 |
| 7 | `has_error` | 0.048 |
| 8 | `client_prev_tx_count` | 0.046 |
| 9 | `amount_vs_client_mean` | 0.045 |
| 10 | `is_first_client_place` | 0.044 |

![Top 10 features of XGBoost](images/step7_feature_importance.png)

**Conclusion**

I went with XGBoost because it has the highest PR-AUC (0.2748), about twice the Logistic Regression baseline. With a threshold of 0.9 it catches 154 of the 460 frauds in the test period (33%) and raises 392 false alarms out of 165,846 normal transactions (0.24%). That means more than 1 alert in 4 is a real fraud, compared to about 1 in 25 for the baseline. If missing a fraud costs more than checking an alert, the threshold could go down to 0.7, which catches 53% of the frauds but gives 1,342 false alarms. The strongest signal by far is a payment at a merchant the client has never used before, followed by the number of transactions in the last 24 hours and the amount.

---

## 8. Scoring and Power BI dashboards

### Scoring every transaction

I applied the XGBoost model and the 0.9 threshold to all 1,240,880 transactions (not only the test set) and saved the result in a new Delta table, `payments.default.fraud_scores`:

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | long | Same ID as `pay_transactions.id`, used for the join |
| `fraud_probability` | double | Fraud probability from 0 to 1 |
| `is_flagged` | boolean | `true` if `fraud_probability` is 0.9 or more |
| `model_name` | string | `xgboost_v1` |
| `scored_at` | timestamp | When the scoring ran (UTC) |

Checks after writing the table:

| Check | Result |
|---|---|
| Rows in `pay_transactions` | 1,240,880 |
| Rows in `fraud_scores` | 1,240,880 (all distinct `transaction_id`) |
| Transactions without a score | 0 |
| Flagged transactions | 10,922 (0.88%), average probability 0.949 |
| Not flagged | 1,229,958, average probability 0.066 |
| Test-set flags vs step 7 | TP = 154, FP = 392, same as the confusion matrix |

The existing tables are untouched: the scores are in their own table and get joined on `transaction_id`. Looking at the preview in the notebook, the highest-risk transactions (probability around 0.998) are all online payments made early in the morning, and the client with the most alerts has 64 flagged transactions out of 3,770 (1.7%).

### Power BI dashboards

There are 5 dashboards split across two Power BI reports, both connected to Databricks:

| # | Dashboard | Report | Main tables |
|---|---|---|---|
| 1 | Overview | `Fraud_Risk_Analysis_final_version.pbit` | `pay_transactions`, `pay_labels`, `pay_users`, `pay_cards` |
| 2 | Merchants | `Fraud_Risk_Analysis_final_version.pbit` | `pay_transactions`, `pay_labels` |
| 3 | Customers | `Fraud_Risk_Analysis_final_version.pbit` | `pay_users`, `pay_transactions`, `pay_labels` |
| 4 | Client Insider | `Fraud_Risk_Analysis_final_version.pbit` | `pay_users`, `pay_labels` |
| 5 | Fraud Risk | `Fraud_Score.pbit` | `fraud_scores`, `pay_transactions` |

All pages have the same layout: a header, filters at the top (transaction date plus a few page-specific ones), a row of KPI cards and the charts below.

#### 1. Overview

Transactions, fraud exposure and payment channels.

- Filters: transaction date, fraud label, gender.
- KPI cards: total transactions, total amount, customers, fraud transactions, fraud amount.
- Charts: a map of the amount by customer location, the monthly amount year over year, amount by card brand, online vs offline amount, fraud transactions by month, and amount by number of credit cards held.

This is the page for a quick picture of volumes, channels and how much of the activity is fraud.

#### 2. Merchants

Where the money goes: cities, merchants, MCC categories and customer segments.

- Filters: transaction date, fraud label, channel (online / offline).
- KPI cards: total amount, transactions, average ticket, active merchants, merchant cities.
- Charts: top 20 merchant cities by amount, amount by city volume level (low / middle / high), amount by customer segment (Boston matrix), top 10 merchants by amount, monthly amount of the top 5 merchants, and a treemap of the top 10 MCC codes.

Useful to see which cities, merchants and business types carry the most money, and where fraud is concentrated when the fraud filter is on.

#### 3. Customers

Income, debt and credit profile of each customer.

- Filters: transaction date, fraud label, channel.
- KPI cards: total yearly income, total debt, average income per capita, customers, average credit score.
- Charts: top 10 customers by yearly income, yearly income vs total debt by generation, and a customer detail table (ID, gender, age, generation, credit band, credit score, income, debt, debt-to-income and fraud risk level) sorted by total debt.

This one is for looking at single customers and comparing their financial profile.

#### 4. Client Insider

Credit quality, indebtedness and fraud-risk segmentation of the customer base.

- Filters: transaction date, fraud label, credit band, customer ID.
- KPI cards: average debt-to-income, average debt per card, average years to retirement, average credit score, and the fraud risk rate (critical + suspicious customers).
- Charts: customers by generation and credit band, customers by credit band (Poor to Excellent), a scatter of debt-to-income vs credit score with one dot per customer, and customers by fraud risk level (Critical / Suspicious / Normal).

It segments the customers by credit quality and by the rule-based risk level.

#### 5. Fraud Risk (from the model)

XGBoost fraud probability for every transaction, model `xgboost_v1`, threshold 0.9.

This is the dashboard I built for step 8. It reads the new `fraud_scores` table, joined to `pay_transactions` with a one-to-one relationship on `transaction_id` (cross filter in both directions).

- Filters: transaction date, flagged by model (true / false), channel.
- KPI cards, built with these DAX measures in `fraud_scores`:

  | Card | Measure |
  |---|---|
  | Scored transactions | `COUNTROWS ( fraud_scores )` |
  | Flagged transactions | `CALCULATE ( COUNTROWS ( fraud_scores ), fraud_scores[is_flagged] = TRUE () )` |
  | Flagged % | `DIVIDE ( [Flagged Transactions], [Scored Transactions] )` |
  | Avg fraud probability | `AVERAGE ( fraud_scores[fraud_probability] )` |
  | Flagged amount | `CALCULATE ( SUM ( pay_transactions[amount] ), fraud_scores[is_flagged] = TRUE () )` |

- Charts:
  - Top 50 highest-risk transactions, sorted by fraud probability, with transaction ID, date, client, amount, merchant city, MCC and flag. This is basically the work list for the fraud team.
  - Flagged transactions by month.
  - Top 15 clients by flagged transactions.
  - A table of flagged transactions per client, with number, % and average probability.

The filters work across the whole page, so picking a client or a month updates the cards and the tables.

I put this dashboard in a separate report instead of adding it to the Client Insider one, so the four existing dashboards stay exactly as they were. Both reports read from the same Databricks schema, and `fraud_scores` can be added to the main report later with the same relationship and measures.

---

## 9. Limitations

- **Only 2010.** `pay_transactions` on Databricks currently has only 2010, even though the original notebook was supposed to load 2010 and 2011. The 2010 load (`mode('overwrite')`) was run again after the 2011 `append`, which removed the 2011 rows and some dashboard columns (`is_online`, `city_amount_level`, `customer_*`). I didn't change the table because of the read-only rule. The Client Insider report still has the older 2010-2011 snapshot, so it shouldn't be fully refreshed until the table is fixed.
- **Label coverage.** Only 67% of the transactions have a label in `pay_labels`, and the model is trained and tested on those.
- **Scores on the training period.** Transactions up to 2010-10-21 were used for training, so their probabilities in `fraud_scores` look better than the real performance. The honest numbers are the test-set ones in step 7.
- **Threshold.** 0.9 was the highest value I tried. A higher threshold, or putting a cost on missed frauds vs false alarms, could change the choice.
