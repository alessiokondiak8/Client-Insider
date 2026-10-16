# Databricks notebook source
# MAGIC %md
# MAGIC # Payments Data Transformation Pipeline
# MAGIC
# MAGIC This notebook loads the **Transactions Fraud Datasets** from Kaggle into Databricks and transforms the raw data into analysis-ready tables. The new columns describe the **channel** of each transaction, the **spending level** of merchant cities, the **segment** of each customer and the **financial and risk profile** of each user.
# MAGIC
# MAGIC **Dataset:** [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) on Kaggle
# MAGIC
# MAGIC ## Pipeline Steps:
# MAGIC
# MAGIC ### 1. Data Loading (Python)
# MAGIC - Downloads the dataset with `kagglehub`
# MAGIC - Loads transactions for **2010 and 2011**, cards, users and fraud labels
# MAGIC - Cleans the transaction `amount` (text `$-77.00` → number `-77.0`)
# MAGIC - Saves everything as Delta tables in `payments.default`
# MAGIC
# MAGIC ### 2. Transaction Features (SQL) — `pay_transactions`
# MAGIC - **Channel:** `is_online` (ONLINE / OFFLINE)
# MAGIC - **City spending level:** `city_amount_level` (Low / Middle / High Level)
# MAGIC - **Customer segmentation:** `customer_frequency_level`, `customer_monetary_level`, `customer_boston_class`
# MAGIC
# MAGIC ### 3. User Features (SQL) — `pay_users`
# MAGIC - **Cleaning:** `yearly_income`, `total_debt`, `per_capita_income` (text → number)
# MAGIC - **Financial ratios:** `debt_to_income`, `income_vs_area`, `debt_per_card`, `years_to_retirement`
# MAGIC - **Bands:** `credit_band`, `generation`
# MAGIC - **Risk:** `risk_segment`, `fraud_risk_level`
# MAGIC
# MAGIC ### 4. Final Check
# MAGIC - Overview of the final tables and of the new columns
# MAGIC
# MAGIC | Table | Grain (one row = ) | Columns added |
# MAGIC |-------|--------------------|---------------|
# MAGIC | `pay_transactions` | one transaction | 5 |
# MAGIC | `pay_users` | one user | 9 (+ 3 cleaned) |
# MAGIC | `pay_cards` | one card | – |
# MAGIC | `pay_labels` | one labelled transaction | – |
# MAGIC
# MAGIC > **Run the notebook once, from top to bottom (Run all).** Every transformation rebuilds a table from itself, so running a single cell twice fails or duplicates data. To start again, rerun the whole notebook from step 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Data Loading
# MAGIC
# MAGIC Install `kagglehub` and restart Python so the package is available.

# COMMAND ----------

# MAGIC %pip install --upgrade kagglehub

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the dataset and read the files
# MAGIC
# MAGIC The dataset has four files:
# MAGIC - `transactions_data.csv`: one row per card transaction (over 13 million rows, 2010–2019)
# MAGIC - `cards_data.csv`: one row per card
# MAGIC - `users_data.csv`: one row per customer
# MAGIC - `train_fraud_labels.json`: fraud label (`Yes` / `No`) for each transaction id

# COMMAND ----------

import kagglehub
import pandas as pd
import json

path = kagglehub.dataset_download("computingvictor/transactions-fraud-datasets")
print("Path to dataset files:", path)

transactions = pd.read_csv(f"{path}/transactions_data.csv")
cards = pd.read_csv(f"{path}/cards_data.csv")
users = pd.read_csv(f"{path}/users_data.csv")

with open(f"{path}/train_fraud_labels.json", "r") as f:
    raw = json.load(f)

print("Transactions:", transactions.shape)
print("Cards:       ", cards.shape)
print("Users:       ", users.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fraud labels: from nested JSON to a table
# MAGIC
# MAGIC The JSON file has the form `{"target": {"7476734": "No", ...}}`. It is flattened into a table with two columns: `transaction_id` (converted to integer, to match the transaction `id`) and `is_fraud`.

# COMMAND ----------

labels = pd.DataFrame(
    list(raw["target"].items()),
    columns=["transaction_id", "is_fraud"]
)
labels["transaction_id"] = labels["transaction_id"].astype(int)
labels.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transactions 2010: filter and clean the amount
# MAGIC
# MAGIC Only transactions from **2010** are kept at first, to keep the table small enough to work with.
# MAGIC
# MAGIC The `amount` column is text with a dollar sign (`$-77.00`). `.str[1:]` removes the first character (`$`) and `.astype(float)` turns the rest into a number. Negative amounts are **refunds**.

# COMMAND ----------

transactions_2010 = transactions[transactions['date'].str.startswith('2010')].copy()
transactions_2010['amount'] = transactions_2010['amount'].str[1:].astype(float)
print("Transactions 2010:", transactions_2010.shape)
transactions_2010.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the tables to Databricks (Delta)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS payments.default")

def save(pdf, table):
    spark.createDataFrame(pdf).write.mode('overwrite') \
        .option("overwriteSchema", "true").saveAsTable(table)
    print(f"{table}: {spark.table(table).count():,} rows")

save(transactions_2010, "payments.default.pay_transactions")
save(cards, "payments.default.pay_cards")
save(users, "payments.default.pay_users")
save(labels, "payments.default.pay_labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add transactions 2011
# MAGIC
# MAGIC The 2011 transactions get the same cleaning and are **appended** (`mode('append')`) to the table, which then contains 2010 and 2011.
# MAGIC
# MAGIC > Run this cell only once: running it again adds 2011 a second time.

# COMMAND ----------

transactions_2011 = transactions[transactions['date'].str.startswith('2011')].copy()
transactions_2011['amount'] = transactions_2011['amount'].str[1:].astype(float)
transactions_2011_spark = spark.createDataFrame(transactions_2011)
transactions_2011_spark.write.mode('append').saveAsTable("payments.default.pay_transactions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT substr(date, 1, 4) AS year, COUNT(*) AS transactions, ROUND(SUM(amount), 0) AS total_amount
# MAGIC FROM payments.default.pay_transactions
# MAGIC GROUP BY substr(date, 1, 4)
# MAGIC ORDER BY year

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Transaction Features — `pay_transactions`
# MAGIC
# MAGIC ### 2.1 Channel: `is_online`
# MAGIC
# MAGIC **What:** marks each transaction as `ONLINE` or `OFFLINE`.
# MAGIC
# MAGIC **How:** in this dataset, online purchases have no physical shop, so their `merchant_city` is the word `ONLINE`. Every other value is a real city, so the purchase happened in a shop.
# MAGIC
# MAGIC **Why:** online and in-store payments behave differently (amounts, fraud patterns), so the channel is one of the most useful ways to split the data.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_transactions AS
# MAGIC SELECT *, CASE WHEN merchant_city = 'ONLINE' THEN 'ONLINE'
# MAGIC ELSE 'OFFLINE'
# MAGIC END AS is_online
# MAGIC FROM payments.default.pay_transactions;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** number of transactions and amount per channel.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT is_online,
# MAGIC        COUNT(*)                                            AS transactions,
# MAGIC        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)  AS pct_transactions,
# MAGIC        ROUND(AVG(amount), 2)                               AS avg_amount
# MAGIC FROM payments.default.pay_transactions
# MAGIC GROUP BY is_online

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 City spending level: `city_amount_level`
# MAGIC
# MAGIC **What:** classifies each merchant city as **Low**, **Middle** or **High Level** by the total money spent there.
# MAGIC
# MAGIC **How**, in three steps:
# MAGIC 1. **`city_totals`:** adds up `amount` for each `merchant_city`, skipping empty cities and missing amounts.
# MAGIC 2. **`city_groups`:** sorts the cities from lowest to highest total and splits them into **three groups of equal size** with `NTILE(3)`: the bottom third becomes *Low Level*, the middle third *Middle Level* and the top third *High Level*.
# MAGIC 3. **Final `SELECT`:** joins the level back to every transaction of that city with a `LEFT JOIN`. Transactions whose city has no level (empty or missing city) get `'Città non disponibile'` through `COALESCE`.
# MAGIC
# MAGIC **Why:** it shows whether a transaction happens in a place where a lot of money is spent or in a minor location.
# MAGIC
# MAGIC **Things to know:**
# MAGIC - The thirds are made over **cities**, not over transactions. Each group has the same number of cities, not the same number of transactions.
# MAGIC - `ONLINE` is treated like a city, and with its large total it is likely to end up in *High Level*.
# MAGIC - Refunds (negative amounts) reduce the city total.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_transactions AS
# MAGIC
# MAGIC WITH city_totals AS (
# MAGIC     SELECT
# MAGIC         merchant_city,
# MAGIC         SUM(amount) AS city_total_amount
# MAGIC     FROM payments.default.pay_transactions
# MAGIC     WHERE merchant_city IS NOT NULL
# MAGIC       AND trim(merchant_city) <> ''
# MAGIC       AND amount IS NOT NULL
# MAGIC     GROUP BY merchant_city
# MAGIC ),
# MAGIC
# MAGIC city_groups AS (
# MAGIC     SELECT
# MAGIC         merchant_city,
# MAGIC         CASE NTILE(3) OVER (ORDER BY city_total_amount)
# MAGIC             WHEN 1 THEN 'Low Level'
# MAGIC             WHEN 2 THEN 'Middle Level'
# MAGIC             WHEN 3 THEN 'High Level'
# MAGIC         END AS city_amount_level
# MAGIC     FROM city_totals
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     t.*,
# MAGIC     COALESCE(c.city_amount_level, 'Città non disponibile') AS city_amount_level
# MAGIC FROM payments.default.pay_transactions AS t
# MAGIC LEFT JOIN city_groups AS c
# MAGIC     ON t.merchant_city = c.merchant_city;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** cities and transactions in each level.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT city_amount_level,
# MAGIC        COUNT(DISTINCT merchant_city)  AS cities,
# MAGIC        COUNT(*)                       AS transactions,
# MAGIC        ROUND(SUM(amount), 0)          AS total_amount
# MAGIC FROM payments.default.pay_transactions
# MAGIC GROUP BY city_amount_level
# MAGIC ORDER BY total_amount DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Customer segmentation (Boston-matrix style)
# MAGIC
# MAGIC **What:** gives each customer three labels, based on **how often** they buy (frequency) and **how much** they spend (monetary value).
# MAGIC
# MAGIC **How**, in four steps:
# MAGIC 1. **`customer_aggregation`:** for each `client_id`, counts the transactions (`customer_n_transactions`) and adds up the amounts (`customer_total_amount`).
# MAGIC 2. **`customer_scores`:** splits the customers into thirds twice with `NTILE(3)`: once by number of transactions (`frequency_tile`) and once by total amount (`monetary_tile`). In both cases 1 = bottom third and 3 = top third.
# MAGIC 3. **`customer_segments`:** turns the tiles into readable labels:
# MAGIC    - `customer_frequency_level`: Bassa / Media / Alta frequenza
# MAGIC    - `customer_monetary_level`: Basso / Medio / Alto valore
# MAGIC    - `customer_boston_class`: combines the two, as in the matrix below
# MAGIC 4. **Final `SELECT`:** adds the three labels to every transaction of the customer. Customers that could not be scored get `'Non classificato'`.
# MAGIC
# MAGIC **The segments** (`customer_boston_class`):
# MAGIC
# MAGIC | | Value: Basso (1) | Value: Medio (2) | Value: Alto (3) |
# MAGIC |---|---|---|---|
# MAGIC | **Frequency: Alta (3)** | Frequent Value Seekers | Frequent Value Seekers | **Core Champions** |
# MAGIC | **Frequency: Media (2)** | Stable | Stable | High-Value Occasional |
# MAGIC | **Frequency: Bassa (1)** | **Watchlist** | Stable | High-Value Occasional |
# MAGIC
# MAGIC - **Core Champions:** buy often and spend a lot. These are the best customers.
# MAGIC - **Frequent Value Seekers:** buy often but spend little or a medium amount.
# MAGIC - **High-Value Occasional:** buy rarely but spend a lot.
# MAGIC - **Watchlist:** buy rarely and spend little. These customers are at risk of becoming inactive. The name does **not** mean they are suspected of fraud.
# MAGIC - **Stable:** all the other combinations.
# MAGIC
# MAGIC **Why:** segments summarise thousands of transactions into a few customer types, which can then be compared (for example, fraud rate by segment).
# MAGIC
# MAGIC **Things to know:** the segment is a customer attribute, so it is repeated on every transaction of that customer. It is computed on all the years loaded, so it changes if more years are added.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_transactions AS
# MAGIC
# MAGIC WITH customer_aggregation AS (
# MAGIC     SELECT
# MAGIC         client_id,
# MAGIC         COUNT(*) AS customer_n_transactions,
# MAGIC         SUM(amount) AS customer_total_amount
# MAGIC     FROM payments.default.pay_transactions
# MAGIC     WHERE client_id IS NOT NULL
# MAGIC       AND amount IS NOT NULL
# MAGIC     GROUP BY client_id
# MAGIC ),
# MAGIC
# MAGIC customer_scores AS (
# MAGIC     SELECT
# MAGIC         client_id,
# MAGIC         customer_n_transactions,
# MAGIC         customer_total_amount,
# MAGIC
# MAGIC         NTILE(3) OVER (
# MAGIC             ORDER BY customer_n_transactions
# MAGIC         ) AS frequency_tile,
# MAGIC
# MAGIC         NTILE(3) OVER (
# MAGIC             ORDER BY customer_total_amount
# MAGIC         ) AS monetary_tile
# MAGIC
# MAGIC     FROM customer_aggregation
# MAGIC ),
# MAGIC
# MAGIC customer_segments AS (
# MAGIC     SELECT
# MAGIC         client_id,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN frequency_tile = 1 THEN 'Bassa frequenza'
# MAGIC             WHEN frequency_tile = 2 THEN 'Media frequenza'
# MAGIC             WHEN frequency_tile = 3 THEN 'Alta frequenza'
# MAGIC         END AS customer_frequency_level,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN monetary_tile = 1 THEN 'Basso valore'
# MAGIC             WHEN monetary_tile = 2 THEN 'Medio valore'
# MAGIC             WHEN monetary_tile = 3 THEN 'Alto valore'
# MAGIC         END AS customer_monetary_level,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN frequency_tile = 3
# MAGIC              AND monetary_tile = 3
# MAGIC                 THEN 'Core Champions'
# MAGIC
# MAGIC             WHEN frequency_tile = 3
# MAGIC              AND monetary_tile IN (1, 2)
# MAGIC                 THEN 'Frequent Value Seekers'
# MAGIC
# MAGIC             WHEN frequency_tile IN (1, 2)
# MAGIC              AND monetary_tile = 3
# MAGIC                 THEN 'High-Value Occasional'
# MAGIC
# MAGIC             WHEN frequency_tile = 1
# MAGIC              AND monetary_tile = 1
# MAGIC                 THEN 'Watchlist'
# MAGIC
# MAGIC             ELSE 'Stable'
# MAGIC         END AS customer_boston_class
# MAGIC
# MAGIC     FROM customer_scores
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     t.*,
# MAGIC
# MAGIC     COALESCE(
# MAGIC         s.customer_frequency_level,
# MAGIC         'Non classificato'
# MAGIC     ) AS customer_frequency_level,
# MAGIC
# MAGIC     COALESCE(
# MAGIC         s.customer_monetary_level,
# MAGIC         'Non classificato'
# MAGIC     ) AS customer_monetary_level,
# MAGIC
# MAGIC     COALESCE(
# MAGIC         s.customer_boston_class,
# MAGIC         'Non classificato'
# MAGIC     ) AS customer_boston_class
# MAGIC
# MAGIC FROM payments.default.pay_transactions AS t
# MAGIC
# MAGIC LEFT JOIN customer_segments AS s
# MAGIC     ON t.client_id = s.client_id;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** customers in each segment, with their average transactions and spending.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT customer_boston_class,
# MAGIC        COUNT(DISTINCT client_id)                                     AS customers,
# MAGIC        ROUND(COUNT(*) / COUNT(DISTINCT client_id), 0)                AS avg_transactions_per_customer,
# MAGIC        ROUND(SUM(amount) / COUNT(DISTINCT client_id), 0)             AS avg_amount_per_customer
# MAGIC FROM payments.default.pay_transactions
# MAGIC GROUP BY customer_boston_class
# MAGIC ORDER BY avg_amount_per_customer DESC

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** the full frequency × value matrix (number of customers in each cell).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT customer_frequency_level,
# MAGIC        COUNT(DISTINCT CASE WHEN customer_monetary_level = 'Basso valore' THEN client_id END) AS basso_valore,
# MAGIC        COUNT(DISTINCT CASE WHEN customer_monetary_level = 'Medio valore' THEN client_id END) AS medio_valore,
# MAGIC        COUNT(DISTINCT CASE WHEN customer_monetary_level = 'Alto valore'  THEN client_id END) AS alto_valore
# MAGIC FROM payments.default.pay_transactions
# MAGIC GROUP BY customer_frequency_level
# MAGIC ORDER BY customer_frequency_level

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. User Features — `pay_users`
# MAGIC
# MAGIC ### 3.1 Cleaning the money columns
# MAGIC
# MAGIC **What:** turns `yearly_income`, `total_debt` and `per_capita_income` from text into numbers.
# MAGIC
# MAGIC **How:** the raw values look like `$59,696`. `REGEXP_REPLACE(..., '[$,]', '')` removes every `$` and `,`, leaving `59696`, and `TRY_CAST(... AS DOUBLE)` converts it into a number. `TRY_CAST` returns **NULL** instead of failing if a value cannot be converted.
# MAGIC
# MAGIC **Why:** money stored as text cannot be added, divided or compared, so none of the ratios below would be possible.
# MAGIC
# MAGIC The query lists the columns to keep explicitly: the three money columns are replaced by their cleaned version, and all the other columns are kept as they are.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_users AS
# MAGIC SELECT
# MAGIC   id, address, gender, birth_month, birth_year, current_age, retirement_age,
# MAGIC   latitude, longitude, credit_score, num_credit_cards,
# MAGIC   TRY_CAST(REGEXP_REPLACE(yearly_income,     '[$,]', '') AS DOUBLE) AS yearly_income,
# MAGIC   TRY_CAST(REGEXP_REPLACE(total_debt,        '[$,]', '') AS DOUBLE) AS total_debt,
# MAGIC   TRY_CAST(REGEXP_REPLACE(per_capita_income, '[$,]', '') AS DOUBLE) AS per_capita_income
# MAGIC FROM payments.default.pay_users;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** the converted columns are numeric, with no values lost in the conversion.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*)                                        AS users,
# MAGIC        SUM(CASE WHEN yearly_income IS NULL THEN 1 ELSE 0 END)     AS null_yearly_income,
# MAGIC        SUM(CASE WHEN total_debt IS NULL THEN 1 ELSE 0 END)        AS null_total_debt,
# MAGIC        SUM(CASE WHEN per_capita_income IS NULL THEN 1 ELSE 0 END) AS null_per_capita_income,
# MAGIC        ROUND(AVG(yearly_income), 0)                    AS avg_yearly_income,
# MAGIC        ROUND(AVG(total_debt), 0)                       AS avg_total_debt
# MAGIC FROM payments.default.pay_users

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Financial ratios, bands and credit risk
# MAGIC
# MAGIC This step adds seven columns in one query.
# MAGIC
# MAGIC **Ratios.** `NULLIF(x, 0)` turns a zero into NULL, so a division by zero gives NULL instead of an error.
# MAGIC
# MAGIC | Column | Formula | Meaning |
# MAGIC |--------|---------|---------|
# MAGIC | `debt_to_income` | total_debt / yearly_income | How many years of income the debt is worth. Above 1 = debt larger than a year's income |
# MAGIC | `income_vs_area` | yearly_income / per_capita_income | Income compared with the area average. 1 = average, 2 = double the average |
# MAGIC | `years_to_retirement` | retirement_age − current_age | Years left before retirement. Negative = already retired |
# MAGIC | `debt_per_card` | total_debt / num_credit_cards | Average debt for each card held |
# MAGIC
# MAGIC **Bands:**
# MAGIC
# MAGIC | `credit_band` (credit score) | | `generation` (birth year) | |
# MAGIC |---|---|---|---|
# MAGIC | below 580 | Poor | 1997 or later | Gen Z |
# MAGIC | 580–669 | Fair | 1981–1996 | Millennial |
# MAGIC | 670–739 | Good | 1965–1980 | Gen X |
# MAGIC | 740–799 | Very Good | before 1965 | Boomer |
# MAGIC | 800 or more | Excellent | | |
# MAGIC
# MAGIC The credit bands follow the ranges of the FICO score used in the US.
# MAGIC
# MAGIC **Credit risk** (`risk_segment`): conditions are checked in order, and the first one that is true decides the segment.
# MAGIC
# MAGIC | Segment | Condition |
# MAGIC |---------|-----------|
# MAGIC | **Alto** | credit score below 620, **or** debt more than 1.5 × yearly income |
# MAGIC | **Medio** | credit score below 700, **or** debt more than 0.8 × yearly income |
# MAGIC | **Basso** | everyone else |
# MAGIC
# MAGIC **Things to know:** a `CASE` without a matching condition goes to the `ELSE` branch. So a missing `credit_score` becomes *Excellent* and a missing `birth_year` becomes *Boomer*.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_users AS
# MAGIC SELECT *,
# MAGIC   ROUND(total_debt / NULLIF(yearly_income,0), 2)            AS debt_to_income,
# MAGIC   ROUND(yearly_income / NULLIF(per_capita_income,0), 2)     AS income_vs_area,
# MAGIC   retirement_age - current_age                               AS years_to_retirement,
# MAGIC   ROUND(total_debt / NULLIF(num_credit_cards,0), 0)         AS debt_per_card,
# MAGIC   CASE WHEN credit_score < 580 THEN 'Poor'
# MAGIC        WHEN credit_score < 670 THEN 'Fair'
# MAGIC        WHEN credit_score < 740 THEN 'Good'
# MAGIC        WHEN credit_score < 800 THEN 'Very Good'
# MAGIC        ELSE 'Excellent' END                                  AS credit_band,
# MAGIC   CASE WHEN birth_year >= 1997 THEN 'Gen Z'
# MAGIC        WHEN birth_year >= 1981 THEN 'Millennial'
# MAGIC        WHEN birth_year >= 1965 THEN 'Gen X'
# MAGIC        ELSE 'Boomer' END                                     AS generation,
# MAGIC   CASE WHEN credit_score < 620 OR total_debt/NULLIF(yearly_income,0) > 1.5 THEN 'Alto'
# MAGIC        WHEN credit_score < 700 OR total_debt/NULLIF(yearly_income,0) > 0.8 THEN 'Medio'
# MAGIC        ELSE 'Basso' END                                      AS risk_segment
# MAGIC FROM payments.default.pay_users;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** users in each risk segment, with their average ratios.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT risk_segment,
# MAGIC        COUNT(*)                            AS users,
# MAGIC        ROUND(AVG(credit_score), 0)         AS avg_credit_score,
# MAGIC        ROUND(AVG(debt_to_income), 2)       AS avg_debt_to_income,
# MAGIC        ROUND(AVG(income_vs_area), 2)       AS avg_income_vs_area,
# MAGIC        ROUND(AVG(debt_per_card), 0)        AS avg_debt_per_card
# MAGIC FROM payments.default.pay_users
# MAGIC GROUP BY risk_segment
# MAGIC ORDER BY avg_credit_score

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** users by credit band and by generation.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'credit_band' AS band_type, credit_band AS band, COUNT(*) AS users FROM payments.default.pay_users GROUP BY credit_band
# MAGIC UNION ALL
# MAGIC SELECT 'generation', generation, COUNT(*) FROM payments.default.pay_users GROUP BY generation
# MAGIC ORDER BY band_type, users DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Rule-based fraud risk: `fraud_risk_level`
# MAGIC
# MAGIC **What:** labels each user as **Critical**, **Suspicious** or **Normal**, based on financial signals that can indicate fraud (for example a synthetic identity or a stolen identity).
# MAGIC
# MAGIC **How:** conditions are checked in order.
# MAGIC
# MAGIC | Level | Condition | Why it is a signal |
# MAGIC |-------|-----------|--------------------|
# MAGIC | **Critical** | debt more than 3 × yearly income | Debt far beyond what the income could repay |
# MAGIC | **Critical** | 6 or more cards **and** income below the area average | Many cards with a low income: possible credit "bust-out" |
# MAGIC | **Suspicious** | debt more than 2 × yearly income | Very high indebtedness |
# MAGIC | **Suspicious** | income more than 5 × the area average | Income unusually high for the area: possibly declared falsely |
# MAGIC | **Normal** | everyone else | |
# MAGIC
# MAGIC **Things to know:** these thresholds are **business rules chosen by hand**, not learned from data. Before relying on them, compare them with the real fraud labels in `pay_labels`. Users with missing values fall into *Normal*.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.pay_users AS
# MAGIC SELECT *,
# MAGIC   CASE WHEN total_debt > yearly_income * 3
# MAGIC          OR (num_credit_cards >= 6 AND yearly_income < per_capita_income) THEN 'Critical'
# MAGIC        WHEN total_debt > yearly_income * 2
# MAGIC          OR yearly_income > per_capita_income * 5 THEN 'Suspicious'
# MAGIC        ELSE 'Normal' END AS fraud_risk_level
# MAGIC FROM payments.default.pay_users;

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** users per fraud risk level, compared with the credit risk segment.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT fraud_risk_level,
# MAGIC        COUNT(*)                                                  AS users,
# MAGIC        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)        AS pct_users,
# MAGIC        SUM(CASE WHEN risk_segment = 'Alto'  THEN 1 ELSE 0 END)   AS risk_alto,
# MAGIC        SUM(CASE WHEN risk_segment = 'Medio' THEN 1 ELSE 0 END)   AS risk_medio,
# MAGIC        SUM(CASE WHEN risk_segment = 'Basso' THEN 1 ELSE 0 END)   AS risk_basso
# MAGIC FROM payments.default.pay_users
# MAGIC GROUP BY fraud_risk_level
# MAGIC ORDER BY users

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Final Check
# MAGIC
# MAGIC ### Rows in each table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'pay_transactions' AS table_name, COUNT(*) AS row_count FROM payments.default.pay_transactions
# MAGIC UNION ALL SELECT 'pay_users',  COUNT(*) FROM payments.default.pay_users
# MAGIC UNION ALL SELECT 'pay_cards',  COUNT(*) FROM payments.default.pay_cards
# MAGIC UNION ALL SELECT 'pay_labels', COUNT(*) FROM payments.default.pay_labels

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample of the transformed transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT id, date, client_id, amount, merchant_city,
# MAGIC        is_online, city_amount_level,
# MAGIC        customer_frequency_level, customer_monetary_level, customer_boston_class
# MAGIC FROM payments.default.pay_transactions
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample of the transformed users

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT id, current_age, credit_score, yearly_income, total_debt,
# MAGIC        debt_to_income, income_vs_area, debt_per_card, years_to_retirement,
# MAGIC        credit_band, generation, risk_segment, fraud_risk_level
# MAGIC FROM payments.default.pay_users
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do the new features relate to real fraud?
# MAGIC
# MAGIC Joins the transactions with the fraud labels to compare the fraud rate across channels and customer segments.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT t.is_online,
# MAGIC        t.customer_boston_class,
# MAGIC        COUNT(*)                                                     AS labelled_transactions,
# MAGIC        SUM(CASE WHEN l.is_fraud = 'Yes' THEN 1 ELSE 0 END)          AS frauds,
# MAGIC        ROUND(AVG(CASE WHEN l.is_fraud = 'Yes' THEN 1.0 ELSE 0 END) * 100, 3) AS fraud_rate_pct
# MAGIC FROM payments.default.pay_transactions AS t
# MAGIC JOIN payments.default.pay_labels AS l
# MAGIC   ON t.id = l.transaction_id
# MAGIC GROUP BY t.is_online, t.customer_boston_class
# MAGIC ORDER BY fraud_rate_pct DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Conclusions
# MAGIC
# MAGIC _Write 3–5 sentences in your own words after running the notebook, for example:_
# MAGIC - How many transactions were loaded for 2010 and 2011, and what share is online?
# MAGIC - How are customers spread across the segments, and which segment spends most?
# MAGIC - What share of users is in the *Alto* credit risk segment, and what share is *Critical* for fraud risk?
# MAGIC - Is the fraud rate higher online, or in some customer segments?
