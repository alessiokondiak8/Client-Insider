# Databricks notebook source
# MAGIC %md
# MAGIC # Data Governance – Payments Feature Engineering
# MAGIC
# MAGIC This notebook documents and checks the features created in the **payments** project (Kaggle: `computingvictor/transactions-fraud-datasets`). It creates three governance tables in `payments.default`:
# MAGIC
# MAGIC | # | Table | Purpose |
# MAGIC |---|-------|---------|
# MAGIC | 1 | `feature_register` | One row per feature: definition, lineage, SQL logic, ownership, privacy, quality rules, known issues |
# MAGIC | 2 | `dq_check_register` | Data quality checks, written as SQL that counts failing rows |
# MAGIC | 3 | `dq_check_results` | Result of every check run (PASS / FAIL), appended at each run |
# MAGIC
# MAGIC Then it copies the feature definitions onto the column comments of the governed tables and shows a governance overview.
# MAGIC
# MAGIC **Governed tables:** `pay_transactions`, `pay_users`, `pay_labels`
# MAGIC
# MAGIC > Run this notebook **after** the feature engineering notebook, so all the feature columns exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Feature Register

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.feature_register (
# MAGIC   feature_id            STRING    COMMENT 'Feature key (F001...)',
# MAGIC   feature_name          STRING    COMMENT 'Column name in the target table',
# MAGIC   business_name         STRING    COMMENT 'Business-friendly name',
# MAGIC   description           STRING    COMMENT 'Business definition',
# MAGIC   data_domain           STRING    COMMENT 'Business domain the feature belongs to',
# MAGIC   source_table          STRING    COMMENT 'Raw table the feature is derived from',
# MAGIC   source_columns        STRING    COMMENT 'Raw columns used (comma separated)',
# MAGIC   depends_on            STRING    COMMENT 'Other derived features it depends on',
# MAGIC   target_table          STRING    COMMENT 'Table where the feature lives',
# MAGIC   granularity           STRING    COMMENT 'Level of one row: User or Transaction',
# MAGIC   script_name           STRING    COMMENT 'Script that creates the feature',
# MAGIC   data_type             STRING    COMMENT 'Output data type',
# MAGIC   feature_category      STRING    COMMENT 'Cleaning / Ratio / Banding / Aggregation / Segmentation / Risk / Label',
# MAGIC   sql_expression        STRING    COMMENT 'SQL logic used to compute the feature',
# MAGIC   null_handling         STRING    COMMENT 'What happens with NULL / zero inputs',
# MAGIC   allowed_values        STRING    COMMENT 'Expected domain or range',
# MAGIC   data_owner            STRING    COMMENT 'Accountable business owner',
# MAGIC   data_steward          STRING    COMMENT 'Day-to-day manager of the definition and quality',
# MAGIC   technical_custodian   STRING    COMMENT 'Team operating the pipeline',
# MAGIC   sensitivity           STRING    COMMENT 'Public / Internal / Confidential / Restricted',
# MAGIC   personal_data         STRING    COMMENT 'Personal data under GDPR: Yes / No',
# MAGIC   regulatory_notes      STRING    COMMENT 'Regulatory or ethical usage notes',
# MAGIC   critical_data_element STRING    COMMENT 'Critical data element: Yes / No',
# MAGIC   data_quality_rule     STRING    COMMENT 'Data quality expectation in plain words',
# MAGIC   dq_check_ids          STRING    COMMENT 'Checks in dq_check_register that test this feature',
# MAGIC   refresh_frequency     STRING    COMMENT 'How often the feature is recomputed',
# MAGIC   retention             STRING    COMMENT 'Retention period',
# MAGIC   status                STRING    COMMENT 'Active / Under review / Deprecated',
# MAGIC   version               STRING    COMMENT 'Feature definition version',
# MAGIC   assumptions           STRING    COMMENT 'Assumptions made when building the feature',
# MAGIC   known_issues          STRING    COMMENT 'Open issues raised during review',
# MAGIC   approved_by           STRING    COMMENT 'Who approved the definition',
# MAGIC   approval_date         DATE      COMMENT 'Date of approval',
# MAGIC   last_updated          TIMESTAMP COMMENT 'When this row was last changed'
# MAGIC )
# MAGIC COMMENT 'Feature register for the payments feature engineering pipeline'

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO payments.default.feature_register (
# MAGIC   feature_id, feature_name, business_name, description, data_domain, source_table, source_columns, depends_on, target_table, granularity, script_name, data_type, feature_category, sql_expression, null_handling, allowed_values, data_owner, data_steward, technical_custodian, sensitivity, personal_data, regulatory_notes, critical_data_element, data_quality_rule, dq_check_ids, refresh_frequency, retention, status, version, assumptions, known_issues, approved_by, approval_date, last_updated
# MAGIC ) VALUES
# MAGIC   ('F001', 'amount', 'Transaction amount', 'Amount of the transaction in USD, converted from text (e.g. \'$-77.00\') to a number. Negative values are refunds', 'Payments / Transactions', 'transactions_data.csv', 'amount', NULL, 'payments.default.pay_transactions', 'Transaction', 'Python load cell (transactions_2010 / transactions_2011)', 'DOUBLE', 'Cleaning', 'transactions[\'amount\'].str[1:].astype(float)', 'Load fails if a value cannot be converted', 'Any number; negative = refund', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', 'Financial transaction data: access limited to payments and analytics teams', 'Yes', 'Never null after conversion', 'DQ001', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Every value starts with a single \'$\' sign', 'Refunds (negative amounts) are included in all sums used by F003, F004-F006', NULL, NULL, current_timestamp()),
# MAGIC   ('F002', 'date', 'Transaction datetime', 'Date and time of the transaction. Only years 2010 and 2011 are loaded', 'Payments / Transactions', 'transactions_data.csv', 'date', NULL, 'payments.default.pay_transactions', 'Transaction', 'Python load cell (filter on year)', 'STRING', 'Cleaning', 'transactions[transactions[\'date\'].str.startswith(\'2010\' / \'2011\')]', 'Rows with NULL date are excluded by the filter', '2010-01-01 to 2011-12-31', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', NULL, 'Yes', 'Only years 2010-2011; no transaction loaded twice', 'DQ002, DQ003', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Date text starts with the 4-digit year', 'Stored as text, not as TIMESTAMP. 2011 is appended with mode(\'append\'): running the cell twice duplicates the whole year', NULL, NULL, current_timestamp()),
# MAGIC   ('F003', 'is_online', 'Online / offline channel', 'ONLINE if the merchant city is \'ONLINE\', otherwise OFFLINE', 'Payments / Transactions', 'transactions_data.csv', 'merchant_city', NULL, 'payments.default.pay_transactions', 'Transaction', 'SQL cell: is_online', 'STRING', 'Segmentation', 'CASE WHEN merchant_city = \'ONLINE\' THEN \'ONLINE\' ELSE \'OFFLINE\' END', 'NULL merchant_city becomes OFFLINE', 'ONLINE, OFFLINE', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', NULL, 'No', 'Only ONLINE or OFFLINE, consistent with merchant_city', 'DQ004', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'All online purchases have merchant_city = \'ONLINE\'', 'Table is rebuilt with CREATE OR REPLACE ... SELECT * FROM itself: re-running the cell fails with a duplicate column error, so the script is not idempotent. Name suggests a boolean but values are text', NULL, NULL, current_timestamp()),
# MAGIC   ('F004', 'city_amount_level', 'Merchant city spending level', 'Merchant cities split into three equal groups by total transaction amount: Low, Middle or High Level', 'Payments / Transactions', 'transactions_data.csv', 'merchant_city, amount', 'F001', 'payments.default.pay_transactions', 'Transaction', 'SQL cell: city_amount_level', 'STRING', 'Banding', 'NTILE(3) OVER (ORDER BY SUM(amount) per merchant_city) -> 1 Low Level, 2 Middle Level, 3 High Level', 'NULL or empty city, or NULL amount -> \'Città non disponibile\'', 'Low Level, Middle Level, High Level, Città non disponibile', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', NULL, 'No', 'Only allowed values; one level per city', 'DQ005', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Thirds are computed over the cities, not over transactions', 'Table is rebuilt with CREATE OR REPLACE ... SELECT * FROM itself: re-running the cell fails with a duplicate column error, so the script is not idempotent. \'ONLINE\' is ranked as if it were a city. Levels change when a new year is loaded. Mixed languages (Italian default value)', NULL, NULL, current_timestamp()),
# MAGIC   ('F005', 'customer_frequency_level', 'Customer purchase frequency', 'Customers split into three equal groups by number of transactions: Bassa, Media or Alta frequenza', 'Payments / Transactions', 'transactions_data.csv', 'client_id', 'F001', 'payments.default.pay_transactions', 'Transaction', 'SQL cell: customer segmentation', 'STRING', 'Banding', 'NTILE(3) OVER (ORDER BY COUNT(*) per client_id)', 'NULL client_id -> \'Non classificato\'', 'Bassa frequenza, Media frequenza, Alta frequenza, Non classificato', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', 'Customer profiling (GDPR Art. 4(4)): use only for analytics, not for automated decisions on the customer', 'No', 'Only allowed values; one level per customer', 'DQ006', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Only transactions with non-null amount are counted', 'Table is rebuilt with CREATE OR REPLACE ... SELECT * FROM itself: re-running the cell fails with a duplicate column error, so the script is not idempotent. Value is a customer attribute repeated on every transaction row', NULL, NULL, current_timestamp()),
# MAGIC   ('F006', 'customer_monetary_level', 'Customer spending value', 'Customers split into three equal groups by total amount spent: Basso, Medio or Alto valore', 'Payments / Transactions', 'transactions_data.csv', 'client_id, amount', 'F001', 'payments.default.pay_transactions', 'Transaction', 'SQL cell: customer segmentation', 'STRING', 'Banding', 'NTILE(3) OVER (ORDER BY SUM(amount) per client_id)', 'NULL client_id -> \'Non classificato\'', 'Basso valore, Medio valore, Alto valore, Non classificato', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', 'Customer profiling (GDPR Art. 4(4)): use only for analytics', 'No', 'Only allowed values; one level per customer', 'DQ006', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Refunds reduce the total', 'Table is rebuilt with CREATE OR REPLACE ... SELECT * FROM itself: re-running the cell fails with a duplicate column error, so the script is not idempotent', NULL, NULL, current_timestamp()),
# MAGIC   ('F007', 'customer_boston_class', 'Customer segment', 'Customer segment from frequency and value thirds: Core Champions, Frequent Value Seekers, High-Value Occasional, Watchlist, Stable', 'Payments / Transactions', 'transactions_data.csv', 'client_id, amount', 'F005, F006', 'payments.default.pay_transactions', 'Transaction', 'SQL cell: customer segmentation', 'STRING', 'Segmentation', 'freq=3 AND value=3 -> Core Champions; freq=3 AND value IN (1,2) -> Frequent Value Seekers; freq IN (1,2) AND value=3 -> High-Value Occasional; freq=1 AND value=1 -> Watchlist; else Stable', 'NULL client_id -> \'Non classificato\'', 'Core Champions, Frequent Value Seekers, High-Value Occasional, Watchlist, Stable, Non classificato', 'Head of Payments', 'Payments Data Steward', 'Data Engineering team', 'Confidential', 'No', 'Customer profiling (GDPR Art. 4(4)): \'Watchlist\' must not trigger actions on the customer without human review', 'Yes', 'Only allowed values; one segment per customer', 'DQ006', 'On every new yearly load (2010, 2011)', 'Life of the project + 5 years', 'Active', '1.0', 'Segments follow a Boston-matrix style split of frequency and value', 'Table is rebuilt with CREATE OR REPLACE ... SELECT * FROM itself: re-running the cell fails with a duplicate column error, so the script is not idempotent. \'Watchlist\' means low activity, not suspected fraud: the name may be misread', NULL, NULL, current_timestamp()),
# MAGIC   ('F008', 'yearly_income', 'Yearly income', 'Customer yearly income in USD, converted from text (\'$59,696\') to a number', 'Customers / Credit risk', 'users_data.csv', 'yearly_income', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users cleaning', 'DOUBLE', 'Cleaning', 'TRY_CAST(REGEXP_REPLACE(yearly_income, \'[$,]\', \'\') AS DOUBLE)', 'Unparseable text becomes NULL (TRY_CAST)', '>= 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'Yes', 'Never null after conversion; not negative', 'DQ008', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Values use \'$\' and \',\' only', 'CREATE OR REPLACE from itself: running the cleaning cell twice fails (column already numeric)', NULL, NULL, current_timestamp()),
# MAGIC   ('F009', 'total_debt', 'Total debt', 'Customer total debt in USD, converted from text to a number', 'Customers / Credit risk', 'users_data.csv', 'total_debt', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users cleaning', 'DOUBLE', 'Cleaning', 'TRY_CAST(REGEXP_REPLACE(total_debt, \'[$,]\', \'\') AS DOUBLE)', 'Unparseable text becomes NULL (TRY_CAST)', '>= 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'Yes', 'Never null after conversion; not negative', 'DQ008', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Values use \'$\' and \',\' only', 'Same re-run issue as F008', NULL, NULL, current_timestamp()),
# MAGIC   ('F010', 'per_capita_income', 'Per capita income of the area', 'Average income per person in the customer\'s area, converted from text to a number', 'Customers / Credit risk', 'users_data.csv', 'per_capita_income', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users cleaning', 'DOUBLE', 'Cleaning', 'TRY_CAST(REGEXP_REPLACE(per_capita_income, \'[$,]\', \'\') AS DOUBLE)', 'Unparseable text becomes NULL (TRY_CAST)', '> 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'No', 'Never null after conversion', 'DQ008', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Refers to the area of the customer\'s address', 'Same re-run issue as F008', NULL, NULL, current_timestamp()),
# MAGIC   ('F011', 'debt_to_income', 'Debt-to-income ratio', 'Total debt divided by yearly income, rounded to 2 decimals', 'Customers / Credit risk', 'users_data.csv', 'total_debt, yearly_income', 'F008, F009', 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'DOUBLE', 'Ratio', 'ROUND(total_debt / NULLIF(yearly_income, 0), 2)', 'NULL when yearly_income is 0 or NULL', '>= 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'Yes', 'Not negative; NULL only when income is 0', 'DQ009', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Higher value means higher indebtedness', NULL, NULL, NULL, current_timestamp()),
# MAGIC   ('F012', 'income_vs_area', 'Income compared with area', 'Yearly income divided by the per capita income of the area', 'Customers / Credit risk', 'users_data.csv', 'yearly_income, per_capita_income', 'F008, F010', 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'DOUBLE', 'Ratio', 'ROUND(yearly_income / NULLIF(per_capita_income, 0), 2)', 'NULL when per_capita_income is 0 or NULL', '>= 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'No', 'Not negative', 'DQ009', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', '1 = earns like the area average', NULL, NULL, NULL, current_timestamp()),
# MAGIC   ('F013', 'years_to_retirement', 'Years to retirement', 'Retirement age minus current age', 'Customers / Credit risk', 'users_data.csv', 'retirement_age, current_age', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'INT', 'Ratio', 'retirement_age - current_age', 'NULL if either age is NULL', 'Any integer; negative = already retired', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'No', 'Not null', 'DQ010', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Ages are up to date at extraction time', 'Category \'Ratio\' used for a difference', NULL, NULL, current_timestamp()),
# MAGIC   ('F014', 'debt_per_card', 'Debt per credit card', 'Total debt divided by the number of credit cards, rounded to whole dollars', 'Customers / Credit risk', 'users_data.csv', 'total_debt, num_credit_cards', 'F009', 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'DOUBLE', 'Ratio', 'ROUND(total_debt / NULLIF(num_credit_cards, 0), 0)', 'NULL when the customer has 0 cards', '>= 0', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'No', 'Not negative', 'DQ009', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'All debt is assumed to be card debt', 'Total debt also includes non-card debt (e.g. mortgage), so the value overstates card debt', NULL, NULL, current_timestamp()),
# MAGIC   ('F015', 'credit_band', 'Credit score band', 'Credit score grouped into FICO-style bands', 'Customers / Credit risk', 'users_data.csv', 'credit_score', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'STRING', 'Banding', '<580 Poor; <670 Fair; <740 Good; <800 Very Good; else Excellent', 'NULL credit_score falls into \'Excellent\' (ELSE branch)', 'Poor, Fair, Good, Very Good, Excellent', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Personal financial data (GDPR): access on a need-to-know basis', 'Yes', 'Only allowed values; credit_score between 300 and 850', 'DQ011, DQ012', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Bands follow the FICO score ranges', 'A NULL credit_score is wrongly labelled \'Excellent\'', NULL, NULL, current_timestamp()),
# MAGIC   ('F016', 'generation', 'Generation', 'Generation of the customer from birth year', 'Customers / Credit risk', 'users_data.csv', 'birth_year', NULL, 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'STRING', 'Banding', '>=1997 Gen Z; >=1981 Millennial; >=1965 Gen X; else Boomer', 'NULL birth_year falls into \'Boomer\' (ELSE branch)', 'Gen Z, Millennial, Gen X, Boomer', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Confidential', 'Yes', 'Age-related attribute: must not be used to make credit or fraud decisions (risk of age discrimination)', 'No', 'Only allowed values', 'DQ012', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Generation cut-offs: 1965, 1981, 1997', 'Customers born before 1946 are also labelled \'Boomer\'. NULL birth_year labelled \'Boomer\'', NULL, NULL, current_timestamp()),
# MAGIC   ('F017', 'risk_segment', 'Credit risk segment', 'Credit risk segment from credit score and debt-to-income: Alto, Medio, Basso', 'Customers / Credit risk', 'users_data.csv', 'credit_score, total_debt, yearly_income', 'F008, F009', 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'STRING', 'Risk', 'credit_score < 620 OR DTI > 1.5 -> Alto; credit_score < 700 OR DTI > 0.8 -> Medio; else Basso', 'NULL inputs fall into \'Basso\' (ELSE branch)', 'Alto, Medio, Basso', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Restricted', 'Yes', 'Credit risk profiling: decisions with significant effect on the customer need human review (GDPR Art. 22)', 'Yes', 'Only allowed values', 'DQ012', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Thresholds 620/700 and DTI 0.8/1.5 are business rules, not learned from data', 'Thresholds not validated against real defaults. Italian labels while credit_band uses English', NULL, NULL, current_timestamp()),
# MAGIC   ('F018', 'fraud_risk_level', 'Rule-based fraud risk level', 'Fraud risk level of the customer from debt, income and number of cards: Critical, Suspicious, Normal', 'Customers / Credit risk', 'users_data.csv', 'total_debt, yearly_income, num_credit_cards, per_capita_income', 'F008, F009, F010', 'payments.default.pay_users', 'User', 'SQL cell: pay_users features', 'STRING', 'Risk', 'debt > 3x income OR (cards >= 6 AND income < area income) -> Critical; debt > 2x income OR income > 5x area income -> Suspicious; else Normal', 'NULL inputs fall into \'Normal\' (ELSE branch)', 'Critical, Suspicious, Normal', 'Head of Credit Risk', 'Customer Data Steward', 'Data Engineering team', 'Restricted', 'Yes', 'Labelling a customer as fraud risk is high-impact: human review required (GDPR Art. 22); document the rule before any use', 'Yes', 'Only allowed values', 'DQ012', 'On every reload of users_data.csv', 'While the customer relationship is active + 5 years (GDPR storage limitation)', 'Active', '1.0', 'Fraud risk can be inferred from financial profile alone', 'Rule not validated against the fraud labels in pay_labels. Customer-level, not transaction-level', NULL, NULL, current_timestamp()),
# MAGIC   ('F019', 'is_fraud', 'Fraud label', 'Confirmed outcome of the transaction: Yes = fraud, No = normal. Target for fraud models', 'Payments / Fraud', 'train_fraud_labels.json', 'target', NULL, 'payments.default.pay_labels', 'Transaction', 'Python load cell (flatten JSON)', 'STRING', 'Label', 'pd.DataFrame(list(raw[\'target\'].items()), columns=[\'transaction_id\', \'is_fraud\'])', 'No NULLs expected', 'Yes, No', 'Head of Fraud Risk', 'Fraud Analytics Data Steward', 'Data Engineering team', 'Restricted', 'No', 'Fraud label: use only for model training and evaluation', 'Yes', 'Only Yes or No; one label per transaction_id', 'DQ013, DQ014', 'On every reload of train_fraud_labels.json', 'Life of the project + 5 years', 'Active', '1.0', 'Labels are final', 'Labels cover all years, while only 2010-2011 transactions are loaded. Stored as text instead of 0/1', NULL, NULL, current_timestamp())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT feature_id, feature_name, target_table, feature_category, sensitivity, personal_data, critical_data_element, dq_check_ids
# MAGIC FROM payments.default.feature_register
# MAGIC ORDER BY feature_id

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Data Quality Check Register
# MAGIC
# MAGIC Each check is a SQL query that returns the **number of failing rows**. A check passes when that number is not above its `threshold`.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE payments.default.dq_check_register (
# MAGIC   dq_check_id  STRING COMMENT 'Check key (DQ001...)',
# MAGIC   check_name   STRING COMMENT 'Short name',
# MAGIC   description  STRING COMMENT 'What the check verifies, in plain words',
# MAGIC   dq_dimension STRING COMMENT 'Completeness / Validity / Uniqueness / Consistency / Integrity',
# MAGIC   target_table STRING COMMENT 'Table the check runs on',
# MAGIC   feature_ids  STRING COMMENT 'Features in feature_register covered by the check',
# MAGIC   check_sql    STRING COMMENT 'SQL returning the number of failing rows',
# MAGIC   threshold    INT    COMMENT 'Maximum failing rows allowed',
# MAGIC   severity     STRING COMMENT 'Critical (blocks use of the data) / Warning (logged)',
# MAGIC   owner        STRING COMMENT 'Who follows up when the check fails',
# MAGIC   frequency    STRING COMMENT 'When the check runs',
# MAGIC   status       STRING COMMENT 'Active / Disabled'
# MAGIC )
# MAGIC COMMENT 'Data quality checks for the payments feature engineering pipeline'

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO payments.default.dq_check_register VALUES
# MAGIC   ('DQ001', 'Amount not null', 'Every transaction has a numeric amount', 'Completeness', 'payments.default.pay_transactions', 'F001', 'SELECT COUNT(*) FROM payments.default.pay_transactions WHERE amount IS NULL', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ002', 'No duplicate transactions', 'Each transaction id appears once (catches a double append of 2011)', 'Uniqueness', 'payments.default.pay_transactions', 'F002', 'SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) AS n FROM payments.default.pay_transactions GROUP BY id HAVING COUNT(*) > 1)', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ003', 'Only years 2010-2011', 'No transaction outside the loaded years', 'Validity', 'payments.default.pay_transactions', 'F002', 'SELECT COUNT(*) FROM payments.default.pay_transactions WHERE substr(date, 1, 4) NOT IN (\'2010\', \'2011\')', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ004', 'is_online consistent', 'is_online is ONLINE exactly when merchant_city is \'ONLINE\'', 'Consistency', 'payments.default.pay_transactions', 'F003', 'SELECT COUNT(*) FROM payments.default.pay_transactions WHERE is_online IS NULL OR is_online <> CASE WHEN merchant_city = \'ONLINE\' THEN \'ONLINE\' ELSE \'OFFLINE\' END', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ005', 'City level valid and unique per city', 'Only allowed values, and one level per merchant city', 'Validity', 'payments.default.pay_transactions', 'F004', 'SELECT (SELECT COUNT(*) FROM payments.default.pay_transactions WHERE city_amount_level NOT IN (\'Low Level\', \'Middle Level\', \'High Level\', \'Città non disponibile\') OR city_amount_level IS NULL) + (SELECT COUNT(*) FROM (SELECT merchant_city FROM payments.default.pay_transactions GROUP BY merchant_city HAVING COUNT(DISTINCT city_amount_level) > 1))', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ006', 'One segment per customer', 'Each client_id has exactly one frequency level, value level and segment', 'Consistency', 'payments.default.pay_transactions', 'F005, F006, F007', 'SELECT COUNT(*) FROM (SELECT client_id FROM payments.default.pay_transactions GROUP BY client_id HAVING COUNT(DISTINCT customer_frequency_level) > 1 OR COUNT(DISTINCT customer_monetary_level) > 1 OR COUNT(DISTINCT customer_boston_class) > 1)', 0, 'Critical', 'Payments Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ007', 'No duplicate users', 'Each user id appears once', 'Uniqueness', 'payments.default.pay_users', 'F008-F018', 'SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) AS n FROM payments.default.pay_users GROUP BY id HAVING COUNT(*) > 1)', 0, 'Critical', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ008', 'Money fields converted', 'yearly_income, total_debt and per_capita_income are numbers (TRY_CAST did not return NULL) and not negative', 'Completeness', 'payments.default.pay_users', 'F008, F009, F010', 'SELECT COUNT(*) FROM payments.default.pay_users WHERE yearly_income IS NULL OR total_debt IS NULL OR per_capita_income IS NULL OR yearly_income < 0 OR total_debt < 0', 0, 'Critical', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ009', 'Ratios not negative', 'debt_to_income, income_vs_area and debt_per_card are not negative', 'Validity', 'payments.default.pay_users', 'F011, F012, F014', 'SELECT COUNT(*) FROM payments.default.pay_users WHERE debt_to_income < 0 OR income_vs_area < 0 OR debt_per_card < 0', 0, 'Warning', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ010', 'Years to retirement present', 'years_to_retirement is filled', 'Completeness', 'payments.default.pay_users', 'F013', 'SELECT COUNT(*) FROM payments.default.pay_users WHERE years_to_retirement IS NULL', 0, 'Warning', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ011', 'Credit score in range', 'credit_score is filled and between 300 and 850 (protects F015 from NULL -> Excellent)', 'Validity', 'payments.default.pay_users', 'F015, F017', 'SELECT COUNT(*) FROM payments.default.pay_users WHERE credit_score IS NULL OR credit_score < 300 OR credit_score > 850', 0, 'Critical', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ012', 'Banding and risk values valid', 'credit_band, generation, risk_segment and fraud_risk_level contain only allowed values', 'Validity', 'payments.default.pay_users', 'F015, F016, F017, F018', 'SELECT COUNT(*) FROM payments.default.pay_users WHERE credit_band NOT IN (\'Poor\', \'Fair\', \'Good\', \'Very Good\', \'Excellent\') OR generation NOT IN (\'Gen Z\', \'Millennial\', \'Gen X\', \'Boomer\') OR risk_segment NOT IN (\'Alto\', \'Medio\', \'Basso\') OR fraud_risk_level NOT IN (\'Critical\', \'Suspicious\', \'Normal\') OR credit_band IS NULL OR generation IS NULL OR risk_segment IS NULL OR fraud_risk_level IS NULL', 0, 'Critical', 'Customer Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ013', 'Labels valid and unique', 'is_fraud is Yes or No and each transaction_id has one label', 'Validity', 'payments.default.pay_labels', 'F019', 'SELECT (SELECT COUNT(*) FROM payments.default.pay_labels WHERE is_fraud NOT IN (\'Yes\', \'No\') OR is_fraud IS NULL) + (SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) AS n FROM payments.default.pay_labels GROUP BY transaction_id HAVING COUNT(*) > 1))', 0, 'Critical', 'Fraud Analytics Data Steward', 'After every run of the feature engineering notebook', 'Active'),
# MAGIC   ('DQ014', 'Transactions without label', 'Loaded transactions that have no fraud label (they cannot be used to train a fraud model)', 'Integrity', 'payments.default.pay_transactions', 'F019', 'SELECT COUNT(*) FROM payments.default.pay_transactions t LEFT ANTI JOIN payments.default.pay_labels l ON t.id = l.transaction_id', 0, 'Warning', 'Fraud Analytics Data Steward', 'After every run of the feature engineering notebook', 'Active')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT dq_check_id, check_name, dq_dimension, target_table, feature_ids, severity
# MAGIC FROM payments.default.dq_check_register
# MAGIC ORDER BY dq_check_id

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Run the Data Quality Checks
# MAGIC
# MAGIC Runs every active check, appends the outcome to `dq_check_results` and shows the result.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS payments.default.dq_check_results (
# MAGIC   run_id       STRING    COMMENT 'Identifier of the run',
# MAGIC   run_ts       TIMESTAMP COMMENT 'When the run happened',
# MAGIC   dq_check_id  STRING    COMMENT 'Check key from dq_check_register',
# MAGIC   check_name   STRING    COMMENT 'Short name of the check',
# MAGIC   target_table STRING    COMMENT 'Table the check ran on',
# MAGIC   failed_rows  BIGINT    COMMENT 'Number of failing rows returned by the check',
# MAGIC   threshold    INT       COMMENT 'Maximum failing rows allowed',
# MAGIC   result       STRING    COMMENT 'PASS / FAIL / ERROR',
# MAGIC   severity     STRING    COMMENT 'Critical / Warning',
# MAGIC   error        STRING    COMMENT 'Error message if the check could not run'
# MAGIC )
# MAGIC COMMENT 'History of data quality check runs'

# COMMAND ----------

import uuid
from datetime import datetime

checks = spark.table("payments.default.dq_check_register").filter("status = 'Active'").orderBy("dq_check_id").collect()
run_id, run_ts = str(uuid.uuid4()), datetime.now()

results = []
for c in checks:
    try:
        failed = int(spark.sql(c.check_sql).collect()[0][0])
        result, error = ("PASS" if failed <= c.threshold else "FAIL"), None
    except Exception as e:
        failed, result, error = None, "ERROR", str(e).splitlines()[0][:300]
    results.append((run_id, run_ts, c.dq_check_id, c.check_name, c.target_table,
                    failed, c.threshold, result, c.severity, error))

results_df = spark.createDataFrame(results, spark.table("payments.default.dq_check_results").schema)
results_df.write.mode("append").saveAsTable("payments.default.dq_check_results")
display(results_df.drop("run_id"))

blocking = [r for r in results if r[7] in ("FAIL", "ERROR") and r[8] == "Critical"]
print("Critical failures or errors:", len(blocking))
print("Data approved for use" if not blocking else "Fix the critical failures before using the data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to read the results
# MAGIC - **PASS**: the rule holds. **FAIL**: `failed_rows` tells how many rows break it. **ERROR**: the check could not run, usually because a column does not exist yet (run the feature engineering notebook first).
# MAGIC - **DQ002 FAIL**: the 2011 cell was run more than once and the year is duplicated. Rebuild `pay_transactions` from scratch.
# MAGIC - **DQ014** is a *Warning*: labels and transactions do not always match. Its count tells how many transactions cannot be used to train a fraud model.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Apply Feature Definitions to the Column Comments
# MAGIC
# MAGIC Copies the business definition of each feature onto its column, so it is visible in Catalog Explorer.

# COMMAND ----------

features = spark.table("payments.default.feature_register").collect()
for f in features:
    table_cols = [c.name for c in spark.table(f.target_table).schema]
    if f.feature_name not in table_cols:
        print(f"Skipped {f.feature_id} {f.feature_name}: column not found in {f.target_table}")
        continue
    comment = f"{f.feature_id} - {f.description}".replace("'", "\\'")
    spark.sql(f"ALTER TABLE {f.target_table} ALTER COLUMN `{f.feature_name}` COMMENT '{comment}'")
print("Column comments updated")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE payments.default.pay_users

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Governance Overview
# MAGIC
# MAGIC ### Features by table and category

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT target_table, feature_category, COUNT(*) AS features
# MAGIC FROM payments.default.feature_register
# MAGIC GROUP BY target_table, feature_category
# MAGIC ORDER BY target_table, feature_category

# COMMAND ----------

# MAGIC %md
# MAGIC ### Personal and restricted data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT feature_id, feature_name, target_table, sensitivity, personal_data, regulatory_notes
# MAGIC FROM payments.default.feature_register
# MAGIC WHERE personal_data = 'Yes' OR sensitivity = 'Restricted'
# MAGIC ORDER BY feature_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Critical data elements

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT feature_id, feature_name, business_name, data_owner, data_quality_rule, dq_check_ids
# MAGIC FROM payments.default.feature_register
# MAGIC WHERE critical_data_element = 'Yes'
# MAGIC ORDER BY feature_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latest data quality run

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT dq_check_id, check_name, target_table, failed_rows, result, severity, error
# MAGIC FROM payments.default.dq_check_results
# MAGIC WHERE run_ts = (SELECT MAX(run_ts) FROM payments.default.dq_check_results)
# MAGIC ORDER BY dq_check_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Open items: known issues and missing approvals

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT feature_id, feature_name, status, known_issues
# MAGIC FROM payments.default.feature_register
# MAGIC WHERE known_issues IS NOT NULL OR approved_by IS NULL
# MAGIC ORDER BY feature_id

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC - **19 features** are documented: 7 on `pay_transactions`, 11 on `pay_users` and 1 on `pay_labels`.
# MAGIC - **Personal data:** all `pay_users` features are personal data under GDPR. `risk_segment`, `fraud_risk_level` and `is_fraud` are *Restricted*.
# MAGIC - **14 data quality checks** cover completeness, validity, uniqueness, consistency and integrity.
# MAGIC - **Main issues found in review:**
# MAGIC   - The feature cells rebuild each table from itself (`CREATE OR REPLACE ... SELECT * FROM` the same table), so they cannot be re-run safely.
# MAGIC   - The 2011 load uses `append`, so running it twice duplicates the year (DQ002).
# MAGIC   - A NULL `credit_score` becomes *Excellent*, and a NULL `birth_year` becomes *Boomer*, because of the `ELSE` branches.
# MAGIC   - `fraud_risk_level` is a business rule that has not been validated against the real fraud labels.
# MAGIC   - Labels mix Italian and English.
# MAGIC - All features are waiting for approval (`approved_by`, `approval_date`).
