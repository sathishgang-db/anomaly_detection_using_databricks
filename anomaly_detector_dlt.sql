-- Databricks notebook source
-- MAGIC %md
-- MAGIC Read the json files (each a sensor reading) from the landing location

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE transaction_readings_raw
COMMENT "The raw transaction readings, ingested from /FileStore/tables/transaction_json_landing"
AS SELECT * FROM cloud_files("/FileStore/tables/transaction_json_landing", "json", map("cloudFiles.inferColumnTypes", "true"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Drop nulls from the data set to create the silver table or transformed, cleansed data. This is done by defining the constraint as showm below

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE transaction_readings_cleaned(
  CONSTRAINT valid_transaction_reading EXPECT (CUST_ID IS NOT NULL OR TIME IS NOT NULL OR V1 IS NOT NULL OR V2 IS NOT NULL OR V3 IS NOT NULL OR V4 IS NOT NULL OR V5 IS NOT NULL OR V6 IS NOT NULL OR V7 IS NOT NULL OR V8 IS NOT NULL OR V9 IS NOT NULL OR V10 IS NOT NULL OR V11 IS NOT NULL OR V12 IS NOT NULL OR V13 IS NOT NULL OR V14 IS NOT NULL OR V15 IS NOT NULL OR V16 IS NOT NULL OR V17 IS NOT NULL OR V18 IS NOT NULL OR V19 IS NOT NULL OR V20 IS NOT NULL OR V21 IS NOT NULL OR V22 IS NOT NULL OR V23 IS NOT NULL OR V24 IS NOT NULL OR V25 IS NOT NULL OR V26 IS NOT NULL OR V27 IS NOT NULL OR V28 IS NOT NULL ) ON VIOLATION DROP ROW
)
COMMENT "Drop all rows with nulls and store these records in a silver delta table"
AS SELECT * FROM STREAM(live.transaction_readings_raw)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Score using the anomaly detection machine learning model, which was registerd as a pandas UDF in the preceding notebook in the Delta Live Table Pipeline

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE predictions
COMMENT "Use the DeepSVDD pandas udf registered in the previous step to predict anomalous transaction readings"
AS SELECT cust_id, detect_anomaly(*) as anomalous from STREAM(live.transaction_readings_cleaned)
