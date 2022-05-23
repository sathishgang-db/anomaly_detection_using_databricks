-- Databricks notebook source
-- MAGIC %md
-- MAGIC Read the json files (each a sensor reading) from the landing location

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE transaction_readings_raw
COMMENT "The raw transaction readings, ingested from /FileStore/tables/transaction_landing_dir"
TBLPROPERTIES ("quality" = "bronze")
AS SELECT * FROM cloud_files("/FileStore/tables/transaction_landing_dir", "json", map("cloudFiles.inferColumnTypes", "true"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Drop nulls from the data set to create the silver table or transformed, cleansed data. This is done by defining the constraint as showm below

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE transaction_readings_cleaned(
  CONSTRAINT valid_transaction_reading EXPECT (AMOUNT IS NOT NULL OR TIME IS NOT NULL) ON VIOLATION DROP ROW
)
TBLPROPERTIES ("quality" = "silver")

COMMENT "Drop all rows with nulls for Time and store these records in a silver delta table"
AS SELECT * FROM STREAM(live.transaction_readings_raw)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Score using the anomaly detection machine learning model, which was registerd as a pandas UDF in the preceding notebook in the Delta Live Table Pipeline

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE predictions
COMMENT "Use the DeepSVDD pandas udf registered in the previous step to predict anomalous transaction readings"
TBLPROPERTIES ("quality" = "gold")
AS SELECT cust_id, detect_anomaly(named_struct("Time", Time,  "V1", V1,  "V2", V2,  "V3", V3,  "V4", V4,  "V5", V5,  "V6", V6,  "V7", V7,  "V8", V8,  "V9", V9,  "V10", V10,  "V11", V11,  "V12", V12,  "V13", V13,  "V14", V14,  "V15", V15,  "V16", V16,  "V17", V17,  "V18", V18,  "V19", V19,  "V20", V20,  "V21", V21,  "V22", V22,  "V23", V23,  "V24", V24,  "V25", V25,  "V26", V26,  "V27", V27,  "V28", V28, "Amount", Amount )) as anomalous from STREAM(live.transaction_readings_cleaned)
