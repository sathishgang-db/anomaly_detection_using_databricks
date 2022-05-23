# Databricks notebook source
# MAGIC %md
# MAGIC Import the necessary packages

# COMMAND ----------

# model/grid search tracking
import mlflow
# helper packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pyspark.sql.functions import col
from sklearn.ensemble import IsolationForest
from mlflow.models.signature import infer_signature


# COMMAND ----------

# MAGIC %md
# MAGIC Read the data into a spark dataframe

# COMMAND ----------

username='sathish.gangichetty@databricks.com'
# File location and type
file_location = "/FileStore/tables/creditcard.csv"
input_file_loc = f"/dbfs/Users/{username}/anomaly_detection"
file_type = "csv"
dbutils.fs.mkdirs(input_file_loc)
dbutils.fs.cp(file_location, input_file_loc)
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# COMMAND ----------

display(df)

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC This is an unsupervised problem, but for evaluating the model, we are formatting the target(anomaly) column in a manner consistent with the notation adopted by isolation forests.
# MAGIC i.e. 1 if not fraud, -1 if fraud (consistent with isolation forests). Note fraud is just an example, generally it should be read as anomalous vs non-anomalous

# COMMAND ----------

df_pd['Class'] = df_pd['Class'].apply(lambda x: 1 if x==0 else -1).reset_index(drop=True) # 1 if not fraud, -1 if fraud (consistent with isolation forests)
type(df_pd)

# COMMAND ----------

display(df_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Perform the train test split

# COMMAND ----------

# split data into train (75%) and test (25%) sets
train, test = train_test_split(df_pd, random_state=123)
X_train = train.drop(columns="Class")
X_test = test.drop(columns="Class")
y_train = train["Class"]
y_test = test["Class"]

# COMMAND ----------

# MAGIC %md
# MAGIC Define the function that 
# MAGIC   - Trains the loaded model (either from scratch or warm restart from model registry)
# MAGIC   - Logs, registers and transitions the model to Poduction stage while archiving the previous model in Production stage

# COMMAND ----------

def train_model(loaded_model, model_name, run_name):
  with mlflow.start_run(run_name=run_name) as run:
    loaded_model.fit(X_train)
    y_train_predict = loaded_model.predict(X_train)
    signature = infer_signature(X_train, y_train_predict)
    runID = run.info.run_id
    mlflow.sklearn.log_model(loaded_model, model_name, signature=signature)
    logged_model_URI = 'runs:/{}/{}'.format(runID, model_name)
    model_details = mlflow.register_model(model_uri=logged_model_URI, name=model_name)
    client.transition_model_version_stage(name= model_name, version = model_details.version, stage='Production', archive_existing_versions= True)

  
  

# COMMAND ----------

model_name = 'iforest_avi'
runName = 'avi_iforest_model'

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()


# COMMAND ----------

try:
  latest_model = client.get_latest_versions('model_name', stages=["Production"])[0].source
  loaded_model = mlflow.sklearn.load_model(latest_model)
  train_model(loaded_model, model_name, runName)
  
except :
  isolation_forest = IsolationForest(n_jobs=-1, warm_start=True, random_state=42)
  train_model(isolation_forest, model_name, runName)
  
  

# COMMAND ----------

client.get_latest_versions(model_name, stages=["Production"])[0].source

# COMMAND ----------

# MAGIC %md
# MAGIC Test if the registered model works in a UDF

# COMMAND ----------

sp_df = spark.createDataFrame(X_train)

# COMMAND ----------

import mlflow
logged_model = client.get_latest_versions(model_name, stages=["Production"])[0].source

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on a Spark DataFrame.
columns = list(sp_df.columns)
sp_df = sp_df.withColumn('predictions', loaded_model(*columns))

# COMMAND ----------

display(sp_df)

# COMMAND ----------

#testing again with udf and SQL
sp_df = spark.createDataFrame(X_train)

# COMMAND ----------

spark.udf.register('detect_anomaly', loaded_model)
sp_df.createOrReplaceTempView('sp_df')


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT detect_anomaly(*)
# MAGIC AS anomalous
# MAGIC FROM sp_df

# COMMAND ----------


