# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install tensorflow
# MAGIC %pip install pyod
# MAGIC %pip install rich

# COMMAND ----------

from pyod.models.deep_svdd import DeepSVDD
import pyod
from pyod.utils.data import evaluate_print
import mlflow
import tempfile
import joblib
import os,shutil
import numpy as np
import pyspark.pandas as ps
import tensorflow.keras as keras
import pandas as pd
import mlflow.pyfunc
import tensorflow as tf
import glob
from mlflow.tracking import MlflowClient
from typing import Iterator
from pyspark.sql.functions import pandas_udf
import dlt

# COMMAND ----------

# MAGIC %md Batch scoring with Pandas UDF - Use Spark Cluster to Score the Model

# COMMAND ----------

client = MlflowClient()
experiment_id = client.get_experiment_by_name("/Users/sathish.gangichetty@databricks.com/anomaly_detection_deepsvdd_dbr").experiment_id
# get most recent run by sorting 
most_recent_run = sorted([{"run_id":i.run_id, "end_time":i.end_time} 
                        for i in client.list_run_infos(experiment_id=experiment_id)
                        if i.status=='FINISHED'],
                       key=lambda x: x['end_time'], 
                       reverse=True)[0]
most_recent_run

# COMMAND ----------

# MAGIC %md
# MAGIC Get the path of the logged model from Experiments/Users/avinash.sooriyarachchi@databricks.com/anomaly_detection_blog/anomaly_detection_deepsvdd_pyfuncRun afa28bf3ae4f4bbc9c2775dbf930155f

# COMMAND ----------

@pandas_udf("double")
def udf_predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
  logged_model = f'runs:/{most_recent_run["run_id"]}/model'
  client.download_artifacts(most_recent_run['run_id'], "deepsvdd","")
  deepSVDD = joblib.load(glob.glob("deepsvdd/*.joblib")[0])
  deepSVDD.model_ =  mlflow.keras.load_model(logged_model)
  
  for features in iterator:
    predict_df = pd.concat(features, axis=1)
    yield pd.Series(deepSVDD.predict(predict_df))
    
spark.udf.register("detect_anomaly", udf_predict)

features = ['cust_id', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

@dlt.create_table(
  comment="anomaly detection model for identifying OOD data",  
  table_properties={
    "quality": "gold"
  }    
)
def anomaly_predictions():
  return dlt.read("transaction_readings_cleaned").withColumn('predictions', udf_predict(*features))