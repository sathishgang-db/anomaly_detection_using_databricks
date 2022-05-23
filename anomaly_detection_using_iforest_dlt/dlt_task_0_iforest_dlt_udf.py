# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow
model_name = 'iforest_avi'
runName = 'avi_iforest_model'
client = mlflow.tracking.MlflowClient()

logged_model = client.get_latest_versions(model_name, stages=["Production"])[0].source

# Load model as a Spark UDF. Override result_type if the model does not return double values.
predict_anomaly = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')


# COMMAND ----------

spark.udf.register("detect_anomaly", predict_anomaly)


# COMMAND ----------


