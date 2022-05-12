# Databricks notebook source
# MAGIC %pip install pyod
# MAGIC %pip install rich

# COMMAND ----------

# MAGIC %md Import all the packages we need

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

(df.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema","true")
 .option("path",input_file_loc)
 .saveAsTable('sgfs.cc_data')
)

# COMMAND ----------

# MAGIC %md Convert to pandas on spark, for quick data manipulations

# COMMAND ----------

ps.set_option('compute.default_index_type', 'distributed')
input_df = df.to_pandas_on_spark()
input_df.head()

# COMMAND ----------

input_df.reset_index(inplace=True)
input_df.rename(columns={'index':'cust_id'}, inplace=True)
selected_cols = [column for column in input_df.columns if column not in ('Amount')]
sub_df = input_df[selected_cols]
train_df = sub_df[sub_df['Class']==0]
test_anom_df = sub_df[sub_df['Class']==1] #instead of train test split, we're going to split as "good data" vs "not", the assumption is we generally don't know what bad is
train_cols = [column for column in train_df.columns if column not in ('Class')]
train_df = train_df[train_cols]

# COMMAND ----------

# MAGIC %md Perform a model run (increase epochs for a real usecase), log the model and the associate pyOD class for model reconstruction at inference time.

# COMMAND ----------

with mlflow.start_run(run_name="anomaly_detection") as run:
  # define params for the model
  mlflow.autolog(log_input_examples=True)
  contamination = 0.01
  clf_name = 'DeepSVDD'
  # borrow from the lib
  clf = DeepSVDD(use_ae=False, epochs=1,
                 random_state=123, contamination=.005)
  clf.fit(train_df.to_numpy())
  # log param
  mlflow.log_param("contamination",contamination)
  # get truth data
  y_train = sub_df[sub_df['Class']==0]['Class'].to_numpy()
  # apply model
  y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
  y_train_scores = clf.decision_scores_  # raw outlier scores
  # get test/unseen data & apply model
  X_test = test_anom_df[train_cols].to_numpy()
  y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
  y_test_scores = clf.decision_function(X_test)  # outlier scores
  # log metric to count anomalies "flagged"
  mlflow.log_metric("train_anomalies",y_train_pred.sum()/len(y_train_pred))
  mlflow.log_metric("test_anomalies",y_test_pred.sum()/len(y_test_pred))

  runID = run.info.run_id
  experimentID = run.info.experiment_id
  # The keras model is wrapped by pyod DeepSVDD class, mlflow saves the model - we have to save the state of the class as well to replicate results
  # See here - https://github.com/yzhao062/pyod/issues/328
  # And here - https://github.com/yzhao062/pyod/blob/76959794abde783a486e831265bb3300c1c65b1b/pyod/models/deep_svdd.py#L186
  clf.model_ = None
  temp = tempfile.NamedTemporaryFile(prefix="deepsvdd-", suffix=".joblib")
  temp_name = temp.name
  print(f"{temp.name} is the name")
  try:
    joblib.dump(clf,temp_name)
    mlflow.log_artifact(temp_name, "deepsvdd" )
  finally:
    temp.close()
  
  print(f"MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# Create a folder locally to hold artifacts. Remember that our model has 2 important pieces
folder = 'artifacts'
if os.path.isdir(folder):
  shutil.rmtree(folder)
  os.makedirs(folder)
  pass
else:
  os.makedirs(folder)

# COMMAND ----------

# MAGIC %md Reconstruct Model for Batch Scoring (single node - distributed scoring later in the notebook)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.download_artifacts(runID, "deepsvdd","artifacts/")
deepSVDD = joblib.load(f"artifacts/deepsvdd/{temp_name.split('/')[-1]}")
logged_model = f'runs:/{runID}/model'
deepSVDD.model_ =  mlflow.keras.load_model(logged_model)
deepSVDD.model_.save('artifacts/keras/model.h5')

# COMMAND ----------

ls artifacts

# COMMAND ----------

# MAGIC %md Run a quick check to see if we yield the same results - check against trained model

# COMMAND ----------

# Sanity Check To See if the model produces the same results
from rich.console import Console
score_array = X_test[0].reshape(1,-1)

def test(test_array:np.array):
  try:
    assert round(deepSVDD.decision_function(test_array)[0],4) == round(y_test_scores[0],4)
    console = Console()
    console.log(locals())
    console.log("Test Passed -- Model Import Runs Ok! ✅")
  except:
    console.log("Test Failed")

# COMMAND ----------

test(score_array)

# COMMAND ----------

# MAGIC %md Now that the model is ready we can safely apply it 

# COMMAND ----------

deepSVDD.predict(X_test)[:50] #runs on single machine - has to be dropped into a UDF/python function for parallel scoring

# COMMAND ----------

# MAGIC %md Package Model for Online Serving - Need a custom pyfunc model to ensure we can properly reconstruct the model

# COMMAND ----------

# capture the env
import platform
import tensorflow
python_version = platform.python_version()
import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      f'python={python_version}',
      'pip',
      {
        'pip': [
          'mlflow',
          'pyod',
          f'tensorflow_cpu =={tensorflow.__version__}',
          f'cloudpickle=={cloudpickle.__version__}',
          f'joblib=={joblib.__version__}',
        ],
      },
    ],
    'name': 'deepsvdd_env'
}

# COMMAND ----------

#link to artifacts - can use paths we saved them on previously
artifacts = {
    "deepSVDD_model": f"artifacts/deepsvdd/{temp_name.split('/')[-1]}",
    "keras_model": "artifacts/keras/model.h5"
}

# COMMAND ----------

# define a custom class
class PyodDeepSVDDAFWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from pyod.models.deep_svdd import DeepSVDD
    import mlflow
    import joblib
    import glob
    import tensorflow as tf
    
    self.deepSVDD_model = joblib.load(glob.glob(context.artifacts["deepSVDD_model"])[0])
    self.deepSVDD_model.model_ =  tf.keras.models.load_model(context.artifacts["keras_model"])
    
  def predict(self,context, model_input):
    logged_model = f'runs:/{runID}/model'
    model_input = model_input.reshape(1,-1)
    return self.deepSVDD_model.predict(model_input)

# COMMAND ----------

# save the model locally
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
mlflow_pyfunc_model_path="deepSVDDmodel_pyfunc_"+timestr
print(mlflow_pyfunc_model_path)
mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,python_model=PyodDeepSVDDAFWrapper(),artifacts=artifacts,
        conda_env=conda_env)

# COMMAND ----------

ls $mlflow_pyfunc_model_path/artifacts/

# COMMAND ----------

# infer model signature
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
loaded_model.predict(score_array)
from mlflow.models.signature import infer_signature
signature = infer_signature(score_array, loaded_model.predict(score_array))
signature

# COMMAND ----------

# log for serving
mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=PyodDeepSVDDAFWrapper(),artifacts=artifacts,
        conda_env=conda_env, signature = signature)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md To generate a sample for serving

# COMMAND ----------

#get a serving example
import json
json.dumps({"data":score_array.tolist()})

# COMMAND ----------

# MAGIC %md Batch scoring with Pandas UDF - Use Spark Cluster to Score the Model

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf
from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

experiment_id = client.get_experiment_by_name("/Users/sathish.gangichetty@databricks.com/anomaly_detection_deepsvdd_dbr").experiment_id
# get most recent run by sorting 
most_recent_run = sorted([{"run_id":i.run_id, "end_time":i.end_time} 
                        for i in client.list_run_infos(experiment_id=experiment_id)
                        if i.status=='FINISHED'],
                       key=lambda x: x['end_time'], 
                       reverse=True)[0]
most_recent_run

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
  

# COMMAND ----------

### This is a check to see if scoring against a dataframe works as expected
# logged_model = f'runs:/{most_recent_run["run_id"]}/model'
# client.download_artifacts(most_recent_run['run_id'], "deepsvdd","")
# deepSVDD = joblib.load(glob.glob("deepsvdd/*.joblib")[0])
# deepSVDD.model_ =  mlflow.keras.load_model(logged_model)
# deepSVDD.predict(test_anom_df[train_cols].to_pandas())
# Load model as a Spark UDF. Override result_type if the model does not return double values.
# loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# COMMAND ----------

df = train_df.to_spark()
preds_df = df.withColumn("prediction",udf_predict(*df.columns))
display(preds_df)