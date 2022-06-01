This demo uses the [kaggle CC dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to show it can be applied for anomaly detection. The specific components used are

    -> The DBR ML runtime 
    -> DBR managed MLFlow
    -> Databricks ML Serving
    -> Optimized Pyspark Pandas UDFs for batch scoring.

There are a couple of examples here in this repo (as of June 1 2022). If you are just getting started with anomaly detection, start from `anomaly_detection_using_iforest_dlt` folder. In there you will find the following assets.

1. `iforest_train_register.py` - quickly trains an isolation forest model and registers it to the model registry. Tweak it as you see fit for your use.
2. `json_record_generator.py` - shows a way to generate dummy json files should you choose to use the [demo data from kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. `dlt_task_0_iforest_dlt_udf.py` - contains code that wrap the model into a pandas udf for scoring inside DLT
4. `dlt_task_1_anomaly_detector.sql` - shows a simplified end to end DLT workflow that reads the json files created in step 2 and scores that with the trained anomaly detection model registered as a udf on step 3.

To read more about DLT and help getting started with DLT - visit the following [link](https://docs.microsoft.com/en-us/azure/databricks/data-engineering/delta-live-tables/delta-live-tables-quickstart)

The model deployed is based on a custom pyfunc support on MLFlow. 

Databricks DLT pipeline based deployment is WIP.

