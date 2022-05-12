This demo uses the kaggle CC dataset to show it can be applied for anomaly detection. We demonstrated DeepSVDD for anomaly detection using the databricks ML stack. The specific components used are

    -> The DBR ML runtime 
    -> DBR managed MLFlow
    -> Databricks ML Serving
    -> Optimized Pyspark Pandas UDFs for batch scoring.

The model deployed is based on a custom pyfunc support on MLFlow. 

Databricks DLT pipeline based deployment is WIP.

