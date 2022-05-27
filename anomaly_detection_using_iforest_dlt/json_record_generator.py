# Databricks notebook source
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Read in the credit card data into a spark 
# MAGIC dataframe

# COMMAND ----------

file_location = "dbfs:/FileStore/tables/creditcard.csv"
file_type = 'csv'

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# COMMAND ----------

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location).toPandas()



# COMMAND ----------

# MAGIC %md
# MAGIC Rename id column to cust_id and drop amount and class columns so this is identical to what the model was trained on

# COMMAND ----------

df.reset_index(inplace=True)
df.rename(columns={'index':'cust_id'}, inplace=True)
df.drop(columns=df.columns[-1], 
        axis=1, 
        inplace=True)


display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC For demo purposes clear the landing location by removing all files

# COMMAND ----------

#Do a streaming read then a streaming write 
json_landing = "/FileStore/tables/transaction_landing_dir"
dbutils.fs.mkdirs(json_landing)

# COMMAND ----------

# MAGIC %md
# MAGIC Add json records row by row into the landing location specified

# COMMAND ----------

#Parametrize this at the notebook level with a widget so it ...

# COMMAND ----------

#Add timestamp for the time in which this was generated, but for the sake of calling out for it 

# COMMAND ----------

import time 
import random 

i = 0
for json_dict in df.to_dict(orient='records'):
  dbutils.fs.put("{}/row{}.json".format(json_landing,i), str(json_dict))
  i += 1 
  #time.sleep(random.random())


# COMMAND ----------

# MAGIC %md
# MAGIC %fs
# MAGIC rm -r "/FileStore/tables/sensor_json_landing"
