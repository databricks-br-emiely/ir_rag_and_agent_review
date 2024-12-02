# Databricks notebook source
# MAGIC %run "./env"

# COMMAND ----------

def clean_string(value, replacement: str = "_") -> str:
    import re
    replacement_2x = replacement+replacement
    value = re.sub(r"[^a-zA-Z\d]", replacement, str(value))
    while replacement_2x in value:
        value = value.replace(replacement_2x, replacement)
    return value

# COMMAND ----------

def get_username():
  row = spark.sql("SELECT current_user() as username, current_catalog() as catalog, current_database() as schema").first()
  return row["username"]

# COMMAND ----------

def get_workspace_id():
  return dbutils.entry_point.getDbutils().notebook().getContext().workspaceId().getOrElse(None)

# COMMAND ----------

from typing import Any
def stable_hash(*args: Any, length: int) -> str:
    import hashlib
    args = [str(a) for a in args]
    data = ":".join(args).encode("utf-8")
    value = int(hashlib.md5(data).hexdigest(), 16)
    numerals = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = []
    for i in range(length):
        result += numerals[value % 36]
        value //= 36
    return "".join(result)

# COMMAND ----------

def unique_name():
  local_part = get_username().split("@")[0]
  hash_basis = f"{get_username()}{get_workspace_id()}"
  username_hash = stable_hash(hash_basis, length=4)
  name = f"{local_part} {username_hash}"
  return clean_string(name).lower()

# COMMAND ----------

schema = prefix_db + unique_name()
spark.sql(f"create database if not exists {catalogo}.{schema}")

# COMMAND ----------

serving_endpoint_name = serving_endpoint_name_prefix + unique_name()

# COMMAND ----------

spark.sql(f"use {catalogo}.{schema}")
volume_folder = f"/Volumes/{catalogo}/{schema}/landing"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalogo}.{schema}.landing")

contratos_folder = f"{volume_folder}/contratos"

# Crie a nova pasta dentro do volume
dbutils.fs.mkdirs(contratos_folder)

import requests 

def download_file(url, destination):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        print('saving ' + destination + '/' + local_filename)
        with open(destination + '/' + local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename
  

