# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook para realizar a configuração do ambiente

# COMMAND ----------

##### Preencha com o nome do catálogo
catalogo = "workshops_databricks"

##### Preencha com o nome do prefixo do schema
prefix_db = "db_rag_ir_"

##### Preencha com o nome do indice do Vector Search
vs_endpoint = "one-env-shared-endpoint-13"
vs_indice = "vs_ir_rag"

##### Preencher com o nome do endpoint de LLM que irá usar
instruct_model= "databricks-dbrx-instruct"
embedding_model= 'databricks-bge-large-en'

model_name="ir_chatbot_model"

