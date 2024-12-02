# Databricks notebook source
# MAGIC %md
# MAGIC #Review App
# MAGIC
# MAGIC Criando um app para os stakeholder poderem avaliar a solução RAG criada nos passos anteriores.

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain langchain_core langchain_community databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./_setup/setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando a Chain

# COMMAND ----------

import mlflow
import yaml
import os

LLM_MODEL_NAME = instruct_model
VECTOR_SEARCH_ENDPOINT_NAME = vs_endpoint
VECTOR_SEARCH_INDEX_NAME = f"{catalogo}.{schema}.{vs_indice}"

# COMMAND ----------

rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": LLM_MODEL_NAME,
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
    },
    "input_example": {
        "messages": [{"content": "Posso declarar minha sogra como dependente?", "role": "user"}]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "Você é um assistente confiável da Receita Federal que ajuda a responder perguntas sobre declaração de imposto de renda com base apenas nas informações fornecidas. Se a pergunta não estiver relacionada a um desses tópicos, recuse-se a responder. Se você não sabe a resposta para uma pergunta, você diz sinceramente que não sabe. Aqui está um contexto que pode ou não ajudá-lo a responder: {context}. Responda diretamente, não repita a pergunta, não comece com algo como: a resposta à pergunta, não adicione IA na frente da sua resposta, não diga: aqui está a resposta, não mencione o contexto ou a pergunta. Com base neste contexto, responda a esta pergunta: {question}",

        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 1, "query_type": "ann"},
        "schema": {"chunk_text": "qa_combined", "document_uri": "file_path", "primary_key": "id"},
        "vector_search_index": VECTOR_SEARCH_INDEX_NAME
    },
}
try:
    with open('rag_chain_config.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC import os
# MAGIC import mlflow
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC #Get the conf from the local conf file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC # Connect to the Vector Search Index
# MAGIC vs_client = VectorSearchClient(disable_notice=True)
# MAGIC vs_index = vs_client.get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC )
# MAGIC vector_search_schema = retriever_config.get("schema")
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     columns=[
# MAGIC         vector_search_schema.get("id"),
# MAGIC         vector_search_schema.get("response"),
# MAGIC         vector_search_schema.get("prompt"),
# MAGIC     ],
# MAGIC ).as_retriever(search_kwargs=retriever_config.get("parameters"))
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the RAG Studio Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=vector_search_schema.get("primary_key"),
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     doc_uri=vector_search_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = PromptTemplate(
# MAGIC     template=llm_config.get("llm_prompt_template"),
# MAGIC     input_variables=llm_config.get("llm_prompt_template_variables"),
# MAGIC )
# MAGIC
# MAGIC from langchain_core.messages.ai import AIMessageChunk
# MAGIC
# MAGIC class chatDatabricksCustom(ChatDatabricks):
# MAGIC   def stream(self, input, config, **kwargs):
# MAGIC     for message in super().stream(input, config = config, **kwargs):
# MAGIC         transformed_output = self.transform_latin_1(message)
# MAGIC         yield transformed_output
# MAGIC
# MAGIC   def transform_latin_1(self, message):
# MAGIC     return AIMessageChunk(content=message.content.encode('latin1').decode('utf-8'))
# MAGIC
# MAGIC # FM for generation
# MAGIC model = chatDatabricksCustom(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # model = ChatDatabricks(
# MAGIC #     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC #     extra_params=llm_config.get("llm_parameters"),
# MAGIC # )
# MAGIC
# MAGIC
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)
# MAGIC
# MAGIC # COMMAND ----------
# MAGIC
# MAGIC # chain.invoke(model_config.get("input_example"))

# COMMAND ----------

print(f"Criando o modelo {model_name} como um experimento do MLFLow.")
uc_model =  f"{catalogo}.{schema}.{model_name}"

# Log the model to MLflow
with mlflow.start_run(run_name=f"{uc_model}"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='rag_chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
    )


# COMMAND ----------

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Implementando nosso aplicativo RAG e liberando para usuários externos

# COMMAND ----------

from databricks import agents

MODEL_NAME_FQN = uc_model

# COMMAND ----------

# agents.deployments.__DEPLOY_ENV_VARS_WITH_STATIC_VALUES = {
#     "ENABLE_LANGCHAIN_STREAMING": "false",
#     "ENABLE_MLFLOW_TRACING": "true",
#     "RETURN_REQUEST_ID_IN_RESPONSE": "true",
# }

# COMMAND ----------

instructions_to_reviewer = f"""### Instruções para testar nosso assistente de chatbot de declaração de Imposto de Renda (IR)

Suas contribuições são inestimáveis ​​para a equipe de desenvolvimento. Ao fornecer comentários e correções detalhadas, você nos ajuda a corrigir problemas e melhorar a qualidade geral do aplicativo. Contamos com sua experiência para identificar quaisquer lacunas ou áreas que precisem de melhorias.

1. **Variedade de perguntas**:
 - Experimente uma ampla variedade de perguntas que você prevê que os usuários finais do aplicativo farão. Isso nos ajuda a garantir que o aplicativo possa lidar com as consultas esperadas de maneira eficaz.

2. **Feedback sobre as respostas**:
 - Depois de fazer cada pergunta, use os widgets de feedback fornecidos para revisar a resposta dada pelo aplicativo.
 - Se você acha que a resposta está incorreta ou pode ser melhorada, use "Editar resposta" para corrigi-la. Suas correções permitirão que nossa equipe refine a precisão do aplicativo.

3. **Revisão de Documentos Devolvidos**:
 - Revise cuidadosamente cada documento que o sistema retorna em resposta à sua pergunta.
 - Use o recurso polegar para cima/para baixo para indicar se o documento era relevante para a pergunta feita. Um polegar para cima significa relevância, enquanto um polegar para baixo indica que o documento não foi útil.

Obrigado pelo seu tempo e esforço em testar nosso assistente. Suas contribuições são essenciais para entregar um produto de alta qualidade aos nossos usuários finais."""

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True)

#agents.enable_trace_reviews(model_name=MODEL_NAME_FQN) 

# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Concedendo às partes interessadas acesso ao aplicativo de avaliação de agente Mosaic AI
# MAGIC
# MAGIC Liberando permissões às partes interessadas para usar o aplicativo Review. Para simplificar o acesso, as partes interessadas não necessitam ter contas Databricks.

# COMMAND ----------

user_list = ["ana.sanchez@databricks.com"]


# Set the permissions.
agents.set_permissions(model_name=MODEL_NAME_FQN, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Compartilhe esta URL com os Stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encontrando o nome do App de Review
# MAGIC Se você perder os outputs deste notebook e precisar encontrar a URL do seu aplicativo de avaliação, poderá listar o chatbot implantado:

# COMMAND ----------

active_deployments = agents.list_deployments()
active_deployment = next((item for item in active_deployments if item.model_name == MODEL_NAME_FQN), None)
if active_deployment:
  print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adiciona as perguntas dos usuários para review

# COMMAND ----------

#agents.enable_trace_reviews(model_name=MODEL_NAME_FQN)

# COMMAND ----------


