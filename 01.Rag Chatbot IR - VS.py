# Databricks notebook source
# MAGIC %md
# MAGIC ## Instala Bibliotecas

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.2 langchain==0.1.6 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks] langchain-community==0.0.19 PyPDF2
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run "./_setup/setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download do PDF no volume

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("volume_path",f"/Volumes/{catalogo}/{schema}/landing","Caminho para o Volume")
dbutils.widgets.text("catalog",catalogo,"Catálogo")
dbutils.widgets.text("schema", schema ,"Schema")
dbutils.widgets.text("vs_endpoint",vs_endpoint,"Nome do endpoint de Vector Search")
dbutils.widgets.text("index_name",vs_indice,"Nome do Índice do Vector Search")

# COMMAND ----------

volume_path=dbutils.widgets.get("volume_path")
catalog=dbutils.widgets.get("catalog")
sch=dbutils.widgets.get("schema")
index_name=dbutils.widgets.get("index_name")
vs_endpoint=dbutils.widgets.get("vs_endpoint")

# COMMAND ----------

https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/pr-irpf-2024.pdf
"https://raw.githubusercontent.com/anasanchezss9/data-for-demos/refs/heads/main/P&R IRPF 2023 - v1.1 - 04042023.pdf"

# COMMAND ----------

# DBTITLE 1,Inicializando Variáveis cadastradas em _setup/env
url = "https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/pr-irpf-2024.pdf"
volume_folder = volume_path  + "/ir"
dbutils.fs.mkdirs(volume_folder)

try:
    download_file(url, volume_folder)
    print(f"Arquivo de contrato baixado com sucesso em {volume_folder}!")
except Exception as err:
    print(f"An unexpected error occurred: {err}")

# COMMAND ----------

files = dbutils.fs.ls(volume_folder)
arquivo = files[0].name if files else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cria a base de conhecimento

# COMMAND ----------

import re
from PyPDF2 import PdfReader
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import monotonically_increasing_id

def read_pdf(pdf_path):
  reader = PdfReader(pdf_path)
  all_text = ""

  for page in reader.pages:
    all_text += page.extract_text()
    
  return all_text.replace("\n", "")

pdf_path = f"{volume_folder}/{arquivo}"

pdf_text = read_pdf(pdf_path)
regex_pattern = r"\d{3} [—-](.*?)Retorno ao sumário"

lista_perguntas = re.findall(regex_pattern,pdf_text)

# lista_perguntas_respostas = [x.strip().split('?', 1) for x in lista_perguntas]
# lista_perguntas_respostas_qm = [(p+"?",r) for p,r in lista_perguntas_respostas]

# schema = StructType([
#     StructField("prompt", StringType(), True),
#     StructField("response", StringType(), True)
# ])

# Combine question and answer into a single string and wrap each in a tuple
lista_perguntas_respostas_combined = [(x.strip(),) for x in lista_perguntas]

# Define schema with a single column for both question and answer
sch = StructType([
    StructField("qa_combined", StringType(), True)
])


df = spark.createDataFrame(lista_perguntas_respostas_combined, sch)
df = df.withColumn('id', monotonically_increasing_id()).select('id', *df.columns)
display(df)

# COMMAND ----------

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalogo}.{schema}.ir_pdf_documentation_id")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `${catalog}`.`${schema}`.`ir_pdf_documentation_id` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Crair o chat LLM sem a base de conhecimento

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatDatabricks

def chat(message):
    messages = [
        SystemMessage(content="Você é um assistente da Receita Federal do Brasil que responde somente no idioma português."),
        HumanMessage(content=message)
        ]

    chat_model = ChatDatabricks(endpoint=instruct_model)
    return (chat_model.invoke(messages).content)

# COMMAND ----------

chat("quais são as regras para declarar minha sogra como dependente?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar o chat com a base de conhecimento (RAG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criando o Vector Search
# MAGIC Antes de executar os próximos passos, vamos criar um index através da interface do Unity.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **Acesse a Tabela Delta**:
# MAGIC    Primeiro, acesse a tabela criada nos passos anteriores:

# COMMAND ----------

displayHTML(f"""
Acesse a tabela criada nos passos anteriores: <a href="/explore/data/{catalog}/{sch}/ir_pdf_documentation_id">{catalog}.{sch}.ir_pdf_documentation_id</a>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Crie o Índice de Busca Vetorial**:
# MAGIC    Clique no botão **Create** no canto superior direito e selecione **Vector search index** no menu suspenso.
# MAGIC
# MAGIC 3. **Configure o Índice**:
# MAGIC    Use os seletores no diálogo para configurar o índice:
# MAGIC    - **Name**: Nome para a tabela online no Unity Catalog. O nome requer um namespace de três níveis, `<catalog>.<schema>.<name>`. Apenas caracteres alfanuméricos e underscores são permitidos.
# MAGIC    - **Primary key**: Coluna a ser usada como chave primária.
# MAGIC    - **Endpoint**: Selecione o endpoint de busca vetorial que você deseja usar.
# MAGIC    - **Columns to sync**: Selecione as colunas a serem sincronizadas com o índice vetorial. Se este campo for deixado em branco, todas as colunas da tabela de origem serão sincronizadas com o índice. A coluna de chave primária e a coluna de origem de embeddings ou coluna de vetor de embeddings são sempre sincronizadas.
# MAGIC    - **Embedding source**: Indique se você deseja que o Databricks compute embeddings para uma coluna de texto na tabela Delta (Compute embeddings), ou se sua tabela Delta contém embeddings pré-computados (Use existing embedding column).
# MAGIC
# MAGIC 4. **Configurações de Embeddings**:
# MAGIC    - Se você selecionou **Compute embeddings**, selecione a coluna para a qual deseja que os embeddings sejam computados e o endpoint que está servindo o modelo de embeddings. Apenas colunas de texto são suportadas.
# MAGIC    - Se você selecionou **Use existing embedding column**, selecione a coluna que contém os embeddings pré-computados e a dimensão dos embeddings. O formato da coluna de embeddings pré-computados deve ser `array[float]`.
# MAGIC
# MAGIC 5. **Sincronização de Embeddings Computados**:
# MAGIC    - Ative esta configuração para salvar os embeddings gerados em uma tabela do Unity Catalog. Para mais informações, veja [Save generated embedding table](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search/).
# MAGIC
# MAGIC 6. **Modo de Sincronização**:
# MAGIC    - **Continuous**: Mantém o índice sincronizado com segundos de latência. No entanto, tem um custo mais alto associado, pois um cluster de computação é provisionado para executar o pipeline de sincronização contínua.
# MAGIC    - **Triggered**: É mais econômico, mas deve ser iniciado manualmente usando a API. Para ambos os modos, a atualização é incremental — apenas os dados que mudaram desde a última sincronização são processados.
# MAGIC
# MAGIC 7. **Criação do Índice**:
# MAGIC    Quando terminar de configurar o índice, clique em **Create**.
# MAGIC
# MAGIC Quando tiver terminado a crição, siga os próximos passos:

# COMMAND ----------

# MAGIC %md
# MAGIC Testando o vector search criado:

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding = DatabricksEmbeddings(endpoint=embedding_model)

index = f"{catalog}.{sch}.{index_name}"

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint,
        index_name=index
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch( 
        vs_index, text_column="qa_combined", embedding=embedding
    )
    return vectorstore.as_retriever(search_kwargs={'k': 1})

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("eu posso declarar minha sogra como dependente?")

print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """Você é um assistente da Receita Federal. Você está respondendo perguntas gerais sobre declaração de imposto de renda. Se a pergunta não estiver relacionada a um desses tópicos, recuse-se a responder. Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta. Não fale para ele consultar perguntas. Mantenha a resposta o mais concisa possível.
Use o seguinte trecho para responder à pergunta no final:
{context}
Pergunta: {question}
Resposta:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])
chat_model = ChatDatabricks(endpoint=instruct_model)

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

import langchain

langchain.debug = False
question = {"query": "posso declarar minha sogra como dependente?", "columns" : ["response"]}
answer = chain.invoke(question)
print(answer['result'])
