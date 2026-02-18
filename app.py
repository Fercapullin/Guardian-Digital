from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


# Inicializando la aplicación Flask
app = Flask(__name__)

# Cargando las variables de entorno
load_dotenv()

# Cargando las variables de entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Descargando el modelo de embeddings
embeddings = download_embeddings()

index_name = "guardian-digital" 
# Creando la vector store en Pinecone a partir del índice existente y los embeddings
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Creando el recuperador a partir de la vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
# Creando el modelo de lenguaje para la generación de respuestas
chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# Creando la cadena de preguntas y respuestas a partir del modelo de lenguaje y el prompt
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Definiendo la ruta para la página principal
@app.route("/")
def index():
    return render_template('chat.html')

# Definiendo la ruta para manejar las solicitudes de chat
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



#
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)