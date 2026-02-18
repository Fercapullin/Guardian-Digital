from dotenv import load_dotenv
import os 
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Cargando las variables de entorno
load_dotenv()

# Configurando las claves de API para Pinecone y OpenAI
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurando las variables de entorno para las claves de API
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Extrayendo el texto de los archivos PDF
extracted_data = load_pdf_files("data/")
# Filtrando los documentos para mantener solo el contenido y la fuente
filter_data = filter_to_minimal_docs(extracted_data)
# Dividiendo el texto en fragmentos manejables
text_chunks = text_split(filter_data)

# Descargando el modelo de embeddings
embedding = download_embeddings()

# Configurando Pinecone
pinecode_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecode_api_key)


# Creando el índice en Pinecone si no existe
index_name = "guardian-digital"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimensión de los vectores de embedding
        metric="cosine",  # Métrica de similitud
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Creando la vector store en Pinecone a partir de los documentos y embeddings
dosearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name,
)