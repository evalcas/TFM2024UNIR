from langchain_community.vectorstores import Chroma
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from time import time
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ruta al directorio donde está almacenado ChromaDB
chromadb_directory = 'chromadb'


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

#embeddings = FastEmbedEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key =  OPENAI_API_KEY)

vector_store = Chroma(persist_directory='chromadb', embedding_function=embeddings, collection_name='chromadb_unc')
metadata = vector_store.get()
documentos = metadata['documents']
metas = metadata['metadatas']

#file_name = 'pdfduplicado.pdf'
file_name = 'statuto UNC 2014.pdf'

# Filtrar los metadatos para encontrar los documentos con el mismo 'source'
#matching_ids = [meta['id'] for meta in metadata['metadatas'] if meta['source'].endswith(file_name)]

# Filtrar los metadatos para encontrar los documentos con el mismo 'source'
matching_ids = [id for id, meta in zip(metadata['ids'], metadata['metadatas']) if meta['source'].endswith(file_name)]

# Imprimir los IDs correspondientes
print(f"IDs de documentos que coinciden con '{file_name}': {matching_ids}")
vector_store.delete(ids=matching_ids)



#source_path = metas[0]['source']
#file_name = os.path.basename(source_path)

#print(metadata['ids'])
#print(documentos[0])
#print(file_name)

#print (metadata['documents'][0])
#print (documentos[0])
