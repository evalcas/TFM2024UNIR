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
vector_store.get()

retriever = vector_store.as_retriever()


qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query = 'cuales son las funciones de un departamento academico'


qs = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever = vector_store.as_retriever(), return_source_documents=True)




time_start = time()
#result = qa.invoke(query)
result = qs.invoke({"question": query}, return_only_outputs=True)
time_end = time()
total_time = f"{round(time_end-time_start, 3)} sec."

full_response =  f"Question: {query}\nAnswer: {result}\nTotal time: {total_time}"
#print(full_response)
#print(colorize_text(full_response))

#answer = response['answer']
answer = result['answer']
sources = result['sources'].split(', ')
source_documents = result['source_documents']
print('\nanswer:'+ answer)
print('\nsources:') 
print(sources)
print('\nsource_documents:')

for doc in source_documents:
    doc_details = doc.to_json()['kwargs']
    print("\nsource: ", doc_details['metadata']['source'])
    print("\nPage:", doc_details['metadata']['page'])
    print("\nPage_content: ", doc_details['page_content'], "\n")

 


"""
docs = vector_store.similarity_search(query)
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print ("Page:", doc_details['metadata']['page'])
    print("Page_content: ", doc_details['page_content'], "\n")


[Document(page_content='55 \n \n 131 pido.', metadata={'page': 55, 'source': 'pdfs/Estatuto UNC 2014.pdf'})]}
"""