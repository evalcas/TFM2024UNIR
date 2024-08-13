import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import json
from langchain_community.document_loaders import PyPDFLoader

from g_drive_service import GoogleDriveService
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime

app = Flask(__name__)

# Configuraciones iniciales
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ruta donde se almacenan los pdf procesados o para procesar
directory = "pdfs/"
# ruta donde se almacenan los .pdf vectorizados
folder_path = "chromadb"
# Nombre del vectorstore generado por chromadb
collection = "chromadb_unc"

#Definicion del embedding que se utilizara para vectorizar
#para uso con Ollama 
cached_llm = Ollama(model = "mistral")
#embeddings = FastEmbedEmbeddings()

embeddings = OpenAIEmbeddings(openai_api_key =  OPENAI_API_KEY)

# alternativa a generar chunks, no se utiliza pero se deja referencia que puede utilizarse, se hizo la prueba y no hay impacto en el rendimiento comparado con load_and_split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Configuracion del prompt inicial
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# APIS
# Gestion y vectorizacion de documentos en formato pdf

# ID de la carpeta compartida en Google Drive
folder_id_GQATFM2024 = '1k1HIb8EhkdvETldjY0HRZ7dXLZudZDC_'
folder_id_pdfs = '19kdDqtegzYj0st4QUIj77ScJ0N9KW3-O'
folder_id_chromadb = '1yNmQornrrqJFQT5Y39mev2sAuF0yDabo'

#####probando google drive
###list files
@app.get('/gdrive/list-files')
def getFileListFromGDrive():
    selected_fields="files(id,name,webViewLink, parents)"
    g_drive_service=GoogleDriveService().build()
    list_file=g_drive_service.files().list(fields=selected_fields).execute()
    return {"files":list_file.get("files")}

### upload file
@app.post('/gdrive/upload-file')
def uploadFileToGDrive():
    g_drive_service=GoogleDriveService().build()
    uploaded_file=request.files.get("file")

    buffer_memory=BytesIO()
    uploaded_file.save(buffer_memory)

    media_body=MediaIoBaseUpload(uploaded_file, uploaded_file.mimetype, resumable=True)

    created_at= datetime.now().strftime("%Y%m%d%H%M%S")
    file_metadata={
        "name":f"{uploaded_file.filename}",# ({created_at})",
         "parents": [folder_id_pdfs]  # directorio deseado para almacenar pdfs
    }

    returned_fields="id, name, mimeType, webViewLink, exportLinks"
    
    upload_response=g_drive_service.files().create(
        body = file_metadata, 
        media_body=media_body,  
        fields=returned_fields
    ).execute()

    return upload_response

### delete file
@app.delete('/gdrive/<id>/')
def deleteFileToGDrive(id):
    g_drive_service=GoogleDriveService().build()
    g_drive_service.files().delete(fileId=id).execute()
    
    return {'message': 'Successfully deleted file'}


## Validar si un archivo ya existe en el drive
@app.post('/gdrive/verify-file')
def verifyFileInGDrive():
    uploaded_file=request.files.get("file")
    g_drive_service=GoogleDriveService().build()

    buffer_memory=BytesIO()
    uploaded_file.save(buffer_memory)
    file_name = uploaded_file.filename
    print(file_name)

    # Realiza una consulta para buscar el archivo por su nombre en la carpeta especificada
    query = f"name='{file_name}' and '{folder_id_pdfs}' in parents and trashed=false"
    response = g_drive_service.files().list(q=query, fields='files(id, name, parents)').execute()

    # Si se encontraron archivos con el mismo nombre en la carpeta, el archivo existe
    if 'files' in response:
        for file in response['files']:
            # Verifica si el archivo tiene un padre, lo que indica que está en la carpeta
            if 'parents' in file:
                # Si el archivo tiene un padre, también existe en la carpeta
                return {'message': 'El archivo ya existe'}
    return {'message': 'El archivo no se encuentra en el drive'}


def getFolderIdFromGDrive(folder_name):
    g_drive_service=GoogleDriveService().build()
    # Busca el ID de la carpeta por su nombre
    folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folder_response = g_drive_service.files().list(q=folder_query, fields='files(id)').execute()
    # Si se encontró la carpeta, obtiene su ID
    if 'files' in folder_response and folder_response['files']:
        folder_id = folder_response['files'][0]['id']
        return folder_id


def getFileIdFromGDrive(file_name):
    g_drive_service=GoogleDriveService().build()
    # Busca el ID de la carpeta por su nombre
    file_query = f"name='{file_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    file_response = g_drive_service.files().list(q=file_query, fields='files(id)').execute()



 
####################################

# Subir un archivo a un directorio, especificamente un archivo pdf para vectorizar
@app.route("/api/vectorizar/uploadpdf",methods = ["POST"])
def upload_file():
    """
    Este endpoint recibe un archivo pdf a través de una solicitud POST, valida el pdf y vectoriza con chormadb.
    Args:
        file: Un archivo con extension .pdf enviado en el body de la solicitud.
    Returns:
        - message (response): Un mensaje de éxito o error.      
    """
    # Validando el archivo .pdf
    file = request.files["file"] #uploaded_file

    if not file or file.filename =='':
        return jsonify({'error': 'No hay archivo seleccionado'}), 400
    if os.path.exists(directory + file.filename):
        return jsonify({'error': 'Un archivo con nombre:' + file.filename + ' ya ha sido procesado',
                        'status': 'No se vectorizo el archivo'}), 400
    
    # Validar si los chunks ya existen en el vectorstore (Falta analizar si se hace)

    # Subiendo el archivo al repositorio de pdfs
    file_name = file.filename
    save_file = directory + file_name
    file.save(save_file)  
    print(f"filename: {file_name}")

    # Extrayendo el texto del .pdf y generando los chunks
    loader = PyPDFLoader(save_file)
    docs = loader.load_and_split() #creando chunks
    
    # Vectorizando y haciendo persistente los chunks en la coleccion (chromadb)
    vectordb = Chroma.from_documents(documents=docs,embedding= embeddings,persist_directory = folder_path, collection_name=collection)
    vectordb.persist
    vectordb = None

    # Generando la respuesta de vectorizacion exitosa
    response ={
        "filename": file_name,
        "status": "Pdf Succesfully Vectorized in chromadb"
        }
    return response , 200

# Endpoint para obtener los archivos del directorio o para crear el vectorstore de los pdf existentes en el directorio establecido
@app.route("/api/vectorizar/directorio",methods = ['GET', 'POST'])
def vectorStorePost():
    files = os.listdir(directory)
    if request.method == 'POST':
        # vectorizando cada archivo pdf
        for file in files:
            loader = PyPDFLoader(directory + file)
            docs = loader.load_and_split()
            vectordb = Chroma.from_documents(documents=docs,embedding= embeddings,persist_directory = folder_path, collection_name=collection)
        vectordb.persist
        vectordb = None

        response ={
            "status": "Vectorstore Succesfully created using chromadb", 
            }
        return response ,200
    
    elif request.method == 'GET':
        # Crear una lista con los nombres de archivo existentes en el directorio pdfs
        list_files= os.listdir(directory)
        json_files = json.dumps(list_files, indent=4)
        response = {
            'status': 'Lista de archivos con extension pdf obtenidas con exito',
            'files': json.loads(json_files)}
        return response, 200

# Enpoint generico, envia un query generico y  retorna una respuesta del llm
@app.route("/api/prompt/iagen", methods = ["POST"])
def iagenPost():
    print("Llamada al post aigen")
    json_content = request.json
    query =  json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"Answer: ": response}
    return response_answer ,200

# Endpoint que permite enviar un query y retorna un response con la respuesta GQA
@app.route("/api/prompt/askpdf", methods = ["POST"])
def askPDFPost():
    print("Post ask_pdf called")
    json_content = request.json
    query =  json_content.get("query")
    
    print(f"query: {query}")
    
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings, collection_name=collection)
    vector_store.get()

    print("Creating chain")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    #qs = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(), k=1, chain_type = "stuff", vectorstore=vectordb)
    qs = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever = vector_store.as_retriever(), return_source_documents=True)

    result = qs.invoke({"question": query}, return_only_outputs=True)
    
    #dando formato en json al response
    answer = result['answer']
    sources = result['sources'].split(', ')
    source_documents = result['source_documents']
    document_details=[]
    for doc in source_documents:
        doc_details = doc.to_json()['kwargs']
        document_details.append({
            "Source": doc_details['metadata']['source'],
            "Page": doc_details['metadata']['page'],
            "Page_content": doc_details['page_content']
        })
        
    response_data = json.dumps(document_details, indent=4)

    response_answer = {
       'question': query, 
       'answer': answer,
       'sources': sources,
       'source_documents' : json.loads(response_data)
    }

    return response_answer,200

def start_app():
    app.run(host="0.0.0.0",port=8084, debug=True)


if(__name__ == "__main__"):
    start_app()