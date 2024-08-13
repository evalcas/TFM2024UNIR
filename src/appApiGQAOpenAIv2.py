#Librerias genericas
import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request
import json
import hashlib
import fitz
from io import BytesIO

#Librerias langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader

# Librerias personalizadas
from customExceptionHandler import CustomExceptionHandler
from responseCreator import ResponseCreator


app = Flask(__name__)

# Configuraciones iniciales
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ruta donde se almacenan los pdf procesados o para procesar
directory = "../pdfs/"
# ruta donde se almacenan los .pdf vectorizados
folder_path = "../chromadb"
# Nombre del vectorstore generado por chromadb
collection = "chromadb_unc"

#Ajusta la temperatura del modelo relacionada con la alucinacion
temperature = 0 

#Definicion del embedding que se utilizara para vectorizar
embeddings = OpenAIEmbeddings(openai_api_key =  OPENAI_API_KEY)

# alternativa a generar chunks, no se utiliza pero se deja referencia que puede utilizarse, se hizo la prueba y no hay impacto en el rendimiento comparado con load_and_split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Configuracion del prompt inicial
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] 
    You are a technical assistant good at searching docuemnts. Your responses should adhere to the following guidelines:
    - Answer the question based on the provided documents.
    - Be direct and factual, limited to 2-4 sentences. Begin your response without using introductory phrases like yes, no etc.
    - Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
    - If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
    - Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
    - Do not fabricate information or include questions in your responses.
    - do not prompt to select answers. do not ask me questions
    - Respond in the language in which the question was asked
    - If you do not have an answer from the provided information say so. 
    [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Definicion de funciones utilitarias
# Funcion que valida si una query es valida
def is_valid_query(query):
    if not query or not isinstance(query, str) or not query.strip():
        return False
    return True

# Funcion que calcula el valor hash de un archivo
def calculate_file_hash(file):
    hasher = hashlib.sha256()
    file.seek(0)  # Asegura para estar al principio del archivo
    while chunk := file.read(8192):
        hasher.update(chunk)
    file.seek(0)  # Resetea el puntero del archivo para no alterar su estado
    return hasher.hexdigest()

# Funcion que valida si un archivo existe en el directorio donde se almancena los pdfs
def file_exists_in_directory_by_content(file_to_check):
    # Calcula el valor hash del archivo a verificar
    file_to_check_hash = calculate_file_hash(file_to_check)
    # Itera sobre los archivos en el directorio y compara sus hashes
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as current_file:
                current_file_hash = calculate_file_hash(current_file)
                if current_file_hash == file_to_check_hash:
                    return True  # Devuelve True si se encuentra una coincidencia
    return False  # Devuelve False si no encuentra coincidencias

# Funcion que valida si el archivo pdf contiene texto
def pdf_contains_text(file):
     # Leer el archivo PDF en memoria
    pdf_bytes = file.read()
    # Convertir los bytes a un objeto de BytesIO
    pdf_stream = BytesIO(pdf_bytes)
    # Abrir el PDF utilizando PyMuPDF
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    # Extraer texto de cada página del PDF
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        text = page.get_text()
        # Si alguna página contiene texto, retornar True
        if text.strip():
            return True
    # Si ninguna página contiene texto, retornar False
    return False


# APIS
# Gestion y vectorizacion de documentos en formato pdf
# Subir un archivo a un directorio, especificamente un archivo pdf para vectorizar
# En el body del request es necesario file = 'archivo a vectorizar.pdf'
@app.route("/api/v1/vectorizar/uploadpdf",methods = ["POST"])
def upload_file():
    """
    Este endpoint recibe un archivo pdf a través de una solicitud POST, valida el pdf y vectoriza con chormadb.
    Args:
        file: Un archivo con extension .pdf enviado en el body de la solicitud.
    Returns:
        - message (response): Un mensaje de éxito o error.      
    """
    # Validacion del archivo .pdf
    file = request.files['file']
    try:
        
        # Validar si en la peticion hay un archivo pdf        
        if not file or file.filename =='':
            return ResponseCreator.create_response('error', 'No hay archivo en el body de la peticion'), 400
        # Validar que el archivo tenga la extension pdf        
        if not file.filename.endswith('.pdf') or file.mimetype != 'application/pdf':
            return ResponseCreator.create_response('error','El archivo no tiene formato PDF'), 400
        # Validar si el archivo ya existe pero con otro nombre        
        if file_exists_in_directory_by_content(file):
            return ResponseCreator.create_response('error','Existe un archivo con contenido similar que ya ha sido procesado'),400
        # Validar si el archivo ya ha sido vectorizado        
        if os.path.exists(directory + file.filename):
            return ResponseCreator.create_response(
                'error', f'Un archivo con nombre: {file.filename}  ya ha sido procesado',
                {'aditional_message': 'No se vectorizo el archivo'}
             ), 400
        # Validar si el archivo contiene texto        
        if not pdf_contains_text(file):
            return ResponseCreator.create_response(
                'error', f'El archivo: {file1.filename}  no contiene texto, posiblemente tenga imagenes',
                {'aditional_message': 'No se vectorizo el archivo'}
            ), 400
        file.seek(0)
    except Exception as e:
        # Si ocurre una excepción, el manejador global la capturará
        return CustomExceptionHandler.handle_global_exception(e)
    
    
    # Codigo para subir el archivo al repositorio de pdfs
    file_name = file.filename
    #save_file = directory + file_name
    save_file = os.path.join(directory, file_name)
    file.save(save_file)  

    print(f"filename: {save_file}")
        
    # Codigo para obtener la fecha de modificación del archivo
    modification_time = os.path.getmtime(save_file)
    formatted_time = datetime.fromtimestamp(modification_time).strftime("%Y-%m-%d %H:%M:%S")

    # Extrayendo el texto del .pdf y generando los chunks
    loader = PyPDFLoader(save_file)
    docs = loader.load_and_split() #creando chunks
   
    # Vectorizando y haciendo persistente los chunks en la coleccion (chromadb)
    vectordb = Chroma.from_documents(documents=docs,embedding= embeddings,persist_directory = folder_path, collection_name=collection)
    vectordb.persist
    vectordb = None

    # Generando la respuesta de vectorizacion exitosa
    response = ResponseCreator.create_response(
        'success',
        'Pdf Succesfully Vectorized in chromadb',
        {'filename': file_name,
         'date_creation': formatted_time,
         'Collection_name': collection,
         'directory': folder_path
         }
    )
    return response , 200


# Eliminar archivo del vectorstore
# En el body del request es necesario file_name = 'archivo a eliminar del vectorstore'
@app.route("/api/v1/vectorizar/deletevector",methods = ["DELETE"])
def delete_file():
    #Vaidaciones del nombre del archivo
    # Verificar el encabezado Content-Type
    if request.content_type != 'application/json':
        response = ResponseCreator.create_response(
                'error',
                'No se intentó cargar los datos JSON porque el Content-Type de la solicitud no era application/json'
            )
        return response , 400
    
    data = request.get_json()
    #Verificar que existan datos en el body
    if not data:
        response = ResponseCreator.create_response(
                'error',
                'No se encontraron datos json en el body'
            )
        return response , 400 

    file_name=data.get('file_name')
    
    # Verificar que file_name no sea None ni una cadena vacía
    if not file_name or not isinstance(file_name, str):
        response = ResponseCreator.create_response(
                'error',
                'El nombre del archivo esta vacio o es no valido'
            )
        return response , 400
    
    # Verificar que file_name tenga una extensión válida
    # Validar que el archivo tenga extensión .pdf o no tenga extensión
    if not file_name.endswith('.pdf'):
        response = ResponseCreator.create_response(
                'error',
                'Extensión de archivo no válida'
            )
        return response , 404 
    
    #Validaciones de que el archivo exista en el directorio
    # Construye la ruta completa del archivo
    file_path = os.path.join(directory, file_name)
    # Verifica si el archivo existe
    if not os.path.exists(file_path):
        response = ResponseCreator.create_response(
                'error',
                'El archivo no existe en el directorio, aun no ha sido vectorizado'
            )
        return response , 404 
      
    #Validacion de que el archivo este en el vectorstore   
    # buscar el nombre del archivo en el vectorstore
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings, collection_name=collection)
    data = vector_store.get()
    # Filtrar los metadatos para encontrar los documentos con el mismo 'source' que contiene el nombre del archivo
    matching_ids = [id for id, meta in zip(data['ids'], data['metadatas']) if meta['source'].endswith(file_name)]
    #Iteración Simultánea: Se usa zip(metadata['ids'], metadata['metadatas']) para iterar simultáneamente sobre los IDs y los metadatos asociados
    #Eliminamos el file_name del vectorstore
    vector_store.delete(ids=matching_ids)
    #Eliminamos el file_name del directorio
    os.remove(file_path)
    
    #print(file_path)
    #print(f"IDs de documentos que coinciden con '{file_name}': {matching_ids}")

    response =ResponseCreator.create_response(
            'success',
            'File Succesfully deleted from  vectorstore',
            {'filename': file_name,
             'directory': folder_path}
        )
    
    return response ,200


# Endpoint para obtener los archivos del directorio o para crear el vectorstore de los pdf existentes en el directorio establecido
@app.route("/api/v1/vectorizar/directorio",methods = ['GET', 'POST'])
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

        response =ResponseCreator.create_response(
            'success',
            'Vectorstore Succesfully created using chromadb',
            {'Collection_name': collection,
             'directory': folder_path}
        )
        return response ,200
    
    elif request.method == 'GET':
        # Crear una lista con los nombres de archivo existentes en el directorio pdfs
        # Obtener la lista de archivos en el directorio
        file_names = os.listdir(directory)
        # Verificar si hay archivos en el directorio
        if not file_names:
            # Si no hay archivos, crear una respuesta indicando que no hay archivos
            response = ResponseCreator.create_response(
                'success',
                'No hay archivos con extensión PDF en el directorio'
            )
        else:
        # Código para obtener la lista de archivos y sus fechas de creación
            files_with_dates = []
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                files_with_dates.append({'file_name': file_name, 'creation_time': creation_time})

            response = ResponseCreator.create_response(
                'success',
                'Lista de archivos con extension pdf obtenidas con exito',
                {'files': files_with_dates}
            )
        return response, 200


# Endpoint que permite enviar un query y retorna un response con la respuesta GQA
@app.route("/api/v1/prompt/askpdf", methods = ["POST"])
def askPDFPost():
    #print("Post ask_pdf called")
    # Se captura la peticion desde el body
    try:
        # Verificar el Content-Type de la solicitud
        if request.content_type is None:
            response = ResponseCreator.create_response(
                            'error',
                            'No hay contenido en el body.'
                        )
            return response, 415
        if request.content_type != 'application/json':
            response = ResponseCreator.create_response(
                            'error',
                            'El Content-Type del body debe ser application/json.'
                        )
            return response, 415
            
        json_content = request.json
        # Validar que el JSON no sea nulo
        if not json_content:
            response = ResponseCreator.create_response(
                'error',
                'No se encontró datos JSON.'
            )
            return response, 400        
        # Acceder al campo 'query' de manera segura
        query =  json_content.get("query")

        # Validar que 'query' tenga texto y no esté en blanco
        if not query or not isinstance(query, str):
            response = ResponseCreator.create_response(
                            'error',
                            'El campo query es invalido, no existe o esta en blanco.'
                        )
            return response, 400   
    except Exception as e:
        # Si ocurre una excepción, el manejador global la capturará
        return CustomExceptionHandler.handle_global_exception(e)
  
    # Obteniendo los embeddings de los documentos del vectorstore
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings, collection_name=collection)
    vector_store.get()

    # Configurando el modelo LLM y la temperatura
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
    # Crear una instancia de RetrievalQAWithSourcesChain con los parámetros especificados como modelo y vectorstore
    qs = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever = vector_store.as_retriever(), return_source_documents=True)
    
    # Realizando la consulta utilizando la consulta y los embbedings
    result = qs.invoke({"question": query}, return_only_outputs=True)
    
    #dando formato en json al response considerando la estructura de qs
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