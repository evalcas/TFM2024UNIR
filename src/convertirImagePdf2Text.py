import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import fitz
import pdfplumber

# Ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Ruta al archivo de datos de entrenamiento de Tesseract
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# Se ha modificado el codigo, funciona mejor la extraccion de texto, sin embargo, falla la extraccion en algunas tablas

def preprocess_image(img):
# Convertir la imagen a escala de grises
    img = img.convert('L')
    
    # Aplicar un filtro de desenfoque para reducir el ruido
    img = img.filter(ImageFilter.GaussianBlur(1))
    
    # Aumentar el contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    # Aumentar el brillo
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)
    
    # Binarizar la imagen
    threshold = 128
    img = img.point(lambda p: p > threshold and 255)
    
    return img

def extract_text_and_tables_from_pdf(file_path): # falta probar
    text_output = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extraer texto y tablas usando pdfplumber
            text_output.append(page.extract_text())
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    text_output.append(' | '.join(row))
            
            # Si la página contiene imágenes, extraer y procesar las imágenes
            im_list = page.images
            for im in im_list:
                img_bbox = (im['x0'], im['top'], im['x1'], im['bottom'])
                im = page.within_bbox(img_bbox).to_image()
                img = Image.fromarray(im.original)
                
                # Preprocesar la imagen
                img = preprocess_image(img)
                
                # Extraer texto de la imagen usando Tesseract con idioma español
                text = pytesseract.image_to_string(img, lang='spa', config=f'--psm 6 {tessdata_dir_config}')
                text_output.append(text)
    return '\n'.join(text_output)

def extract_text_from_pdf(file_path):

    pdf_document = fitz.open(file_path)
    extracted_text = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Aumentar la DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Mostrar la imagen extraída
        # img.show()
        # Preprocesar la imagen
        img = preprocess_image(img)
        # Mostrar la imagen preprocesada
        # img.show()
        # Extraer texto de la imagen usando Tesseract con idioma español
        text = pytesseract.image_to_string(img, lang='spa', config=f'--psm 6 {tessdata_dir_config}')
        extracted_text.append(text)
    
    pdf_document.close()
    return extracted_text

def create_pdf_with_text(text_list, output_pdf_path):
    # Crear un PDF con el texto extraído
    pdf_document = fitz.open()
    for text in text_list:
        page = pdf_document.new_page()
        page.insert_text((72, 72), text)  # Insertar texto en la página
    
    pdf_document.save(output_pdf_path)
    pdf_document.close()

def convert_pdf_images_to_text_pdf(input_pdf_path, output_pdf_path):
    # Extraer texto del PDF de imágenes
    text = extract_text_from_pdf(input_pdf_path)
    # Crear un nuevo PDF con el texto extraído
    create_pdf_with_text(text, output_pdf_path)

# Path to the input PDF with images
input_pdf_path = 'input_with_images.pdf'
# Path to the output PDF with extracted text
output_pdf_path = 'output_with_text.pdf'

convert_pdf_images_to_text_pdf(input_pdf_path, output_pdf_path)
print(f"Converted PDF saved to {output_pdf_path}")
