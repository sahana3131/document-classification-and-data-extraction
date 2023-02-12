import os
import PyPDF2
import docx
import pytesseract
from PIL import Image

def classify_and_split(file_path):
    # check the file type
    ext = os.path.splitext(file_path)[1]

    if ext == '.pdf':
        # open the pdf file
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfFileReader(file)
            # iterate over the pages
            for page in range(pdf.getNumPages()):
                text = pdf.getPage(page).extractText()
                classify_and_split_text(text)
    elif ext == '.docx':
        # open the word document
        doc = docx.Document(file_path)
        # extract the text
        text = '\n'.join([para.text for para in doc.paragraphs])
        classify_and_split_text(text)
    elif ext in ['.jpg', '.jpeg', '.png']:
        # open the image
        image = Image.open(file_path)
        # extract the text using OCR
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"

        text = pytesseract.image_to_string(image)
        classify_and_split_text(text)
    else:
        print("Invalid file type")

def classify_and_split_text(text):
    # classify and split the text based on document type
    if "PAN" in text:
        print("PAN document found")
    if "Aadhaar" in text:
        print("Aadhaar document found")
    if "Bank Statement" in text:
        print("Bank statement found")
    if "ITR" in text or "Form 16" in text:
        print("ITR/Form 16 document found")
    if "Customer Photograph" in text or "Selfie" in text:
        print("Customer photograph found")
    if "Utility Bill" in text or "Power" in text or "Water" in text or "Gas" in text or "Landline" in text:
        print("Utility bill found")
    if "Cheque Leaf" in text:
        print("Cheque leaf found")
    if "Salary Slip" in text or "Certificate" in text:
        print("Salary slip/certificate found")
    if "Driving License" in text:
        print("Driving license found")
    if "Voter ID" in text:
        print("Voter ID found")
    if "Passport" in text:
        print("Passport found")

classify_and_split("C:/Users/SAHANA/Desktop/SAHANA-EDU/AMRITA-CSE/extra/ihack/inputs/OIP.jpg")
