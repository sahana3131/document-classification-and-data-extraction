import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Accept user-supplied file
file_path = input("Enter the file path:")

# Use OCR to extract text from the document
text = pytesseract.image_to_string(Image.open(file_path))

# Use NLTK to tokenize and tag the text
tokens = word_tokenize(text)
tagged_tokens = pos_tag(tokens)

# Initialize empty lists to store the different document types
pan_list = []
aadhaar_list = []
bank_statement_list = []
itr_form_16_list = []
customer_photograph_list = []
utility_bill_list = []
cheque_leaf_list = []
salary_slip_certificate_list = []
driving_license_list = []
voter_id_list = []
passport_list = []

# Iterate through the tagged tokens and classify the document
for token in tagged_tokens:
    if token[1] == "NNP" and token[0] == "PAN":
        pan_list.append(token)
    elif token[1] == "NNP" and token[0] == "Aadhaar":
        aadhaar_list.append(token)
    elif token[1] == "NNP" and token[0] == "Bank":
        bank_statement_list.append(token)
    elif token[1] == "NNP" and token[0] == "ITR":
        itr_form_16_list.append(token)
    elif token[1] == "NNP" and token[0] == "Customer":
        customer_photograph_list.append(token)
    elif token[1] == "NNP" and token[0] == "Utility":
        utility_bill_list.append(token)
    elif token[1] == "NNP" and token[0] == "Cheque":
        cheque_leaf_list.append(token)
    elif token[1] == "NNP" and token[0] == "Salary":
        salary_slip_certificate_list.append(token)
    elif token[1] == "NNP" and token[0] == "Driving":
        driving_license_list.append(token)
    elif token[1] == "NNP" and token[0] == "Voter":
        voter_id_list.append(token)
    elif token[1] == "NNP" and token[0] == "Passport":
        passport_list.append(token)

# Print the different document types
print("PAN: ", pan_list)
print("Aadhaar: ", aadhaar_list)
print("Bank Statement: ", bank_statement_list)
print("ITR/Form 16: ", itr_form_16_)
