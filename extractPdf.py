import fitz
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# pdf_text = extract_text_from_pdf("AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees.pdf")
# print(pdf_text[0:500])

def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# sentence = split_sentences(pdf_text)
# print("-----",sentence[:5])