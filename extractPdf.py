import fitz

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# pdf_text = extract_text_from_pdf("AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees.pdf")
# print(pdf_text[100:500])