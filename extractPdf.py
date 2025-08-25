import fitz
import docx
import spacy
from typing import List, Dict


nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path_lower.endswith(".docx"):
        return extract_text_from_docx(path)
    elif path_lower.endswith(".txt"):
        return extract_text_from_txt(path)
    else:
        raise ValueError("Unsupported file type. Use PDF/DOCX/TXT.")


# pdf_text = extract_text_from_pdf("AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees.pdf")
# print(pdf_text[0:500])

def split_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 1]

# sentence = split_sentences(pdf_text)
# print("-----",sentence[:5])