import streamlit as st
import tempfile
import os
import pandas as pd

from extractPdf import extract_text, split_sentences
from local import classify_with_ollama


st.set_page_config(page_title="AI Authorship Detector", layout="wide")
st.title("📑 AI Authorship Detector (Ollama + Gemma 3)")

st.markdown("Detect whether your text (sentence-by-sentence) is **AI-generated** or **Human-written**.")

# Sidebar options
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["gemma3:1b", "gemma3n:latest"],
    index=0
)

input_mode = st.radio("Choose Input Mode", ["📋 Paste Text", "📂 Upload File"])

text = ""
if input_mode == "📋 Paste Text":
    text = st.text_area("Paste your text here", height=250)

elif input_mode == "📂 Upload File":
    uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            text = extract_text(tmp.name)

if text:
    if st.button("🔍 Run Detection"):
        sentences = split_sentences(text)
        results = []
        st.write(f"Found **{len(sentences)} sentences**. Running detection with `{model_choice}` ...")

        progress = st.progress(0)
        for i, s in enumerate(sentences, start=1):
            verdict = classify_with_ollama(s, model=model_choice)
            results.append({
                "Sentence": s,
                "Label": verdict["label"],
                "Confidence": verdict["confidence"]
            })
            progress.progress(i / len(sentences))

        st.success("Detection complete ✅")

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results (CSV)", csv, "results.csv", "text/csv")