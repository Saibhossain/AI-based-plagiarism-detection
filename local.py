import subprocess
import json
import tempfile
import re, os
SYSTEM_INSTRUCTION = (
    "You are an AI-authorship detector. "
    "Classify the input sentence as either AI-generated or Human-written. "
    "Be conservative: if uncertain, prefer 'Human'. "
    "Return STRICT JSON ONLY with keys: label (AI|Human), confidence (0..1)."
)

PROMPT_TEMPLATE = (
    "Task: Determine if the following sentence is AI-generated or Human-written.\n\n"
    "Sentence:\n\"\"\"\n{sentence}\n\"\"\"\n\n"
    "Respond JSON only:\n"
    "{\n  \"label\": \"AI\" | \"Human\",\n  \"confidence\": float between 0 and 1\n}"
)

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
    m = JSON_PATTERN.search(s)
    return m.group(0) if m else ""

def classify_with_ollama(sentence: str, model: str = "gemma3n:e4b") -> Dict:
    prompt = f"{SYSTEM_INSTRUCTION}\n\n" + PROMPT_TEMPLATE.format(sentence=sentence)
    cmd = ["ollama", "run", model]

    try:
        proc = subprocess.run(
            cmd, input=prompt.encode("utf-8"),
            capture_output=True, timeout=60
        )
        raw_out = proc.stdout.decode("utf-8", errors="ignore").strip()
        js = _extract_json(raw_out)
        if js:
            obj = json.loads(js)
            return {
                "label": obj.get("label", "Human"),
                "confidence": float(obj.get("confidence", 0.5)),
            }
        return {"label": "Human", "confidence": 0.5}
    except Exception:
        return {"label": "Human", "confidence": 0.5}