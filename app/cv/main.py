# main.py

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from vector_store import add_document, search_similar
from ats_score import ats_score
from embedding import Embedder
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from app.chat_layer.utils import GROQ_API_KEY2
import os
import io
import PyPDF2  # pip install PyPDF2
# OR use: import fitz  # pip install pymupdf  (أفضل بكتير)

app = FastAPI()

# Load embedder
embedder = Embedder()

# =========================
# Groq API
# =========================


client = Groq(api_key=GROQ_API_KEY2)


# =========================
# PDF Parser
# =========================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    بتاخد الـ PDF كـ bytes وبترجع النص منه
    """
    try:
        # الطريقة 1: PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise ValueError("PDF فاضي أو مش قابل للقراءة (scanned image)")

        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"مشكلة في قراءة الـ PDF: {str(e)}")


# =========================
# AI Analysis
# =========================

def llm_feedback(cv_text, jd_text, ats):

    prompt = f"""
You are a professional ATS resume analyzer.

Candidate CV:
{cv_text}

Job Description:
{jd_text}

ATS Score: {ats}%

Give a clear analysis:

1- What is good in the CV

2- Mistakes in the CV

3- Missing skills

4- How to improve the CV

5- Suggest better words to replace weak words

Write in simple clear sentences.
No bullet symbols.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1024  # زيادة من 350 لتحليل أوفى
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Groq Error: {e}"


# =========================
# API Endpoints
# =========================

@app.post("/add_cv")
def add_cv(id: str = Form(...), text: str = Form(...)):
    add_document(id, text, "cv")
    return {"status": "CV added"}


@app.post("/match")
async def match(
    cv_file: UploadFile = File(...),       
    jd_text: str = Form(...)               
):
    # 1. التحقق إن الملف PDF
    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="الملف لازم يكون PDF")

    # 2. قراءة الـ PDF واستخراج النص
    file_bytes = await cv_file.read()
    cv_text = extract_text_from_pdf(file_bytes)

    # 3. ATS Score
    ats, matched, missing = ats_score(cv_text, jd_text)

    # 4. AI Analysis
    analysis = llm_feedback(cv_text, jd_text, ats)

    return {
        "ats_score": ats,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "analysis": analysis
    }


@app.get("/")
def home():
    return {
        "message": "ATS Analyzer with PDF Support is running!"
    }