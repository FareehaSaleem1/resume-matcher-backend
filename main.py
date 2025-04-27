from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from fastapi.responses import JSONResponse
import traceback
from nltk.tokenize import sent_tokenize  


app = FastAPI()

origins = [
    "http://localhost:3000",  # (development on local)
    "https://resume-matcher-frontend-six.vercel.app"  # (your Vercel deployed frontend)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Skills reference lists
TECH_SKILLS = [
    'python', 'java', 'c++', 'sql', 'html', 'css', 'javascript', 'react',
    'node', 'django', 'flask', 'tensorflow', 'pytorch', 'scikit-learn'
]
SOFT_SKILLS = [
    'communication', 'teamwork', 'leadership', 'problem-solving',
    'adaptability', 'creativity', 'time management', 'critical thinking'
]

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_skills(text: str):
    tokens = word_tokenize(text)
    words = set([word.lower() for word in tokens if word.isalpha()])
    tech = [skill for skill in TECH_SKILLS if skill in words]
    soft = [skill for skill in SOFT_SKILLS if skill in words]
    return {"technical": tech, "soft": soft}
# ✨ Explanation generator
def generate_explanation(skills, score, job_desc):
    reasons = []

    if score > 80:
        reasons.append("strong alignment with the job requirements")
    elif score > 60:
        reasons.append("moderate alignment with relevant skills")
    else:
        reasons.append("some relevant experience")

    if skills["technical"]:
        reasons.append("technical skills like " + ", ".join(skills["technical"][:3]))

    if skills["soft"]:
        reasons.append("soft skills such as " + ", ".join(skills["soft"][:2]))

    return f"This candidate matched due to {', and '.join(reasons)}."


@app.post("/match")
async def match_resumes(resumes: List[UploadFile] = File(...), job_description: str = Form(...)):
    try:
        job_emb = model.encode(job_description, convert_to_tensor=True)
        results = []

        for resume in resumes:
            text = extract_text_from_pdf(resume)
            res_emb = model.encode(text, convert_to_tensor=True)
            score = float(util.cos_sim(job_emb, res_emb)[0][0])
            skills = extract_skills(text)
            summary_sentences = sent_tokenize(text)[:3]
            summary = " ".join(summary_sentences)
            why_matched = generate_explanation(skills, score, job_description)

            results.append({
                "filename": resume.filename,
                "score": round(score * 100, 2),
                "skills": skills,
                "summary": summary,
                "why_matched": why_matched

            })

        return {"matches": results}
    
    except Exception as e:
        print("⚠️ Backend crashed during matching:")
        traceback.print_exc()  # 🔍 shows exact error
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "Resume Matcher API is live."}
