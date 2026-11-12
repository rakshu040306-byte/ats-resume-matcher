import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from supabase import create_client, Client
from app.resume_analyzer import analyze_resume

# Safely connect to Supabase if environment variables are set
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Smart Resume & ATS Job Matcher")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/analyze")
async def score_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    pdf_bytes = await resume.read()
    result = analyze_resume(pdf_bytes, job_description)

    if "error" in result:
        return result

    if supabase:
        try:
            supabase.table("ats_scans").insert({
                "filename": resume.filename,
                "match_score": result["match_score"],
                "missing_keywords": ", ".join(result["missing_keywords"][:5])
            }).execute()
        except Exception as e:
            print("Supabase log skipped:", e)

    return result
