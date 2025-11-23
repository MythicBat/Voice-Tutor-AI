import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env for local dev
load_dotenv()

# Create Vertex AI Gemini client (config via env vars)
client = genai.Client()

GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

# --- FastAPI setup ---
app = FastAPI(
    title="VoiceTutor AI Backend",
    description="STEM tutoring brain powered by Gemini on Vertex AI",
    version="0.1.0",
)

# CORS so our frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response models
class TutorRequest(BaseModel):
    message: str
    grade_level: Optional[str] = None
    subject: Optional[str] = None
    language: Optional[str] = "en"

class TutorResponse(BaseModel):
    reply: str
    # later add steps, summary, quizz, etc

SYSTEM_INSTRUCTION = """
You are VoiceTutor AI, a patient and friendly STEM teacher for students.

Goals:
- Explain STEM concepts (maths, physics, chemistry, computer science) very clearly.
- Use simple language first, then gently add more details.
- Always encourage the student and avoid making them feel dumb.
- Prefer step-by-step reasoning with short steps.
- Check for understanding and suggest a small follow-up question or practice problem.

Output style:
- Use short paragraphs and bullet points.
- Use LaTeX-style math (like x^2, sqrt(3), or fractions as 1/2) but keep it readable in plain text.
- Avoid overloading the student; if the concept is big, break it into mini-concepts.
"""

# Routes
@app.get("/health")
def health_check():
    return {"status": "ok", "model": GEMINI_MODEL_ID}

@app.post("/api/tutor", response_model=TutorResponse)
async def tutor(request: TutorRequest):
    """
    Main endpoint: takes a student's question and returns a friendly STEM explanation.
    """
    level = request.grade_level or "unspecified"
    subject = request.subject or "general STEM"

    user_prompt = f"""
Student level: {level}
Subject: {subject}
Language: {request.language}

The student says:
\"\"\"{request.message}\"\"\"

Your job:
1. Briefly restate the question in your own words
2. Give an intuitive explanation.
3. Show a simple worked example if relevant.
4. Finish by asking the student a small check question to confirm understanding.
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_output_tokens=512,
        ),
    )

    answer = response.text

    return TutorResponse(reply=answer)