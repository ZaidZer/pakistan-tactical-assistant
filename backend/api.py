# backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.src.RAG import run_tactical_query
import logging
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize app
app = FastAPI(title="Pakistan Tactical Assistant API", version="1.0")

# Configure CORS (secure in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Health check (for Render)
@app.get("/")
async def root():
    return {"status": "ok", "message": "API running successfully"}

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Core endpoint
@app.post("/analyze")
async def analyze(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = await run_tactical_query(question)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error while processing query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
