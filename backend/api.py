from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.src.RAG import run_tactical_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 Define the expected request body
class QuestionRequest(BaseModel):
    question: str

@app.post("/analyze")
async def analyze(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Question cannot be empty"}

    try:
        answer = run_tactical_query(question)
        return {"answer": answer}
    except Exception as e:
        import traceback
        print("\n[SERVER ERROR]", traceback.format_exc())
        return {"error": f"Server exception: {e}"}
