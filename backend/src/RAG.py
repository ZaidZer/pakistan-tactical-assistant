# backend/src/RAG.py
import os
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- Load environment variables early ---
load_dotenv()

# --- Initialize OpenAI client with key ---
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Constants ---
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
TOP_K = 5  # number of chunks to retrieve per team


# ---------- Embedding ----------
async def embed_query(query: str):
    """Embed the coach's question asynchronously."""
    resp = await client.embeddings.create(model=EMBED_MODEL, input=query)
    return np.array(resp.data[0].embedding, dtype=np.float32)


# ---------- Retrieval ----------
def load_index(team_name: str):
    """Load FAISS index for a given team."""
    index_path = Path(__file__).resolve().parents[1] / f"data/indexes/{team_name}_faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found for {team_name}")
    return faiss.read_index(str(index_path))


def retrieve_chunks(index, query_vec, k=TOP_K):
    """Retrieve top vector indices and distances."""
    distances, indices = index.search(np.array([query_vec]), k)
    return indices[0], distances[0]


# ---------- Reasoning ----------
async def generate_advice(pakistan_ctx, myanmar_ctx, question):
    """Use the LLM to generate tactical reasoning."""
    prompt = f"""
You are an elite football tactical analyst.
Use the provided match data and visuals to give coach-level advice.

Pakistan (data + visuals):
{pakistan_ctx}

Myanmar (data + visuals):
{myanmar_ctx}

Question:
{question}

Provide:
1. Summary of Pakistan’s style and patterns.
2. Tactical weaknesses of the opponent (if available).
3. Recommended strategies, shape, and press adjustments.
4. Practical formation suggestions.
    """
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return resp.choices[0].message.content


# ---------- Main orchestrator ----------
async def run_tactical_query(question: str):
    """Orchestrates retrieval + reasoning for tactical queries."""
    print(f"\n[+] Embedding query: {question}")
    q_vec = await embed_query(question)

    # Try loading indexes
    pk_index = None
    my_index = None
    try:
        pk_index = load_index("pakistan_strat")
    except FileNotFoundError:
        print("[!] Pakistan index not found.")
    try:
        my_index = load_index("myanmar_strat")
    except FileNotFoundError:
        print("[!] Myanmar index not found — skipping opponent context.")

    if pk_index is None and my_index is None:
        print("[❌] No indexes available. Please run embeddings first.")
        return "No tactical data available."

    pakistan_ctx = ""
    myanmar_ctx = ""

    if pk_index:
        pk_idx, pk_dist = retrieve_chunks(pk_index, q_vec)
        pakistan_ctx = f"Pakistan context vector IDs: {pk_idx.tolist()}"
    if my_index:
        my_idx, my_dist = retrieve_chunks(my_index, q_vec)
        myanmar_ctx = f"Myanmar context vector IDs: {my_idx.tolist()}"

    print(f"[INFO] Retrieved available context.")
    print("[+] Generating tactical advice via GPT-4o...")

    answer = await generate_advice(pakistan_ctx, myanmar_ctx, question)

    print("\n=================  Tactical Report  =================\n")
    print(answer)
    print("\n=====================================================\n")

    return answer


# ✅ Local run support
if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_tactical_query(
            "Analyze Pakistan’s current formation and suggest tactical adjustments to counter Myanmar’s wing play."
        )
    )
