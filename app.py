# website/app.py
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
from pathlib import Path

# --- CONFIG ---
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
TOP_K = 5

client = OpenAI()

# --- Utility Functions ---
def embed_query(query: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def load_index(team_name: str):
    index_path = Path(f"data/indexes/{team_name}_faiss.index")
    if not index_path.exists():
        st.warning(f"⚠️ FAISS index not found for {team_name}")
        return None
    return faiss.read_index(str(index_path))


def retrieve_chunks(index, query_vec, k=TOP_K):
    distances, indices = index.search(np.array([query_vec]), k)
    return indices[0], distances[0]


def generate_advice(pakistan_ctx, myanmar_ctx, question):
    prompt = f"""
You are an elite football tactical analyst.
Use Pakistan’s match data (and Myanmar’s if available) to give detailed coaching advice.

Pakistan context:
{pakistan_ctx}

Myanmar context:
{myanmar_ctx}

Question:
{question}

Provide:
1. Tactical tendencies for each team
2. Key vulnerabilities
3. A recommended game plan for Pakistan
4. Suggested formations or role adjustments
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return resp.choices[0].message.content


# --- Streamlit UI ---
st.set_page_config(page_title="Pakistan Tactical Analyst", page_icon="⚽", layout="centered")
st.title("⚽ Pakistan Tactical Analysis Assistant")
st.write("Ask tactical questions about Pakistan’s matches and strategies. The model will reason from match reports and visuals.")

question = st.text_area("💬 Ask a tactical question:", placeholder="e.g. How can Pakistan break Myanmar’s wide press?")

if st.button("Analyze"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Analyzing match data and generating tactical advice..."):
            q_vec = embed_query(question)

            pk_index = load_index("pakistan_strat")
            my_index = load_index("myanmar_strat")

            pk_ctx = ""
            my_ctx = ""

            if pk_index:
                pk_idx, pk_dist = retrieve_chunks(pk_index, q_vec)
                pk_ctx = f"Pakistan context vector IDs: {pk_idx.tolist()}"
            if my_index:
                my_idx, my_dist = retrieve_chunks(my_index, q_vec)
                my_ctx = f"Myanmar context vector IDs: {my_idx.tolist()}"

            if pk_ctx or my_ctx:
                answer = generate_advice(pk_ctx, my_ctx, question)
                st.success("✅ Tactical Report Generated")
                st.markdown("### 🧠 Tactical Report")
                st.write(answer)
            else:
                st.error("No FAISS indexes found. Please embed match data first.")
