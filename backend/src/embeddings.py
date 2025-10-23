import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found. Make sure your .env file is in the project root.")


# --- CONFIG ---
CHUNK_SIZE = 1500  # words per chunk
OVERLAP = 200      # overlap between chunks for context


def chunk_text(text):
    """Split long text into smaller overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks


def create_embedding(text):
    """Generate an embedding using OpenAI's text-embedding-3-large model."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def build_faiss_index(team_name: str):
    """Create a FAISS index for all processed text and image captions of a team."""
    processed_dir = Path(f"data/processed/{team_name}")
    index_dir = Path("data/indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{team_name}_faiss.index"

    all_embeddings = []
    all_metadata = []

    # --- Embed text chunks ---
    for txt_file in processed_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text)
        print(f"[+] Chunking {txt_file.name} into {len(chunks)} parts...")

        for chunk in chunks:
            emb = create_embedding(chunk)
            all_embeddings.append(emb)
            all_metadata.append(txt_file.name)

    # --- Embed image captions ---
    caption_dir = Path(f"data/processed/{team_name}/image_captions")
    if caption_dir.exists():
        print(f"[INFO] Found image captions for {team_name}, embedding...")
        for caption_file in caption_dir.glob("*.txt"):
            with open(caption_file, "r", encoding="utf-8") as f:
                caption_text = f.read().strip()
            if caption_text:
                emb = create_embedding(caption_text)
                all_embeddings.append(emb)
                all_metadata.append(caption_file.name)
                print(f"[+] Embedded caption: {caption_file.name}")

    # --- Save index ---
    if not all_embeddings:
        print(f"[!] No embeddings created for {team_name}")
        return

    embeddings = np.vstack(all_embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    print(f"[✅] Saved FAISS index: {index_path}")
    print(f"    Total vectors: {len(all_embeddings)}")


if __name__ == "__main__":
    build_faiss_index("pakistan_strat")
    build_faiss_index("myanmar_strat")

