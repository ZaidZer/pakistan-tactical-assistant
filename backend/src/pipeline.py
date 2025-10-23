import glob
from src.pdf_extraction import extract_text_from_pdf
from src.image_extraction import extract_images_from_pdf
from src.embeddings import build_faiss_index
from src.RAG import run_tactical_query
import asyncio


def process_team(team_name: str, folder_path: str):
    """Extract text & images from all PDFs for a given team."""
    print(f"\n[INFO] Processing team: {team_name}")
    all_pdfs = glob.glob(f"{folder_path}/*.pdf")

    full_text = ""
    for pdf_path in all_pdfs:
        print(f"  → Extracting from: {pdf_path}")
        text = extract_text_from_pdf(pdf_path, team_name)
        extract_images_from_pdf(pdf_path, team_name)
        full_text += "\n" + text

    print(f"[INFO] Generating embeddings for {team_name}...")
    build_faiss_index(team_name)
    print(f"[✅] {team_name} FAISS index updated.\n")



if __name__ == "__main__":
    process_team("pakistan_strat", "data/raw/pakistan_strat")
    process_team("myanmar_strat", "data/raw/myanmar_strat")

    print("[INFO] Running strategy comparison test query...")
    response = asyncio.run(run_tactical_query("Compare Pakistan vs Myanmar and suggest tactical improvements."))
    print(response)
    print("\n=================  Tactical Analysis  =================\n")
    print(response)
    print("\n========================================================\n")

