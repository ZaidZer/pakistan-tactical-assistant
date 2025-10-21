# src/pdf_extraction.py
import fitz  # PyMuPDF
import os
from pathlib import Path

def extract_text_from_pdf(pdf_path: str, team_name: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")

    # Save processed text
    output_dir = Path(f"data/processed/{team_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(pdf_path).stem}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[+] Extracted and saved text for {team_name}: {output_file}")
    return text

def batch_extract(team_name: str):
    team_folder = Path(f"data/raw/{team_name}")
    for pdf_file in team_folder.glob("*.pdf"):
        extract_text_from_pdf(str(pdf_file), team_name)

if __name__ == "__main__":
    batch_extract("pakistan_strat")
    batch_extract("myanmar_strat")
