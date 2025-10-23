import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str, team_name: str):
    """Extracts text from a single PDF and saves it under processed/{team_name}"""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    # Save extracted text
    output_dir = Path(f"data/processed/{team_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(pdf_path).stem}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"[+] Extracted and saved text for {team_name}: {output_file}")
    return text


def extract_all_from_team(team_name: str):
    """Extract text for all PDFs inside a given team’s raw folder."""
    raw_dir = Path(f"data/raw/{team_name}")
    if not raw_dir.exists():
        print(f"[!] Folder not found: {raw_dir}")
        return

    for pdf_file in raw_dir.glob("*.pdf"):
        extract_text_from_pdf(str(pdf_file), team_name)


if __name__ == "__main__":
    extract_all_from_team("pakistan_strat")
    extract_all_from_team("myanmar_strat")
