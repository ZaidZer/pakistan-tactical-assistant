# src/image_extraction.py
import fitz  # PyMuPDF
from pathlib import Path

def render_pdf_pages(pdf_path: str, team_name: str, dpi: int = 200):
    doc = fitz.open(pdf_path)
    output_dir = Path(f"data/processed/{team_name}/page_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        image_path = output_dir / f"{Path(pdf_path).stem}_page{i+1}.png"
        pix.save(image_path)

    print(f"[+] Rendered {len(doc)} pages from {pdf_path} → {output_dir}")

def batch_render(team_name: str):
    raw_folder = Path(f"data/raw/{team_name}")
    for pdf_file in raw_folder.glob("*.pdf"):
        render_pdf_pages(str(pdf_file), team_name)

if __name__ == "__main__":
    batch_render("pakistan_strat")
    batch_render("myanmar_strat")
