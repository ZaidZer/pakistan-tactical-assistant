if __name__ == "__main__":
    from src.pdf_extraction import extract_text_from_pdf
    from src.image_extraction import extract_images_from_pdf
    from src.embeddings import embed_and_index
    from src.RAG import run_strategy_advice

    text = extract_text_from_pdf("data/raw/pakistan_strat/Pakistan - Afghanistan.pdf")
    images = extract_images_from_pdf("data/raw/pakistan_strat/Pakistan - Afghanistan.pdf")
    embed_and_index(text, team="pakistan")
    response = run_strategy_advice("Compare Pakistan vs Myanmar and suggest tactics.")
    print(response)
