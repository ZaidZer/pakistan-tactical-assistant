# src/image_captioning.py
from openai import OpenAI
from pathlib import Path
import base64

client = OpenAI()

def caption_image(image_path: str):
    """Generate a tactical caption for a soccer report page using GPT-4o."""
    with open(image_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" for higher quality
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this soccer tactics page in detail: "
                            "identify formations, player positions, team shapes, and key observations. "
                            "Focus on how Pakistan and the opponent are arranged and how shapes change."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.5,
    )

    caption = response.choices[0].message.content
    return caption



def batch_caption(team_name: str):
    img_folder = Path(f"data/processed/{team_name}/page_images")
    out_folder = Path(f"data/processed/{team_name}/image_captions")
    out_folder.mkdir(parents=True, exist_ok=True)

    # 👇 Add this line here
    print(f"[INFO] Looking for images in: {img_folder.resolve()}")

    for img_file in img_folder.glob("*.png"):
        caption_file = out_folder / f"{img_file.stem}.txt"

        print(f"[+] Captioning {img_file.name} ...")
        caption = caption_image(str(img_file))

        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)

        print(f"    Saved: {caption_file}")


if __name__ == "__main__":
    batch_caption("pakistan_strat")
    batch_caption("myanmar_strat")
