
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
LINKS_DIR = DATA_DIR / "links"


def extract_links(text: str) -> list[dict]:
    """Extract Markdown links and images before cleaning."""

    links = []

    # Extract Markdown links: [label](URL)
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        links.append({
            "text": match.group(1).strip(),
            "url": match.group(2).strip(),
            "type": "markdown_link",
        })

    # Extract Markdown images: ![alt](URL)
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        links.append({
            "text": match.group(1).strip(),
            "url": match.group(2).strip(),
            "type": "image",
        })

    # Remove duplicate entries without losing their original order.
    unique_links = []
    seen = set()

    for link in links:
        key = (link["text"], link["url"], link["type"])

        if key not in seen:
            seen.add(key)
            unique_links.append(link)

    return unique_links


def clean_markdown(text: str) -> str:
    # Remove images from the text while retaining them in extracted links.
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

    # Keep link labels, but remove URLs from the text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove obvious navigation noise.
    text = re.sub(
        r"(?im)^\s*(?:Skip to content|Scroll to Top)\s*$",
        "",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", text)

    return text.strip() + "\n"


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    LINKS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.txt"))

    if not files:
        print(f"No text files found in {RAW_DIR}")
        return

    for file_path in files:
        original = file_path.read_text(encoding="utf-8")

        # Extract first, before removing links from the text.
        links = extract_links(original)
        cleaned = clean_markdown(original)

        (CLEANED_DIR / file_path.name).write_text(
            cleaned, encoding="utf-8"
        )

        # Store links separately for each source page.
        links_path = LINKS_DIR / f"{file_path.stem}.json"
        links_path.write_text(
            json.dumps(links, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"{file_path.name}: "
            f"{len(links)} links extracted"
        )

    print("Cleaning and link extraction completed.")


if __name__ == "__main__":
    main()
