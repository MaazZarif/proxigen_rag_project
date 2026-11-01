import hashlib
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3] 
CLEANED_DATA_DIR =  PROJECT_ROOT / "data" / "cleaned"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEQUENCE_MAX = 2200 
STEP_TOPIC = re.compile(r"^\s*(Step\s*\d+|\d+\.)", re.IGNORECASE)

# File stem -> product. Anything not listed covers both models.
PRODUCT_BY_FILE = {"cmt01": "CTM01", "cmt02": "CTM02"}
DEFAULT_PRODUCT = "CTM01 and CTM02"

# Authoring notes that must never be indexed as content.
AUTHORING_NOTE = re.compile(r"^\s*\[?\s*Preserve the\b.*$", re.IGNORECASE)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def tables_to_sentences(text: str) -> str:
    """Turn Markdown table rows into standalone 'label: value' lines."""
    out: list[str] = []
    table: list[str] = []

    def flush() -> None:
        if not table:
            return
        rows = [
            [cell.strip() for cell in row.strip().strip("|").split("|")]
            for row in table
        ]
        rows = [
            r for r in rows if not all(TABLE_SEPARATOR_CELL.fullmatch(c) for c in r)
        ]
        table.clear()
        if len(rows) < 2:
            return
        header, body = rows[0], rows[1:]
        for r in body:
            if len(header) == 2:
                out.append(f"{r[0]}: {r[1]}")
            else:
                pairs = "; ".join(f"{h} = {v}" for h, v in zip(header[1:], r[1:]))
                out.append(f"{r[0]}: {pairs}")

    for line in text.split("\n"):
        if TABLE_ROW.match(line):
            table.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def clean_text(text: str) -> str:
    """Normalize whitespace, drop authoring notes, flatten tables."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not AUTHORING_NOTE.match(line)]
    text = "\n".join(lines)
    text = tables_to_sentences(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Chunk text
# --------------------------------------------------------------------------
def prepare_chunk_text(document: Document) -> str:
    """Prefix the chunk with product / document / heading context."""
    m = document.metadata

    crumb_parts = [m.get("section", ""), m.get("subsection", "")]
    topic = m.get("topic", "")
    if topic and ";" not in topic:  # packed chunks already contain their ### headings
        crumb_parts.append(topic)
    crumb = " > ".join(p.strip() for p in crumb_parts if p and p.strip())

    header = f"[{m.get('product', DEFAULT_PRODUCT)} | {m.get('document_name', '')}]"
    if crumb:
        header = f"{header} {crumb}"

    content = document.page_content.strip()
    if content.startswith("[") and "|" in content.split("\n", 1)[0]:
        return content  # already prefixed; never add the header twice
    return f"{header}\n\n{content}"


def group_step_sequences(sections: list[Document]) -> list[Document]:
    """Merge consecutive 'Step N' / 'N.' topics under one parent into ONE section.

    Keeps a procedure (e.g. the 8 app-setup steps) together instead of letting
    the packer cut it in the middle.
    """
    out: list[Document] = []
    run: list[Document] = []

    def key(d: Document):
        return (d.metadata.get("section", ""), d.metadata.get("subsection", ""))

    def flush_run() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            meta = dict(run[0].metadata)
            meta["topic"] = "; ".join(
                d.metadata.get("topic", "") for d in run if d.metadata.get("topic")
            )
            meta["_seq"] = True
            text = "\n\n".join(d.page_content.strip() for d in run)
            out.append(Document(page_content=text, metadata=meta))
        run.clear()

    for sec in sections:
        is_step = bool(STEP_TOPIC.match(sec.metadata.get("topic", "")))
        if is_step and run and key(run[0]) == key(sec):
            run.append(sec)
            continue
        flush_run()
        if is_step:
            run.append(sec)
        else:
            out.append(sec)
    flush_run()
    return out


def pack_sections(
    sections: list[Document], splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """Pack sibling sections (same section + subsection) up to CHUNK_SIZE.

    Sections larger than CHUNK_SIZE are split on their own, exactly once.
    """
    packed: list[Document] = []
    cur_text = ""
    cur_meta: dict | None = None
    cur_topics: list[str] = []

    def flush() -> None:
        nonlocal cur_text, cur_meta, cur_topics
        if cur_text.strip() and cur_meta is not None:
            meta = dict(cur_meta)
            meta["topic"] = "; ".join(dict.fromkeys(cur_topics))
            packed.append(Document(page_content=cur_text.strip(), metadata=meta))
        cur_text, cur_meta, cur_topics = "", None, []

    for sec in sections:
        content = sec.page_content.strip()
        if not content:
            continue

        meta = dict(sec.metadata)
        limit = SEQUENCE_MAX if meta.pop("_seq", False) else CHUNK_SIZE
        topic = meta.get("topic", "")
        key = (meta.get("section", ""), meta.get("subsection", ""))

        # Oversized: flush what we have, split this one section, move on.
        if len(content) > limit:
            flush()
            packed.extend(
                splitter.split_documents([Document(page_content=content, metadata=meta)])
            )
            continue

        cur_key = (
            (cur_meta.get("section", ""), cur_meta.get("subsection", ""))
            if cur_meta is not None
            else None
        )
        candidate = f"{cur_text}\n\n{content}" if cur_text else content

        if cur_meta is not None and (cur_key != key or len(candidate) > CHUNK_SIZE):
            flush()
            candidate = content

        if cur_meta is None:
            cur_meta = meta
        cur_text = candidate
        if topic:
            cur_topics.append(topic)

    flush()
    return packed


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def load_and_split_documents() -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "subsection"), ("###", "topic")],
        strip_headers=False,  # keep each section's own heading inside its text
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for file_path in sorted(CLEANED_DATA_DIR.glob("*.txt")):
        text = clean_text(file_path.read_text(encoding="utf-8"))
        if not text:
            continue

        product = PRODUCT_BY_FILE.get(file_path.stem, DEFAULT_PRODUCT)

        sections = markdown_splitter.split_text(text)
        for sec in sections:
            for key in ("section", "subsection", "topic"):
                sec.metadata.setdefault(key, "")
            sec.metadata.update(
                source_file=file_path.name,
                document_name=file_path.stem,
                product=product,
            )

        sections = group_step_sequences(sections)
        file_chunks = pack_sections(sections, text_splitter)

        for i, chunk in enumerate(file_chunks):
            chunk.metadata["chunk_index"] = i  # per-file, so IDs stay stable
            chunk.page_content = prepare_chunk_text(chunk)  # embed with context
            chunk.metadata["chunk_id"] = hashlib.sha256(
                f"{file_path.name}:{i}:{chunk.page_content}".encode("utf-8")
            ).hexdigest()
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_split_documents()
    print(f"\nTotal chunks: {len(chunks)}")

    for n, chunk in enumerate(chunks):
        print(f"\n{'=' * 80}")
        print(f"Chunk {n} | {len(chunk.page_content)} chars")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content)