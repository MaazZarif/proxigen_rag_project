"""Shared formatting helpers for context and conversation history."""

MAX_CONTEXT_CHARS = 12000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_CHARS = 1500



def format_context(
    documents: list,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    parts = []
    total_chars = 0

    for i, doc in enumerate(documents or [], start=1):
        content = getattr(doc, "page_content", "").strip()

        if not content:
            continue

        metadata = getattr(doc, "metadata", {}) or {}

        header = (
            f"[Source: {metadata.get('source_file', 'Unknown')} | "
            f"Section: {metadata.get('section', '')} | "
            f"Subsection: {metadata.get('subsection', '')} | "
            f"Topic: {metadata.get('topic', '')}]"
        )

        block = f"{header}\n{content}"

        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        block = block[:remaining]
        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def format_history(messages: list) -> str:
    history = []

    for msg in (messages or [])[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content[:MAX_HISTORY_MESSAGE_CHARS]

        history.append(f"{role}: {content}")

    return "\n".join(history)