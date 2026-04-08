import re


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting so the classifier receives plain text.

    Strips heading markers, bold/italic, table pipes, horizontal rules,
    link/image syntax.
    """
    # Remove images ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove links [text](url)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"(\*{1,3}|_{1,3})", "", text)
    # Remove table separator rows  |---|---|
    text = re.sub(r"^\|[\s\-:|]+\|$", "", text, flags=re.MULTILINE)
    # Remove leading/trailing pipes on table rows
    text = re.sub(r"^\|(.+)\|$", r"\1", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^(\-{3,}|\*{3,}|_{3,})$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
