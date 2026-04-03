"""
Post-processing utilities for OCR output.

Handles:
  - whitespace normalization
  - Arabic text cleanup (diacritics, kashida, directional marks)
  - mixed RTL / LTR paragraph assembly
  - table rendering to markdown / CSV / plain text
  - per-page diagnostics for quality analysis
"""

import re
import unicodedata
from typing import Optional
from ocr_engine import OCRResult, ExtractedTable, PageInfo


# ── Arabic-specific Unicode ranges ────────────────────────────────────────

# Optional diacritical marks (tashkeel): fathah, dammah, kasrah, etc.
_ARABIC_DIACRITICS = re.compile("[\u0617-\u061a\u064b-\u0652\u0656-\u065f\u0670]")

# Kashida / tatweel (decorative elongation)
_KASHIDA = "\u0640"

# Unicode bidirectional control characters
_BIDI_CONTROLS = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

# Arabic letter range (for detection)
_ARABIC_LETTER = re.compile(
    "[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]"
)


# ── Public helpers ────────────────────────────────────────────────────────


def clean_text(
    text: str,
    strip_diacritics: bool = False,
    strip_kashida: bool = True,
    normalize_whitespace: bool = True,
    strip_bidi_marks: bool = True,
) -> str:
    """
    Clean raw OCR text.

    Args:
        text:                 raw string from the OCR engine.
        strip_diacritics:     remove Arabic tashkeel marks (default False
                              because diacritics carry meaning).
        strip_kashida:        remove tatweel elongation (almost always noise).
        normalize_whitespace: collapse multiple spaces / blank lines.
        strip_bidi_marks:     remove Unicode LTR/RTL override characters.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    if strip_diacritics:
        text = _ARABIC_DIACRITICS.sub("", text)

    if strip_kashida:
        text = text.replace(_KASHIDA, "")

    if strip_bidi_marks:
        text = _BIDI_CONTROLS.sub("", text)

    if normalize_whitespace:
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

    return text


def detect_primary_language(text: str) -> str:
    """
    Quick heuristic: returns 'ar' if Arabic characters dominate,
    'en' if Latin characters dominate, or 'mixed'.
    """
    if not text:
        return "unknown"

    arabic_count = len(_ARABIC_LETTER.findall(text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    total = arabic_count + latin_count

    if total == 0:
        return "unknown"

    ar_ratio = arabic_count / total

    if ar_ratio > 0.7:
        return "ar"
    elif ar_ratio < 0.3:
        return "en"
    return "mixed"


def tables_to_text(tables: list[ExtractedTable], fmt: str = "markdown") -> str:
    """
    Render all extracted tables as a single string block.

    Args:
        tables: list of ExtractedTable from OCRResult.
        fmt:    'markdown', 'csv', or 'plain'.
    """
    if not tables:
        return ""

    parts: list[str] = []
    for idx, table in enumerate(tables, start=1):
        header = f"Table {idx} (page {table.page}, {table.row_count}x{table.col_count})"

        if fmt == "markdown":
            body = table.to_markdown()
        elif fmt == "csv":
            body = table.to_csv()
        else:
            rows = table.to_rows()
            body = "\n".join("\t".join(row) for row in rows)

        parts.append(f"### {header}\n{body}")

    return "\n\n".join(parts)


def build_document_text(
    result: OCRResult,
    include_tables: bool = True,
    table_format: str = "markdown",
    clean: bool = True,
    strip_diacritics: bool = False,
) -> str:
    """
    Produce a single string combining body text and tables,
    ready to feed into the classification / embedding pipeline.

    Args:
        result:            OCRResult from the engine.
        include_tables:    append a tables section at the end.
        table_format:      'markdown', 'csv', or 'plain'.
        clean:             apply clean_text() to the body.
        strip_diacritics:  passed through to clean_text().
    """
    body = result.text
    if clean:
        body = clean_text(body, strip_diacritics=strip_diacritics)

    sections: list[str] = [body]

    if include_tables and result.tables:
        table_block = tables_to_text(result.tables, fmt=table_format)
        sections.append(table_block)

    return "\n\n".join(sections)


def quality_summary(result: OCRResult) -> dict:
    """
    Return a dict summarising extraction quality, useful for logging
    and for populating the metadata schema's `ocr_confidence` field.
    """
    return {
        "page_count": result.page_count,
        "word_count": result.word_count,
        "avg_confidence": result.avg_confidence,
        "languages": result.languages,
        "tables_found": len(result.tables),
        "cost_usd": result.cost_usd,
        "elapsed_seconds": result.elapsed_seconds,
        "model_id": result.model_id,
        "has_error": result.error is not None,
    }


def page_diagnostics(result: OCRResult) -> list[dict]:
    """
    Return per-page diagnostics for quality analysis.

    Each entry includes:
      - page_number
      - words_count
      - lines_count
      - avg_confidence
      - status: 'good', 'low_confidence', 'sparse', or 'empty'

    This helps identify individual pages that may have failed OCR
    or produced poor results.
    """
    diagnostics: list[dict] = []

    for page in result.pages:
        if page.words_count == 0:
            status = "empty"
        elif page.words_count < 5:
            status = "sparse"
        elif page.avg_confidence < 0.5:
            status = "low_confidence"
        else:
            status = "good"

        diagnostics.append(
            {
                "page": page.page_number,
                "words": page.words_count,
                "lines": page.lines_count,
                "confidence": round(page.avg_confidence, 4),
                "status": status,
            }
        )

    return diagnostics
