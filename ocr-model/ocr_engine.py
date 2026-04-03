"""
Core OCR engine using Azure AI Document Intelligence.

Wraps the Azure SDK to provide a clean interface for analyzing
documents (PDFs, images) and extracting text, tables, and layout.

Key design decisions:
  - output_content_format = MARKDOWN for maximum text recall
  - features = [OCR_HIGH_RESOLUTION, LANGUAGES] for scanned / Arabic docs
  - locale = "ar" hint for Arabic-heavy documents
  - body wrapped in BytesIO (documented SDK pattern)
  - page-level text assembly as fallback when result.content is sparse
"""

import io
import os
import time
from typing import Optional
from dataclasses import dataclass, field

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    DocumentTable,
    DocumentAnalysisFeature,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
PRICING = {
    "prebuilt-read": 0.0015,
    "prebuilt-layout": 0.0100,
}

# Add-on cost per feature per page
FEATURE_PRICING = {
    "ocr.highResolution": 0.0100, 
    "languages": 0.0,  
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TableCell:
    """Single cell inside a table."""

    row: int
    col: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


@dataclass
class ExtractedTable:
    """A table found in the document."""

    page: int
    row_count: int
    col_count: int
    cells: list[TableCell] = field(default_factory=list)

    def to_rows(self) -> list[list[str]]:
        """Return table as a 2-D list of strings (row-major order)."""
        grid: list[list[str]] = [
            ["" for _ in range(self.col_count)] for _ in range(self.row_count)
        ]
        for c in self.cells:
            if c.row < self.row_count and c.col < self.col_count:
                grid[c.row][c.col] = c.text
        return grid

    def to_csv(self, delimiter: str = ",") -> str:
        rows = self.to_rows()
        lines: list[str] = []
        for row in rows:
            escaped = [
                f'"{cell}"' if delimiter in cell or '"' in cell else cell
                for cell in row
            ]
            lines.append(delimiter.join(escaped))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = self.to_rows()
        if not rows:
            return ""
        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * self.col_count) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in rows[1:]]
        return "\n".join([header, separator, *body_lines])


@dataclass
class PageInfo:
    """Metadata for a single page."""

    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    unit: Optional[str] = None
    angle: Optional[float] = None
    lines_count: int = 0
    words_count: int = 0
    text: str = "" 
    avg_confidence: float = 0.0


@dataclass
class OCRResult:
    """Complete result returned by the OCR engine."""

    text: str  
    pages: list[PageInfo] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    page_count: int = 0
    word_count: int = 0
    avg_confidence: float = 0.0
    cost_usd: float = 0.0
    model_id: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class AzureOCREngine:
    """
    Wrapper around Azure AI Document Intelligence with full-recall settings.

    Critical parameters for maximizing extraction quality:

    1. output_content_format = MARKDOWN
       The TEXT format drops content from complex layouts (multi-column,
       headers, sidebars).  MARKDOWN preserves structure and captures
       significantly more text.

    2. features = [OCR_HIGH_RESOLUTION, LANGUAGES]
       OCR_HIGH_RESOLUTION: uses a higher-DPI pipeline for scanned pages
       that would otherwise produce fragmented or missing words.
       LANGUAGES: enables per-span language detection, critical for
       Arabic text that might otherwise be treated as noise.

    3. locale = "ar"
       Hints the engine that Arabic is expected.  Without this, the
       engine defaults to English-first heuristics and can skip Arabic
       glyphs on mixed-language scans.

    4. pages = "1-"
       Explicitly requests all pages.

    5. body = BytesIO(data)
       The SDK type signature expects IO[bytes].  Raw bytes are coerced
       in some versions but wrapping avoids edge-case failures.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: str = "prebuilt-layout",
        high_resolution: bool = True,
        locale: Optional[str] = None,
        output_format: str = "markdown",
    ):
        self.endpoint = endpoint or os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
        self.api_key = api_key or os.getenv("AZURE_DOC_INTELLIGENCE_KEY", "")
        self.model_id = model_id
        self.high_resolution = high_resolution
        self.locale = locale
        self.output_format = output_format

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing Azure Document Intelligence credentials.\n"
                "Set environment variables:\n"
                "  AZURE_DOC_INTELLIGENCE_ENDPOINT\n"
                "  AZURE_DOC_INTELLIGENCE_KEY"
            )

        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )

    # ── public ────────────────────────────────────────────────────────────

    def analyze_file(self, file_path: str) -> OCRResult:
        """Analyze a local file."""
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return self.analyze_bytes(file_bytes)

    def analyze_bytes(self, data: bytes) -> OCRResult:
        """Analyze raw file bytes with full-recall settings."""
        start = time.time()
        try:
            features = self._build_features()
            content_fmt = self._content_format()

            poller = self.client.begin_analyze_document(
                model_id=self.model_id,
                body=io.BytesIO(data),
                pages="1-",
                locale=self.locale,
                features=features if features else None,
                output_content_format=content_fmt,
                content_type="application/octet-stream",
            )
            result: AnalyzeResult = poller.result()
            elapsed = time.time() - start
            return self._build_result(result, elapsed)

        except Exception as exc:
            return OCRResult(
                text="",
                error=str(exc),
                model_id=self.model_id,
                elapsed_seconds=time.time() - start,
            )

    def analyze_url(self, url: str) -> OCRResult:
        """Analyze a document from a public URL."""
        start = time.time()
        try:
            features = self._build_features()
            content_fmt = self._content_format()

            poller = self.client.begin_analyze_document(
                model_id=self.model_id,
                body=AnalyzeDocumentRequest(url_source=url),
                pages="1-",
                locale=self.locale,
                features=features if features else None,
                output_content_format=content_fmt,
            )
            result: AnalyzeResult = poller.result()
            elapsed = time.time() - start
            return self._build_result(result, elapsed)

        except Exception as exc:
            return OCRResult(
                text="",
                error=str(exc),
                model_id=self.model_id,
                elapsed_seconds=time.time() - start,
            )

    # ── private ───────────────────────────────────────────────────────────

    def _build_features(self) -> list:
        """Build feature flags list."""
        features = [DocumentAnalysisFeature.LANGUAGES]
        if self.high_resolution:
            features.append(DocumentAnalysisFeature.OCR_HIGH_RESOLUTION)
        return features

    def _content_format(self):
        """Return the output content format enum."""
        if self.output_format == "markdown":
            return DocumentContentFormat.MARKDOWN
        return DocumentContentFormat.TEXT

    def _build_result(self, raw: AnalyzeResult, elapsed: float) -> OCRResult:
        """Convert the SDK response into our OCRResult dataclass."""

        api_text: str = raw.content or ""
        pages: list[PageInfo] = []
        total_words = 0
        confidence_sum = 0.0
        confidence_count = 0
        page_texts: list[str] = []

        for p in raw.pages or []:
            words_on_page = p.words or []
            lines_on_page = p.lines or []
            total_words += len(words_on_page)

            page_conf_sum = 0.0
            page_conf_count = 0
            for w in words_on_page:
                if w.confidence is not None:
                    confidence_sum += w.confidence
                    confidence_count += 1
                    page_conf_sum += w.confidence
                    page_conf_count += 1

            page_avg_conf = (
                (page_conf_sum / page_conf_count) if page_conf_count else 0.0
            )

            page_line_texts = [line.content for line in lines_on_page if line.content]
            page_text = "\n".join(page_line_texts)
            page_texts.append(page_text)

            pages.append(
                PageInfo(
                    page_number=p.page_number,
                    width=p.width,
                    height=p.height,
                    unit=p.unit,
                    angle=p.angle,
                    lines_count=len(lines_on_page),
                    words_count=len(words_on_page),
                    text=page_text,
                    avg_confidence=round(page_avg_conf, 4),
                )
            )

        avg_conf = (confidence_sum / confidence_count) if confidence_count else 0.0

        page_assembled = "\n\n".join(t for t in page_texts if t)

        api_word_count = len(api_text.split())
        page_word_count = len(page_assembled.split())

        if page_word_count > api_word_count * 1.15:
            full_text = page_assembled
            print(
                f"[OCR] Text recall fallback activated: "
                f"API content={api_word_count} words, "
                f"page assembly={page_word_count} words. "
                f"Using page-level assembly."
            )
        else:
            full_text = api_text

        final_word_count = max(total_words, len(full_text.split()))

        tables: list[ExtractedTable] = []
        for t in raw.tables or []:
            tables.append(self._parse_table(t))

        detected_langs: list[str] = []
        for lang in raw.languages or []:
            if lang.locale and lang.locale not in detected_langs:
                detected_langs.append(lang.locale)

        page_count = len(pages)
        base_cost = page_count * PRICING.get(self.model_id, 0.01)
        feature_cost = 0.0
        if self.high_resolution:
            feature_cost += page_count * FEATURE_PRICING.get("ocr.highResolution", 0.0)
        cost = base_cost + feature_cost

        return OCRResult(
            text=full_text,
            pages=pages,
            tables=tables,
            languages=detected_langs,
            page_count=page_count,
            word_count=final_word_count,
            avg_confidence=round(avg_conf, 4),
            cost_usd=round(cost, 6),
            model_id=self.model_id,
            elapsed_seconds=round(elapsed, 2),
        )

    @staticmethod
    def _parse_table(raw_table: DocumentTable) -> ExtractedTable:
        cells: list[TableCell] = []
        page_number = 1

        for c in raw_table.cells or []:
            is_header = (c.kind == "columnHeader") if c.kind else False
            cells.append(
                TableCell(
                    row=c.row_index,
                    col=c.column_index,
                    text=c.content or "",
                    is_header=is_header,
                    row_span=c.row_span or 1,
                    col_span=c.column_span or 1,
                )
            )
            if c.bounding_regions:
                page_number = c.bounding_regions[0].page_number

        return ExtractedTable(
            page=page_number,
            row_count=raw_table.row_count,
            col_count=raw_table.column_count,
            cells=cells,
        )
