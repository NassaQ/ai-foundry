"""
High-level OCR pipeline.

Orchestrates:  file validation -> (split if needed) -> OCR engine -> merge -> post-process.

Large PDFs (>4 MB or >2000 pages) are automatically split into chunks,
each chunk is sent to Azure separately, and the results are merged
transparently before being returned to the caller.

Usage:
    from ocr_pipeline import OCRPipeline

    pipeline = OCRPipeline()                        # uses .env config
    result   = pipeline.run("scanned_report.pdf")   # PipelineResult

    print(result.cleaned_text)
    print(result.quality)
    for t in result.tables_markdown:
        print(t)
"""

import os
import time
from typing import Optional
from dataclasses import dataclass, field

from ocr_engine import AzureOCREngine, OCRResult, ExtractedTable, PageInfo
from postprocessor import (
    build_document_text,
    detect_primary_language,
    quality_summary,
    page_diagnostics,
)
from pdf_splitter import needs_splitting, split_pdf_bytes


# ── Supported file extensions ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".heif",
    ".heic",
    ".docx",
    ".xlsx",
    ".pptx",
}


# ── Pipeline result ───────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    """Everything the pipeline produces for a single document."""

    raw_text: str = ""
    cleaned_text: str = ""
    primary_language: str = "unknown"

    # tables
    tables_markdown: list[str] = field(default_factory=list)
    tables_csv: list[str] = field(default_factory=list)

    # quality / metadata
    quality: dict = field(default_factory=dict)
    page_count: int = 0
    word_count: int = 0
    avg_confidence: float = 0.0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0

    # chunking info
    chunks_used: int = 1

    # per-page diagnostics
    per_page: list[dict] = field(default_factory=list)

    # status
    success: bool = False
    error: Optional[str] = None
    source_file: Optional[str] = None


# ── Pipeline ──────────────────────────────────────────────────────────────
class OCRPipeline:
    """
    End-to-end OCR pipeline with automatic large-file splitting.

    Azure Document Intelligence limits:
      - 4 MB per request
      - 2 000 pages per request

    Files that exceed either limit are split page-by-page into chunks,
    each chunk is processed independently, and the results are merged
    before being returned. The caller sees a single PipelineResult
    regardless of how many chunks were needed.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: str = "prebuilt-layout",
        strip_diacritics: bool = False,
        high_resolution: bool = True,
        locale: Optional[str] = None,
        output_format: str = "markdown",
    ):
        self.engine = AzureOCREngine(
            endpoint=endpoint,
            api_key=api_key,
            model_id=model_id,
            high_resolution=high_resolution,
            locale=locale,
            output_format=output_format,
        )
        self.strip_diacritics = strip_diacritics

    # ── public ────────────────────────────────────────────────────────────

    def run(self, file_path: str) -> PipelineResult:
        """Analyze a local file end-to-end."""
        err = self._validate_file(file_path)
        if err:
            return PipelineResult(error=err, source_file=file_path)

        with open(file_path, "rb") as f:
            data = f.read()

        return self.run_bytes(data, filename=file_path)

    def run_bytes(self, data: bytes, filename: str = "") -> PipelineResult:
        """
        Analyze raw bytes.

        Automatically splits PDFs that exceed Azure's 4 MB / 2000-page limits.
        Non-PDF files (images, DOCX, etc.) are sent as-is; they are always
        single-page and well within size limits.
        """
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        is_pdf = ext == ".pdf" or _looks_like_pdf(data)

        if is_pdf and needs_splitting(data):
            return self._run_chunked(data, filename)

        ocr = self.engine.analyze_bytes(data)
        return self._post_process(ocr, source_file=filename, chunks_used=1)

    def run_url(self, url: str) -> PipelineResult:
        """Analyze a document from a public URL (must be publicly reachable)."""
        ocr = self.engine.analyze_url(url)
        return self._post_process(ocr, source_file=url, chunks_used=1)

    # ── chunked processing ────────────────────────────────────────────────

    def _run_chunked(self, data: bytes, filename: str) -> PipelineResult:
        """
        Split the PDF into <=4 MB chunks, OCR each one, then merge.
        """
        chunks = list(split_pdf_bytes(data))
        total_chunks = len(chunks)

        print(
            f"[OCR] File is large — splitting into {total_chunks} chunk(s) "
            f"({len(data) / 1_048_576:.1f} MB total)"
        )

        ocr_results: list[OCRResult] = []

        for chunk_bytes, page_start, page_end in chunks:
            size_kb = len(chunk_bytes) / 1024
            print(
                f"[OCR] Processing pages {page_start}-{page_end} ({size_kb:.0f} KB) ..."
            )
            result = self.engine.analyze_bytes(chunk_bytes)
            if result.error:
                return PipelineResult(
                    error=f"Failed on pages {page_start}-{page_end}: {result.error}",
                    source_file=filename,
                )
            ocr_results.append(result)

        merged = _merge_ocr_results(ocr_results)
        return self._post_process(
            merged, source_file=filename, chunks_used=total_chunks
        )

    # ── post-processing ───────────────────────────────────────────────────

    def _post_process(
        self,
        ocr: OCRResult,
        source_file: str = "",
        chunks_used: int = 1,
    ) -> PipelineResult:
        if ocr.error:
            return PipelineResult(
                error=ocr.error,
                source_file=source_file,
                cost_usd=ocr.cost_usd,
                elapsed_seconds=ocr.elapsed_seconds,
            )

        cleaned = build_document_text(
            ocr,
            include_tables=False,
            clean=True,
            strip_diacritics=self.strip_diacritics,
        )

        lang = detect_primary_language(cleaned)
        quality = quality_summary(ocr)
        quality["chunks_used"] = chunks_used
        per_page = page_diagnostics(ocr)

        md_tables = [t.to_markdown() for t in ocr.tables]
        csv_tables = [t.to_csv() for t in ocr.tables]

        return PipelineResult(
            raw_text=ocr.text,
            cleaned_text=cleaned,
            primary_language=lang,
            tables_markdown=md_tables,
            tables_csv=csv_tables,
            quality=quality,
            page_count=ocr.page_count,
            word_count=ocr.word_count,
            avg_confidence=ocr.avg_confidence,
            cost_usd=ocr.cost_usd,
            elapsed_seconds=ocr.elapsed_seconds,
            chunks_used=chunks_used,
            per_page=per_page,
            success=True,
            source_file=source_file,
        )

    # ── validation ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_file(file_path: str) -> Optional[str]:
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return (
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 500:
            return f"File too large ({size_mb:.1f} MB). Maximum is 500 MB."

        return None


# ── Helpers ───────────────────────────────────────────────────────────────


def _looks_like_pdf(data: bytes) -> bool:
    """Quick magic-bytes check — no file extension needed."""
    return data[:4] == b"%PDF"


def _merge_ocr_results(results: list[OCRResult]) -> OCRResult:
    """
    Combine multiple OCRResult objects (one per chunk) into a single one.

    - text          : joined with double newlines
    - pages         : concatenated (page numbers kept as-is from each chunk,
                      they will be 1-relative within each chunk — that's fine
                      for downstream use)
    - tables        : concatenated
    - languages     : union of detected languages
    - totals        : summed / averaged as appropriate
    """
    if not results:
        return OCRResult(text="", error="No chunks to merge")

    if len(results) == 1:
        return results[0]

    combined_text = "\n\n".join(r.text for r in results if r.text)
    combined_pages: list[PageInfo] = []
    combined_tables: list[ExtractedTable] = []
    all_languages: list[str] = []

    total_words = 0
    total_cost = 0.0
    total_elapsed = 0.0
    conf_sum = 0.0
    conf_count = 0

    for r in results:
        combined_pages += r.pages
        combined_tables += r.tables

        for lang in r.languages:
            if lang not in all_languages:
                all_languages.append(lang)

        total_words += r.word_count
        total_cost += r.cost_usd
        total_elapsed += r.elapsed_seconds

        if r.avg_confidence > 0:
            conf_sum += r.avg_confidence * r.word_count
            conf_count += r.word_count

    avg_conf = (conf_sum / conf_count) if conf_count else 0.0

    return OCRResult(
        text=combined_text,
        pages=combined_pages,
        tables=combined_tables,
        languages=all_languages,
        page_count=len(combined_pages),
        word_count=total_words,
        avg_confidence=round(avg_conf, 4),
        cost_usd=round(total_cost, 6),
        model_id=results[0].model_id,
        elapsed_seconds=round(total_elapsed, 2),
    )
