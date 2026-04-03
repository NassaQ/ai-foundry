"""
PDF splitter for large files that exceed Azure Document Intelligence limits.

Azure limits:
  - Max file size : 4 MB per request
  - Max pages     : 2 000 per request

Strategy: split the PDF into page-range chunks that each stay under
CHUNK_SIZE_BYTES. Each chunk is returned as raw bytes so it can be
sent directly to the OCR engine without touching the filesystem.
"""

import io
from typing import Generator

from pypdf import PdfReader, PdfWriter

MAX_BYTES: int = 4 * 1024 * 1024  # 4 MB
MAX_PAGES: int = 2_000

CHUNK_SIZE_BYTES: int = int(MAX_BYTES * 0.90)


def _write_pages_to_bytes(reader: PdfReader, page_indices: list[int]) -> bytes:
    """Render a subset of pages from a PdfReader into a bytes PDF."""
    writer = PdfWriter()
    for i in page_indices:
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def split_pdf_bytes(data: bytes) -> Generator[tuple[bytes, int, int], None, None]:
    """
    Split a PDF (given as raw bytes) into chunks that each fit within
    Azure's 4 MB size limit.

    Yields:
        (chunk_bytes, first_page_1indexed, last_page_1indexed)

    Example:
        for chunk, start, end in split_pdf_bytes(data):
            print(f"Sending pages {start}-{end}")
            result = engine.analyze_bytes(chunk)
    """
    reader = PdfReader(io.BytesIO(data))
    total_pages = len(reader.pages)

    if len(data) <= CHUNK_SIZE_BYTES and total_pages <= MAX_PAGES:
        yield data, 1, total_pages
        return

    chunk_pages: list[int] = []

    for i in range(total_pages):
        chunk_pages.append(i)

        candidate = _write_pages_to_bytes(reader, chunk_pages)

        needs_flush = (
            len(candidate) >= CHUNK_SIZE_BYTES
            or len(chunk_pages) >= MAX_PAGES
            or i == total_pages - 1 
        )

        if needs_flush:
            if len(candidate) >= CHUNK_SIZE_BYTES and len(chunk_pages) > 1:
                without_last = _write_pages_to_bytes(reader, chunk_pages[:-1])
                first = chunk_pages[0] + 1  
                last = chunk_pages[-2] + 1
                yield without_last, first, last
                chunk_pages = [i]  
                if i == total_pages - 1:
                    final = _write_pages_to_bytes(reader, chunk_pages)
                    yield final, i + 1, i + 1
            else:
                first = chunk_pages[0] + 1
                last = chunk_pages[-1] + 1
                yield candidate, first, last
                chunk_pages = []


def needs_splitting(data: bytes) -> bool:
    """Return True if the file exceeds Azure's limits and must be split."""
    if len(data) > CHUNK_SIZE_BYTES:
        return True
    try:
        reader = PdfReader(io.BytesIO(data))
        return len(reader.pages) > MAX_PAGES
    except Exception:
        return False
