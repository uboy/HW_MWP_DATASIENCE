"""PDF document processor: smart chunking with structural metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import pymupdf4llm
    _PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    _PYMUPDF4LLM_AVAILABLE = False

import fitz  # pymupdf — always available

SPACE_RE = re.compile(r"\s+")
# Universal paragraph number: "46. " at line start, 1–3 digits
PARA_RE = re.compile(r"^\s{0,4}(\d{1,3})\.\s+\S")
# Markdown header
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
# List item starters (language-agnostic)
LIST_START_RE = re.compile(r"^\s*(?:[а-яёА-ЯЁa-zA-Z]\)|[•\-\*])\s")


@dataclass
class ChunkMetadata:
    chunk_id: int
    page: int
    section_title: str | None
    paragraph_number: int | None
    chunk_type: Literal["header", "body"]
    paragraph_numbers: tuple[int, ...] = ()


@dataclass
class StructuredChunk:
    text: str
    metadata: ChunkMetadata


def _normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to markdown. Uses pymupdf4llm when available, else plain text."""
    if _PYMUPDF4LLM_AVAILABLE:
        return pymupdf4llm.to_markdown(str(pdf_path))
    # Fallback: plain text via fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


class DocumentProcessor:
    """Converts a PDF to a list of StructuredChunks with section and paragraph metadata."""

    def __init__(self, chunk_size: int = 1100, overlap: int = 220) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, pdf_path: Path) -> list[StructuredChunk]:
        md_text = _pdf_to_markdown(pdf_path)
        chunks = self._smart_split(md_text)
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smart_split(self, text: str) -> list[StructuredChunk]:  # noqa: C901
        """Split markdown text into StructuredChunks preserving structure."""
        lines = text.splitlines()
        chunks: list[StructuredChunk] = []

        current_lines: list[str] = []
        current_section: str | None = None
        current_para: int | None = None
        current_paragraphs: list[int] = []
        current_page: int = 0
        chunk_id: int = 0
        is_first_chunk = True

        def flush(chunk_type: Literal["header", "body"] = "body") -> None:
            nonlocal chunk_id, is_first_chunk
            raw = "\n".join(current_lines).strip()
            # collapse excessive blank lines
            raw = re.sub(r"\n{3,}", "\n\n", raw)
            if not raw:
                return
            ct = "header" if is_first_chunk else chunk_type
            paragraph_numbers = tuple(current_paragraphs)
            chunks.append(
                StructuredChunk(
                    text=_normalize(raw),
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        page=current_page,
                        section_title=current_section,
                        paragraph_number=paragraph_numbers[0] if paragraph_numbers else current_para,
                        chunk_type=ct,
                        paragraph_numbers=paragraph_numbers,
                    ),
                )
            )
            chunk_id += 1
            is_first_chunk = False

        page_sep_count = 0

        for line in lines:
            # Track page separators that pymupdf4llm emits
            stripped = line.strip()
            if stripped == "-----":
                page_sep_count += 1
                current_page = page_sep_count
                current_lines.append(line)
                continue

            # Detect markdown headers
            hm = HEADER_RE.match(line)
            if hm:
                title = hm.group(2).strip()
                # Flush current chunk before starting a new section
                if current_lines:
                    flush()
                    current_lines = []
                    current_paragraphs = []
                current_section = title
                current_para = None
                current_lines = [line]
                continue

            # Detect numbered paragraph start
            pm = PARA_RE.match(line)
            if pm:
                para_num = int(pm.group(1))
                # If current chunk is getting large, flush before starting new paragraph
                current_text = "\n".join(current_lines)
                if len(current_text) > self.chunk_size - 150 and current_lines:
                    flush()
                    # Keep a small overlap: last overlap characters as context
                    tail = current_text[-self.overlap:] if len(current_text) > self.overlap else ""
                    current_lines = [tail] if tail.strip() else []
                    current_paragraphs = [current_para] if current_para is not None else []
                current_para = para_num
                if para_num not in current_paragraphs:
                    current_paragraphs.append(para_num)
                current_lines.append(line)
                continue

            # Regular line — just append
            current_lines.append(line)

            # Check if chunk is too big and we can safely split here
            if not LIST_START_RE.match(line):
                current_text = "\n".join(current_lines)
                if len(current_text) > self.chunk_size:
                    flush()
                    tail = current_text[-self.overlap:] if len(current_text) > self.overlap else ""
                    current_lines = [tail] if tail.strip() else []
                    current_paragraphs = [current_para] if current_para is not None else []

        # Flush remaining
        if current_lines:
            flush()

        return chunks
