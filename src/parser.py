"""PDF parsing using PyMuPDF."""

from __future__ import annotations

from typing import BinaryIO, Iterable

import fitz

from src.utils import Document


def parse_pdf(file: BinaryIO, filename: str | None = None) -> list[Document]:
    """Extract non-empty pages from an uploaded PDF."""
    name = filename or getattr(file, "name", "document.pdf")
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    documents: list[Document] = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append(Document(text=text, source=name, page=page_number))
    return documents


def parse_pdfs(files: Iterable[BinaryIO]) -> list[Document]:
    documents: list[Document] = []
    for file in files:
        documents.extend(parse_pdf(file, getattr(file, "name", None)))
    return documents

