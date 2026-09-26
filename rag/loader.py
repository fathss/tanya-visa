"""Load source documents into chunks carrying citation metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config

_ENCODER = tiktoken.get_encoding(config.TOKENIZER_ENCODING)

_DROP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form", "iframe")
_HEADING_TAGS = ("h1", "h2", "h3", "h4")
_TEXT_TAGS = ("p", "li", "td")
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    locator_type: str
    locator_value: str | int
    title: str

    @property
    def locator_label(self) -> str:
        if self.locator_type == "page":
            return f"Halaman {self.locator_value}"
        if self.locator_type == "document":
            return ""
        return f"Bagian: {self.locator_value}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "locator_type": self.locator_type,
            "locator_value": self.locator_value,
            "title": self.title,
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=lambda text: len(_ENCODER.encode(text)),
        separators=_SEPARATORS,
    )


_CONTENT_HINTS = ("content", "article", "post", "entry", "main")
_MIN_CONTENT_CHARS = 200


def _document_name(soup: BeautifulSoup, path: Path) -> str:
    headings = soup.find_all("h1")
    if len(headings) == 1:
        heading = _clean(headings[0].get_text(" ", strip=True))
        if heading:
            return heading

    if soup.title is not None:
        title = _clean(soup.title.get_text())
        if title:
            return re.split(r"\s+[–—|]\s+|\s+-\s+", title)[0].strip()

    return path.stem


def _content_root(soup: BeautifulSoup):
    candidates = []

    main = soup.find("main")
    if main is not None:
        candidates.append(main)

    for element in soup.find_all(["div", "section", "article"]):
        if element.name == "section":
            candidates.append(element)
            continue
        marker = " ".join(element.get("class") or []) + " " + (element.get("id") or "")
        if any(hint in marker.lower() for hint in _CONTENT_HINTS):
            candidates.append(element)

    if candidates:
        best = max(candidates, key=lambda element: len(element.get_text(" ", strip=True)))
        best_text = best.get_text(" ", strip=True)
        if len(best_text) >= _MIN_CONTENT_CHARS:
            parent = best.parent
            if (
                best.name == "section"
                and parent is not None
                and parent.name != "body"
                and len(parent.get_text(" ", strip=True)) > len(best_text)
            ):
                return parent
            return best

    return soup.body or soup


def _html_blocks(path: Path) -> tuple[str, list[tuple[str | None, str]]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    doc_name = _document_name(soup, path)

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    root = _content_root(soup)

    sections: list[list] = []
    current: list | None = None

    for element in root.find_all([*_HEADING_TAGS, *_TEXT_TAGS]):
        if element.name in _TEXT_TAGS and element.find(_TEXT_TAGS):
            continue

        text = _clean(element.get_text(" ", strip=True))
        if not text:
            continue

        if element.name in _HEADING_TAGS:
            current = [text, []]
            sections.append(current)
        else:
            if current is None:
                current = [None, []]
                sections.append(current)
            current[1].append(text)

    return doc_name, [(heading, "\n".join(body)) for heading, body in sections if body]


def _pdf_blocks(path: Path) -> tuple[str, list[tuple[int, str]]]:
    reader = PdfReader(str(path))
    doc_name = _clean(reader.metadata.title) if reader.metadata and reader.metadata.title else path.stem

    blocks: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            blocks.append((number, text))

    return doc_name, blocks


def load_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc_name, blocks = _pdf_blocks(path)
        entries = [(doc_name, "page", number, text) for number, text in blocks]
    elif suffix in (".html", ".htm"):
        doc_name, blocks = _html_blocks(path)
        entries = [(doc_name, "section", heading or doc_name, text) for heading, text in blocks]
    else:
        raise ValueError(f"Unsupported file type: {path}")

    splitter = _splitter()
    chunks: list[Chunk] = []

    for doc_name, locator_type, locator_value, text in entries:
        if locator_type == "section" and locator_value == doc_name:
            locator_type = "document"

        if locator_type == "page":
            label = f"Halaman {locator_value}"
        elif locator_type == "document":
            label = ""
        else:
            label = str(locator_value)

        title = f"{doc_name} — {label}" if label else doc_name

        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc_name,
                    locator_type=locator_type,
                    locator_value=locator_value,
                    title=title,
                )
            )

    return chunks


def load_documents(raw_docs_dir: str | Path = config.RAW_DOCS_DIR) -> list[Chunk]:
    directory = Path(raw_docs_dir)
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_SUFFIXES
    )

    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(load_file(path))

    return chunks
