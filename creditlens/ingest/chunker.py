"""Paragraph-aware chunking with overlap.

Chunk boundaries decide what retrieval can ever return, so they follow
document structure rather than a character count: paragraphs are packed
greedily up to a token target and never split mid-sentence unless a single
paragraph exceeds the budget on its own. Overlap carries the tail of the
previous chunk forward so a claim spanning a boundary is still retrievable.

Token counts are word-based estimates. `tiktoken` is deliberately not used -
it is OpenAI's tokenizer and undercounts Claude tokens materially; when an
API key is present, `count_tokens` gives exact numbers, and chunk sizing does
not need that precision.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_HEADING = re.compile(r"^[A-Z][A-Za-z0-9 ,'&/\-()]{3,80}$")

TOKENS_PER_WORD = 1.33


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


@dataclass
class TextChunk:
    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    item_code: str | None = None
    section_title: str | None = None
    heading_path: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:40]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "item_code": self.item_code,
            "section_title": self.section_title,
            "heading_path": self.heading_path,
            "tokens": self.token_count,
            "chars": len(self.text),
            "text": self.text,
        }


def _paragraphs(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    cursor = 0
    for block in text.split("\n\n"):
        stripped = block.strip()
        start = text.find(block, cursor)
        cursor = start + len(block) if start >= 0 else cursor
        if stripped:
            out.append((stripped, max(start, 0)))
    return out


def _split_long(paragraph: str, target: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(paragraph)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if size + tokens > target and current:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail(text: str, overlap_tokens: int) -> str:
    words = text.split()
    keep = int(overlap_tokens / TOKENS_PER_WORD)
    if keep <= 0 or keep >= len(words):
        return ""
    return " ".join(words[-keep:])


def chunk_text(
    text: str,
    *,
    target_tokens: int = 380,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
    item_code: str | None = None,
    section_title: str | None = None,
    start_ordinal: int = 0,
    base_offset: int = 0,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    buffer_start = base_offset
    heading: str | None = None
    ordinal = start_ordinal

    def flush(end_offset: int) -> None:
        nonlocal buffer, buffer_tokens, ordinal, buffer_start
        if not buffer:
            return
        body = "\n\n".join(buffer).strip()
        if estimate_tokens(body) >= min_tokens:
            chunks.append(TextChunk(
                ordinal=ordinal,
                text=body,
                token_count=estimate_tokens(body),
                char_start=buffer_start,
                char_end=end_offset,
                item_code=item_code,
                section_title=section_title,
                heading_path=" > ".join(
                    p for p in (section_title, heading) if p and p != section_title
                ) or section_title,
            ))
            ordinal += 1
        carry = _tail(body, overlap_tokens) if overlap_tokens else ""
        buffer = [carry] if carry else []
        buffer_tokens = estimate_tokens(carry) if carry else 0
        buffer_start = end_offset

    for paragraph, offset in _paragraphs(text):
        if _HEADING.match(paragraph) and len(paragraph) < 90:
            heading = paragraph
        pieces: Sequence[str] = (
            _split_long(paragraph, target_tokens)
            if estimate_tokens(paragraph) > target_tokens
            else [paragraph]
        )
        for piece in pieces:
            tokens = estimate_tokens(piece)
            if buffer_tokens + tokens > target_tokens and buffer:
                flush(base_offset + offset)
            if not buffer:
                buffer_start = base_offset + offset
            buffer.append(piece)
            buffer_tokens += tokens

    flush(base_offset + len(text))
    return chunks


def chunk_sections(
    sections: Iterable[Any],
    *,
    target_tokens: int = 380,
    overlap_tokens: int = 60,
) -> list[TextChunk]:
    """Chunk each parsed `Section`, keeping ordinals globally sequential."""
    chunks: list[TextChunk] = []
    for section in sections:
        chunks.extend(chunk_text(
            section.text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            item_code=getattr(section, "item_code", None),
            section_title=getattr(section, "title", None),
            start_ordinal=len(chunks),
            base_offset=getattr(section, "char_start", 0),
        ))
    return chunks
