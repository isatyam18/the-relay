"""Chunking engine for extracted research papers preserving full provenance metadata."""

import logging
import re
from typing import Dict, List, Optional
from pydantic import BaseModel

from .extractor import ExtractedPaper

logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    version: str
    page: int
    section: str
    text: str
    token_count: int
    char_count: int


class DocumentChunker:
    """Configurable document chunker that maintains precise citation metadata."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        size_metric: str = "tokens"  # "tokens" or "words" or "chars"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.size_metric = size_metric

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (1 token ~= 4 chars or 0.75 words)."""
        words = len(text.split())
        return int(words * 1.3)

    def chunk_paper(self, paper: ExtractedPaper) -> List[TextChunk]:
        """Split an extracted paper into chunks preserving page and section references."""
        chunks: List[TextChunk] = []

        for page in paper.pages:
            current_section = page.active_section
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split into paragraphs/blocks first to preserve coherent blocks
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [page_text]

            # Build sliding chunks from paragraphs on this page
            current_words: List[str] = []
            chunk_idx = 0

            for para in paragraphs:
                # Update current section if paragraph looks like a heading
                first_line = para.split("\n")[0].strip()
                if len(first_line) < 60 and (
                    first_line.startswith(("1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 "))
                    or first_line.lower() in {"abstract", "introduction", "background", "results", "conclusion"}
                ):
                    current_section = first_line

                words = para.split()
                if not words:
                    continue

                if len(current_words) + len(words) <= self.chunk_size:
                    current_words.extend(words)
                else:
                    # Flush current chunk if non-empty
                    if current_words:
                        chunk_text = " ".join(current_words)
                        chunk_id = f"{paper.paper_id}_p{page.page_number}_c{chunk_idx}"
                        token_count = self.estimate_tokens(chunk_text)
                        chunks.append(
                            TextChunk(
                                chunk_id=chunk_id,
                                paper_id=paper.paper_id,
                                paper_title=paper.title,
                                version=paper.version,
                                page=page.page_number,
                                section=current_section,
                                text=chunk_text,
                                token_count=token_count,
                                char_count=len(chunk_text)
                            )
                        )
                        chunk_idx += 1

                        # Keep overlap words
                        overlap_words = current_words[-self.chunk_overlap :] if self.chunk_overlap > 0 else []
                        current_words = overlap_words + words
                    else:
                        current_words = words

            # Flush final chunk on page
            if current_words:
                chunk_text = " ".join(current_words)
                chunk_id = f"{paper.paper_id}_p{page.page_number}_c{chunk_idx}"
                token_count = self.estimate_tokens(chunk_text)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        version=paper.version,
                        page=page.page_number,
                        section=current_section,
                        text=chunk_text,
                        token_count=token_count,
                        char_count=len(chunk_text)
                    )
                )

        logger.info(
            "Created %d chunks for paper %s (chunk_size=%d, overlap=%d)",
            len(chunks), paper.paper_id, self.chunk_size, self.chunk_overlap
        )
        return chunks

    def chunk_all(self, papers: List[ExtractedPaper]) -> List[TextChunk]:
        """Chunk a list of extracted papers."""
        all_chunks: List[TextChunk] = []
        for p in papers:
            all_chunks.extend(self.chunk_paper(p))
        return all_chunks

