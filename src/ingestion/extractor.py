"""PDF text and section extractor using PyMuPDF."""

import logging
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
import pymupdf

from config.settings import PaperSpec

logger = logging.getLogger(__name__)

# Heuristic pattern for academic paper section headings
SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)|(?:[A-Z]\.\s+))?([A-Z][A-Za-z0-9\s,\-–:]{2,60})$"
)
KNOWN_STANDALONE_SECTIONS = {
    "abstract", "introduction", "background", "model architecture",
    "attention", "training", "results", "conclusion", "broader impact",
    "limitations", "references", "appendix", "acknowledgements"
}


class ExtractedBlock(BaseModel):
    block_id: int
    text: str
    is_heading: bool = False
    heading_title: Optional[str] = None


class ExtractedPage(BaseModel):
    page_number: int  # 1-indexed
    text: str
    blocks: List[ExtractedBlock] = Field(default_factory=list)
    active_section: str = "Preamble"


class ExtractedPaper(BaseModel):
    paper_id: str
    title: str
    version: str
    total_pages: int
    pages: List[ExtractedPage]


class PDFExtractor:
    """Extracts text, page boundaries, and section structure from research PDFs."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def extract_paper(self, pdf_path: Path, spec: PaperSpec) -> ExtractedPaper:
        """Extract all pages and sections from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        doc = pymupdf.open(pdf_path)
        pages: List[ExtractedPage] = []
        current_section = "Abstract"

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            raw_text = page.get_text()
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            extracted_blocks: List[ExtractedBlock] = []

            for b in blocks:
                # b[4] is text, b[6] is block type (0 is text, 1 is image)
                if len(b) > 6 and b[6] != 0:
                    continue
                block_text = b[4].strip()
                if not block_text:
                    continue

                is_heading = False
                heading_title = None

                first_line = block_text.split("\n")[0].strip()
                # Check for section headings
                cleaned_line = re.sub(r"\s+", " ", first_line)
                if len(cleaned_line) < 70:
                    lower_clean = cleaned_line.lower()
                    if lower_clean in KNOWN_STANDALONE_SECTIONS:
                        is_heading = True
                        heading_title = cleaned_line
                        current_section = cleaned_line
                    elif SECTION_PATTERN.match(cleaned_line) and not cleaned_line.endswith("."):
                        # Verify it's not just a regular sentence
                        is_heading = True
                        heading_title = cleaned_line
                        current_section = cleaned_line

                extracted_blocks.append(
                    ExtractedBlock(
                        block_id=len(extracted_blocks),
                        text=block_text,
                        is_heading=is_heading,
                        heading_title=heading_title
                    )
                )

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=raw_text,
                    blocks=extracted_blocks,
                    active_section=current_section
                )
            )

        doc.close()
        logger.info(
            "Extracted %d pages from %s (%s)",
            len(pages), spec.paper_id, pdf_path.name
        )

        return ExtractedPaper(
            paper_id=spec.paper_id,
            title=spec.title,
            version=spec.version,
            total_pages=len(pages),
            pages=pages
        )

