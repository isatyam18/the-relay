"""Tests for Paper Ingestion, Verification, Extraction, and Chunking."""

import pytest
from pathlib import Path

from config.settings import settings
from src.ingestion.downloader import PaperDownloader
from src.ingestion.extractor import PDFExtractor, ExtractedPaper
from src.ingestion.chunker import DocumentChunker, TextChunk


@pytest.fixture(scope="session")
def downloader(tmp_path_factory):
    # Use real papers directory so downloaded papers are cached
    return PaperDownloader(target_dir=settings.PAPERS_DIR)


@pytest.fixture(scope="session")
def downloaded_papers(downloader):
    """Download all 3 mandatory papers."""
    return downloader.download_all(force=False)


def test_download_and_verify_all_papers(downloaded_papers):
    """Ensure all 3 required papers are downloaded and exist on disk."""
    assert "attention" in downloaded_papers
    assert "moshi" in downloaded_papers
    assert "dynamic_chunking" in downloaded_papers

    for paper_id, path in downloaded_papers.items():
        assert path.exists()
        assert path.stat().st_size > 100_000  # Each PDF should be > 100KB


def test_pdf_extraction_page_counts(downloaded_papers):
    """Verify PyMuPDF extracts text, total pages, and sections properly."""
    extractor = PDFExtractor()

    # Test Attention paper
    attention_spec = settings.PAPERS["attention"]
    attention_extracted = extractor.extract_paper(downloaded_papers["attention"], attention_spec)

    assert isinstance(attention_extracted, ExtractedPaper)
    assert attention_extracted.total_pages >= 11  # Attention Is All You Need is ~11-15 pages
    assert len(attention_extracted.pages) == attention_extracted.total_pages

    # Check page 1 has non-empty text and mentions Transformer / Attention
    p1 = attention_extracted.pages[0]
    assert p1.page_number == 1
    assert "attention" in p1.text.lower()
    assert "transformer" in p1.text.lower()


def test_pdf_extraction_dynamic_chunking(downloaded_papers):
    """Verify Dynamic Chunking paper extraction and version metadata."""
    extractor = PDFExtractor()
    spec = settings.PAPERS["dynamic_chunking"]
    extracted = extractor.extract_paper(downloaded_papers["dynamic_chunking"], spec)

    assert extracted.total_pages > 5
    first_page_text = extracted.pages[0].text.lower()
    assert "dynamic chunking" in first_page_text or "h-net" in first_page_text


def test_chunker_provenance_and_integrity(downloaded_papers):
    """Verify DocumentChunker preserves exact page, section, and paper provenance."""
    extractor = PDFExtractor()
    spec = settings.PAPERS["attention"]
    extracted = extractor.extract_paper(downloaded_papers["attention"], spec)

    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    chunks = chunker.chunk_paper(extracted)

    assert len(chunks) > 10
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.paper_id == "attention"
        assert chunk.paper_title == spec.title
        assert chunk.version == "v7"
        assert chunk.page >= 1
        assert len(chunk.text) > 0
        assert chunk.token_count > 0
        assert chunk.char_count == len(chunk.text)
        assert chunk.chunk_id.startswith("attention_p")

