"""Vector and lexical indexer for research paper chunks using FAISS and Scikit-Learn."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import settings
from src.ingestion.extractor import ExtractedPaper, PDFExtractor
from src.ingestion.downloader import PaperDownloader
from src.ingestion.chunker import DocumentChunker, TextChunk

logger = logging.getLogger(__name__)


class IndexedCorpus:
    """Encapsulates indexed chunks, vectorizer, and FAISS index for a specific chunking setting."""

    def __init__(
        self,
        setting_name: str,
        chunk_size: int,
        chunk_overlap: int,
        chunks: List[TextChunk],
        vectorizer: TfidfVectorizer,
        faiss_index: faiss.IndexFlatIP,
        embeddings: np.ndarray,
    ):
        self.setting_name = setting_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.faiss_index = faiss_index
        self.embeddings = embeddings  # Shape: (num_chunks, dim), L2-normalized float32

    def save(self, filepath: Path) -> None:
        """Serialize index and chunks to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Note: faiss index is serialized separately or converted to array
        data = {
            "setting_name": self.setting_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [c.model_dump() for c in self.chunks],
            "vectorizer": self.vectorizer,
            "embeddings": self.embeddings,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved index '%s' (%d chunks) to %s", self.setting_name, len(self.chunks), filepath)

    @classmethod
    def load(cls, filepath: Path) -> "IndexedCorpus":
        """Load serialized index from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        chunks = [TextChunk(**c) for c in data["chunks"]]
        embeddings = data["embeddings"].astype("float32")
        dim = embeddings.shape[1]

        # Reconstruct FAISS inner product index (cosine similarity since L2-normalized)
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)

        return cls(
            setting_name=data["setting_name"],
            chunk_size=data["chunk_size"],
            chunk_overlap=data["chunk_overlap"],
            chunks=chunks,
            vectorizer=data["vectorizer"],
            faiss_index=faiss_index,
            embeddings=embeddings,
        )


class PaperIndexer:
    """Builds and manages multi-setting indices for the research papers."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or settings.PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = PDFExtractor()
        self.downloader = PaperDownloader()

    def build_index(
        self,
        setting_name: str,
        chunk_size: int,
        chunk_overlap: int,
        extracted_papers: Optional[List[ExtractedPaper]] = None,
        force_rebuild: bool = False,
    ) -> IndexedCorpus:
        """Build or load an index for a given chunk size and overlap."""
        cache_path = self.processed_dir / f"index_{setting_name}.pkl"
        if cache_path.exists() and not force_rebuild:
            logger.info("Loading cached index for %s from %s", setting_name, cache_path)
            try:
                return IndexedCorpus.load(cache_path)
            except Exception as e:
                logger.warning("Failed to load cached index %s: %s. Rebuilding...", cache_path, e)

        if extracted_papers is None:
            # Download and extract all papers
            downloaded = self.downloader.download_all()
            extracted_papers = [
                self.extractor.extract_paper(path, settings.PAPERS[pid])
                for pid, path in downloaded.items()
            ]

        # Chunk all papers
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_all(extracted_papers)

        logger.info(
            "Building vector index for setting '%s' with %d chunks...",
            setting_name, len(chunks)
        )

        # Build TF-IDF with sublinear TF and n-grams for technical terminology
        corpus_texts = [f"{c.section} {c.text}" for c in chunks]
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=12000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_texts).toarray().astype("float32")

        # L2-normalize vectors for exact cosine similarity via FAISS inner product
        faiss.normalize_L2(tfidf_matrix)

        dim = tfidf_matrix.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(tfidf_matrix)

        indexed = IndexedCorpus(
            setting_name=setting_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunks=chunks,
            vectorizer=vectorizer,
            faiss_index=faiss_index,
            embeddings=tfidf_matrix,
        )

        indexed.save(cache_path)
        return indexed
