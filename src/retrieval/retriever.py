"""Configurable multi-setting paper retriever."""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss
from pydantic import BaseModel, Field

from config.settings import settings
from src.ingestion.chunker import TextChunk
from .indexer import IndexedCorpus, PaperIndexer
from .reranker import HybridReranker, ScoredChunk

logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    setting_name: str
    piece_size: int
    chunk_overlap: int
    top_k: int
    rerank: bool
    candidate_pool_size: int = 12

    @classmethod
    def get_setting_fine(cls) -> "RetrievalConfig":
        """Setting 1: Fine chunks (256 tokens), Top-8, no reranking."""
        return cls(
            setting_name="fine_k8_norank",
            piece_size=256,
            chunk_overlap=32,
            top_k=8,
            rerank=False,
            candidate_pool_size=8,
        )

    @classmethod
    def get_setting_medium(cls) -> "RetrievalConfig":
        """Setting 2: Medium chunks (512 tokens), Top-4, with hybrid reranking (Balanced)."""
        return cls(
            setting_name="medium_k4_rerank",
            piece_size=512,
            chunk_overlap=64,
            top_k=4,
            rerank=True,
            candidate_pool_size=12,
        )

    @classmethod
    def get_setting_coarse(cls) -> "RetrievalConfig":
        """Setting 3: Coarse chunks (1024 tokens), Top-2, no reranking."""
        return cls(
            setting_name="coarse_k2_norank",
            piece_size=1024,
            chunk_overlap=128,
            top_k=2,
            rerank=False,
            candidate_pool_size=2,
        )


class RetrievalResult(BaseModel):
    setting_name: str
    chunks: List[TextChunk]
    scores: List[float]
    retrieval_time_ms: float
    tokens_total: int


class PaperRetriever:
    """Manages retrieval across the 3 configured settings."""

    # Presets mapping
    STANDARD_SETTINGS: Dict[str, RetrievalConfig] = {
        "fine_k8_norank": RetrievalConfig.get_setting_fine(),
        "medium_k4_rerank": RetrievalConfig.get_setting_medium(),
        "coarse_k2_norank": RetrievalConfig.get_setting_coarse(),
    }

    def __init__(self, indexer: Optional[PaperIndexer] = None):
        self.indexer = indexer or PaperIndexer()
        self.reranker = HybridReranker()
        self._loaded_corpora: Dict[str, IndexedCorpus] = {}

    def get_or_load_corpus(self, config: RetrievalConfig) -> IndexedCorpus:
        """Load or build the corpus corresponding to a retrieval configuration."""
        if config.setting_name not in self._loaded_corpora:
            corpus = self.indexer.build_index(
                setting_name=config.setting_name,
                chunk_size=config.piece_size,
                chunk_overlap=config.chunk_overlap,
            )
            self._loaded_corpora[config.setting_name] = corpus
        return self._loaded_corpora[config.setting_name]

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        degraded: bool = False,
    ) -> RetrievalResult:
        """Execute passage retrieval for a query using the specified configuration.
        
        If degraded=True (e.g. rate limit approaching), reduces top_k to save tokens.
        """
        start_time = time.perf_counter()
        config = config or self.STANDARD_SETTINGS["medium_k4_rerank"]

        top_k = max(1, config.top_k // 2) if degraded else config.top_k
        pool_size = max(top_k, config.candidate_pool_size)

        corpus = self.get_or_load_corpus(config)

        # 1. Transform query using TF-IDF vectorizer
        query_vec = corpus.vectorizer.transform([query]).toarray().astype("float32")
        faiss.normalize_L2(query_vec)

        # 2. Search FAISS index
        scores_arr, indices_arr = corpus.faiss_index.search(query_vec, pool_size)
        raw_scores = scores_arr[0]
        raw_indices = indices_arr[0]

        candidates: List[Tuple[TextChunk, float]] = []
        for idx, score in zip(raw_indices, raw_scores):
            if idx >= 0 and idx < len(corpus.chunks):
                candidates.append((corpus.chunks[idx], float(score)))

        # 3. Optional Reranking
        if config.rerank:
            scored_candidates = self.reranker.rerank(query, candidates, top_k=top_k)
            final_chunks = [sc.chunk for sc in scored_candidates]
            final_scores = [sc.final_score for sc in scored_candidates]
        else:
            final_chunks = [c[0] for c in candidates[:top_k]]
            final_scores = [c[1] for c in candidates[:top_k]]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        total_tokens = sum(c.token_count for c in final_chunks)

        logger.info(
            "Retrieved %d chunks in %.2f ms (setting=%s, tokens=%d)",
            len(final_chunks), elapsed_ms, config.setting_name, total_tokens
        )

        return RetrievalResult(
            setting_name=config.setting_name,
            chunks=final_chunks,
            scores=final_scores,
            retrieval_time_ms=elapsed_ms,
            tokens_total=total_tokens,
        )
