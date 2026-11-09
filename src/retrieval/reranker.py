"""Hybrid Reranker combining dense vector similarity, lexical exact-match density, and section heuristics."""

import logging
import re
from typing import List, Tuple
from pydantic import BaseModel

from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


class ScoredChunk(BaseModel):
    chunk: TextChunk
    vector_score: float
    lexical_score: float
    final_score: float


class HybridReranker:
    """Reranks candidate chunks by combining vector score with exact keyword density and paper disambiguation."""

    # Keywords associated with specific papers to resolve ambiguous cross-paper vocabulary
    PAPER_DISAMBIGUATION_KEYWORDS = {
        "attention": {
            "transformer", "vaswani", "wmt", "bleu", "multi-head", "scaled dot-product",
            "self-attention", "sinusoidal", "encoder-decoder", "bpe"
        },
        "moshi": {
            "moshi", "mimi", "helium", "audio", "speech", "dialogue", "duplex",
            "codec", "rq-vae", "streaming", "latency", "24khz", "12.5hz", "kyutai"
        },
        "dynamic_chunking": {
            "dynamic chunking", "h-net", "hierarchical", "byte-level", "bytes",
            "dna", "cartesia", "gu", "subword", "chunking", "modality", "lossless"
        }
    }

    def __init__(
        self,
        vector_weight: float = 0.50,
        lexical_weight: float = 0.35,
        section_weight: float = 0.15
    ):
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.section_weight = section_weight

    def extract_keywords(self, query: str) -> List[str]:
        """Extract clean alphanumeric query tokens."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower())
        stopwords = {
            "what", "is", "the", "in", "and", "or", "of", "to", "a", "an", "how",
            "does", "which", "are", "were", "for", "with", "on", "by", "from", "at"
        }
        return [w for w in words if w not in stopwords]

    def compute_lexical_score(self, query_tokens: List[str], chunk_text: str) -> float:
        """Compute exact keyword match ratio with bonus for multi-word phrases."""
        if not query_tokens:
            return 0.0

        lower_chunk = chunk_text.lower()
        matched = 0
        for token in query_tokens:
            if token in lower_chunk:
                matched += 1

        base_ratio = matched / len(query_tokens)

        # Bonus if full query bigrams appear consecutively
        bigram_bonus = 0.0
        if len(query_tokens) >= 2:
            for i in range(len(query_tokens) - 1):
                bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                if bigram in lower_chunk:
                    bigram_bonus += 0.2
        return min(1.0, base_ratio + bigram_bonus)

    def compute_paper_affinity_boost(self, query: str, paper_id: str) -> float:
        """Disambiguate query intent across the 3 papers to avoid cross-paper bleed."""
        lower_query = query.lower()
        paper_keywords = self.PAPER_DISAMBIGUATION_KEYWORDS.get(paper_id, set())

        # Check if query specifically names the paper or key distinct concepts
        for kw in paper_keywords:
            if kw in lower_query:
                return 0.15  # Positive boost for matching paper context

        # Check if query names another paper's distinctive entity
        for other_pid, other_kws in self.PAPER_DISAMBIGUATION_KEYWORDS.items():
            if other_pid != paper_id:
                for kw in other_kws:
                    if kw in lower_query and len(kw) > 4:
                        return -0.15  # Penalty for cross-paper collision

        return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[TextChunk, float]],
        top_k: int = 4
    ) -> List[ScoredChunk]:
        """Rerank candidates using vector score, lexical density, and section/paper affinity."""
        if not candidates:
            return []

        query_tokens = self.extract_keywords(query)
        scored_chunks: List[ScoredChunk] = []

        # Find min and max vector scores for min-max normalization
        v_scores = [c[1] for c in candidates]
        min_v = min(v_scores)
        max_v = max(v_scores)
        v_range = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0

        for chunk, raw_vector_score in candidates:
            # Normalize vector score to [0, 1]
            norm_vector = (raw_vector_score - min_v) / v_range

            # Lexical exact match score
            lexical_score = self.compute_lexical_score(query_tokens, chunk.text)

            # Section heading bonus
            section_bonus = 0.0
            lower_section = chunk.section.lower()
            if any(term in lower_section for term in query_tokens):
                section_bonus = 0.2
            elif any(s in lower_section for s in ["model", "architecture", "results", "attention", "dynamic", "mimi"]):
                section_bonus = 0.1

            # Paper affinity boost
            affinity_boost = self.compute_paper_affinity_boost(query, chunk.paper_id)

            final_score = (
                self.vector_weight * norm_vector
                + self.lexical_weight * lexical_score
                + self.section_weight * section_bonus
                + affinity_boost
            )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=float(raw_vector_score),
                    lexical_score=float(lexical_score),
                    final_score=float(final_score),
                )
            )

        # Sort descending by final score
        scored_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return scored_chunks[:top_k]
