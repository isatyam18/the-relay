"""Unit and integration tests for Multi-Setting Retrieval Engine and Reranker."""

import pytest
from src.retrieval.retriever import PaperRetriever, RetrievalConfig
from src.retrieval.indexer import PaperIndexer


@pytest.fixture(scope="session")
def retriever():
    indexer = PaperIndexer()
    return PaperRetriever(indexer=indexer)


def test_build_and_retrieve_setting_fine(retriever):
    """Test Setting 1: Fine chunks (256 tokens), Top 8, no reranking."""
    config = RetrievalConfig.get_setting_fine()
    query = "What is the formula for Scaled Dot-Product Attention and what is d_k?"
    result = retriever.retrieve(query, config=config)

    assert result.setting_name == "fine_k8_norank"
    assert len(result.chunks) == 8
    assert result.retrieval_time_ms > 0
    # At least one chunk must be from Attention paper
    attention_chunks = [c for c in result.chunks if c.paper_id == "attention"]
    assert len(attention_chunks) > 0
    # Every chunk must have valid provenance
    for c in result.chunks:
        assert c.page >= 1
        assert len(c.section) > 0
        assert len(c.text) > 0


def test_build_and_retrieve_setting_medium_rerank(retriever):
    """Test Setting 2: Medium chunks (512 tokens), Top 4, with Hybrid Reranking."""
    config = RetrievalConfig.get_setting_medium()
    query = "What is the Mimi neural audio codec and what frame rate does it operate at in Moshi?"
    result = retriever.retrieve(query, config=config)

    assert result.setting_name == "medium_k4_rerank"
    assert len(result.chunks) == 4
    # The top retrieved chunk should be from Moshi
    top_chunk = result.chunks[0]
    assert top_chunk.paper_id == "moshi"
    assert "mimi" in top_chunk.text.lower() or "codec" in top_chunk.text.lower()


def test_build_and_retrieve_setting_coarse(retriever):
    """Test Setting 3: Coarse chunks (1024 tokens), Top 2, no reranking."""
    config = RetrievalConfig.get_setting_coarse()
    query = "How does H-Net perform byte-level dynamic chunking without tokenizers on DNA sequences?"
    result = retriever.retrieve(query, config=config)

    assert result.setting_name == "coarse_k2_norank"
    assert len(result.chunks) == 2
    # Verify dynamic chunking paper chunk is returned
    dc_chunks = [c for c in result.chunks if c.paper_id == "dynamic_chunking"]
    assert len(dc_chunks) > 0
    assert any("h-net" in c.text.lower() or "byte" in c.text.lower() for c in dc_chunks)


def test_reranker_disambiguation(retriever):
    """Verify reranker successfully boosts relevant paper and penalizes false matches."""
    config_rerank = RetrievalConfig.get_setting_medium()
    config_norank = RetrievalConfig(
        setting_name="medium_k4_norank",
        piece_size=512,
        chunk_overlap=64,
        top_k=4,
        rerank=False,
    )

    query = "Moshi real-time conversational speech dialogue architecture"
    result_rerank = retriever.retrieve(query, config=config_rerank)
    result_norank = retriever.retrieve(query, config=config_norank)

    # Reranked result should have Moshi as the top 1
    assert result_rerank.chunks[0].paper_id == "moshi"
    # Provenance details
    assert result_rerank.chunks[0].page >= 1
    assert result_rerank.chunks[0].version == "v1"


def test_degraded_retrieval(retriever):
    """Verify degraded retrieval reduces chunk count to conserve quota under rate limit."""
    config = RetrievalConfig.get_setting_medium()
    query = "Transformer multi-head attention"
    normal_result = retriever.retrieve(query, config=config, degraded=False)
    degraded_result = retriever.retrieve(query, config=config, degraded=True)

    assert len(normal_result.chunks) == 4
    assert len(degraded_result.chunks) == 2
    assert degraded_result.tokens_total < normal_result.tokens_total
