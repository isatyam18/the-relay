# The Relay: High-Reliability Research Assistant & Benchmark

A production-inspired, multi-step RAG-based research assistant and load/rate-limit evaluation harness grounded in three specific machine learning foundation papers.

Built with an asynchronous pipeline (`asyncio`), dual model providers (live **Groq API** and a calibrated **FastAPI stand-in service**), bounded admission queues, exponential backoff, and verifiable citation verification.

---

## 1. Grounded Library (Exact arXiv Papers)

The assistant answers questions exclusively from these three verified papers. If a question cannot be resolved using evidence from these papers, the system explicitly refuses to answer rather than hallucinating facts or citations.

| Paper ID | Title | arXiv Identifier | Version | Total Pages | Primary Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `attention` | *Attention Is All You Need* | `arXiv:1706.03762` | **v7** | 15 | Transformer Architecture, Self-Attention |
| `moshi` | *Moshi: a speech-text foundation model for real-time dialogue* | `arXiv:2410.00037` | **v1** | 22 | Duplex Speech-Text Dialogue, Mimi Neural Codec |
| `dynamic_chunking` | *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* | `arXiv:2507.07955` | **v2** | 24 | H-Net Architecture, Byte-level Dynamic Chunking |

---

## 2. Core Architecture & Pipeline Stages

```
                                [User / Evaluation Harness]
                                             |
                                             v
                             +-------------------------------+
                             |  Scheduler & Traffic Control  |
                             |  - Bounded In-Memory Queue    |
                             |  - Concurrency Semaphore      |
                             |  - Sliding-Window RPM/TPM     |
                             |  - Request Deadline Tracker   |
                             +---------------+---------------+
                                             |
                                             v
   +------------------------------------------------------------------------------------+
   |                                 5-Stage Pipeline                                   |
   |                                                                                    |
   |  Stage 1: Intent & Scope Classification                                            |
   |     |                                                                              |
   |     v                                                                              |
   |  Stage 2: Passage Retrieval (FAISS Inner Product + Chunk Provenance)               |
   |     |                                                                              |
   |     v                                                                              |
   |  Stage 3: Planning & Tool Execution (Deterministic Section / Table Inspector)      |
   |     |                                                                              |
   |     v                                                                              |
   |  Stage 4: Answer Generation with Verifiable Citations [Paper, Page X, Section Y]   |
   |     |                                                                              |
   |     v                                                                              |
   |  Stage 5: Citation & Factual Verification                                          |
   |     |-- (Claims Verified)  --> Return Verified Answer                              |
   |     `-- (Claims Missing)   --> Trigger 1 Refinement Loop or Explicit Refusal       |
   +-----------------------------------------+------------------------------------------+
                                             |
                                             v
                             +-------------------------------+
                             |      BaseModelProvider        |
                             +---------------+---------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
             [GroqProvider]                                  [StandinProvider]
            (Real Hosted API)                           (Calibrated FastAPI Server)
       - Live header tracking (RPM/TPM)            - Calibrated Log-Normal TTFT
       - Real rate-limit responses                 - Concurrency-dependent latency lag
       - Baseline probe telemetry                  - Controlled failure & 429 injection
```

---

## 3. Retrieval Settings & Trade-Off Analysis

To analyze the trade-off between retrieval cost (prompt tokens) and answer accuracy, the system provides three configurable retrieval settings:

| Setting Key | Chunk Size (Tokens) | Overlap (Tokens) | Fetched Pieces ($K$) | Reranker Enabled | Engineering Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fine_k8_norank` | 256 | 32 | 8 | No | High-granularity localized facts; tests whether fragmented chunks harm multi-paragraph synthesis. |
| `medium_k4_rerank` | 512 | 64 | 4 | **Yes** (Hybrid Reranker) | **Primary balanced setting**. Uses dense FAISS inner-product + exact keyword density + section heuristics + paper disambiguation. |
| `coarse_k2_norank` | 1024 | 128 | 2 | No | Broad section context; tests whether high token counts dilute specific technical facts. |

### How this library misleads a naive retriever:
1. **Cross-Paper Keyword Collision**: Terms like *"Transformer"*, *"attention"*, and *"encoder-decoder"* appear across all three papers (original Transformer in Vaswani et al., Helium Transformer in Moshi, and baseline comparisons in Dynamic Chunking). Naive dense search bleeds across papers unless guided by paper disambiguation.
2. **Polysemous "Chunking" Vocabulary**: *"Dynamic chunking"* refers to learned hierarchical byte segmentation in H-Net, while Moshi discusses streaming *audio frame chunking* (80ms frames, 12.5Hz), and Attention discusses *subword BPE tokenization*. Naive vector search conflates audio frame chunking with byte chunking.
3. **Prior Art vs. Proposed Architecture**: Introductions and background sections frequently cite prior codecs (SoundStream, EnCodec) or older sequence models (Mamba, RNNs). Vector similarity often scores general introduction chunks higher than the technical methodology sections.

---

## 4. Implementation Roadmap & Progress Tracker

- [x] **Milestone 1: Paper Ingestion & Version Verification**
  - Downloaded arXiv:1706.03762v7, arXiv:2410.00037v1, arXiv:2507.07955v2.
  - PyMuPDF extraction preserving 1-indexed pages and academic section headings.
  - Chunking engine preserving `{chunk_id, paper_id, paper_title, version, page, section, text}`.
  - Automated unit test suite (`tests/test_ingestion.py`).
- [x] **Milestone 2: Multi-Setting Retrieval Engine & Disambiguating Reranker**
  - FAISS inner-product vector indexing with sublinear TF-IDF n-grams.
  - Three distinct retrieval configurations (`fine_k8_norank`, `medium_k4_rerank`, `coarse_k2_norank`).
  - Hybrid Reranker resolving cross-paper entity collisions.
  - Graceful degradation mode (`degraded=True`) halving chunk counts under rate pressure.
  - Automated unit test suite (`tests/test_retrieval.py`).
- [x] **Milestone 3: Model Provider Layer & Real Groq Calibration Probe**
  - Unified `BaseModelProvider` asynchronous contract (`generate`, `stream_tokens`, `estimate_tokens`).
  - `GroqProvider` capturing live HTTP headers (`x-ratelimit-*`, `retry-after`), TTFT (`first_token_ms`), and total latency (`total_ms`).
  - Empirical calibration probe engine (`src/bench/groq_calibration_probe.py`) generating empirical profiles and baseline statistics for Table 1.
  - Automated test suite (`tests/test_providers.py`).
- [ ] **Milestone 4: Calibrated FastAPI Stand-in Service**
  - Simulating Groq `/v1/chat/completions` (REST & SSE streaming).
  - Concurrency latency curve and failure injection.
- [ ] **Milestone 5: 5-Stage Agentic Pipeline with Tool Calling & Loops**
  - Request classifier, retrieval, table/section inspector tool, generator, and verification loop.
- [ ] **Milestone 6: Scheduler, Reliability & Traffic Control**
  - Admission control, concurrency semaphore, deadline propagation, retry storms defense.
- [ ] **Milestone 7: Benchmark Scenarios & `results.csv` Generation**
  - 15+ curated questions, 5 load shapes (steady, burst, overload, halved limits, daily exhaustion).
- [ ] **Milestone 8: Results Compilation & 2-Page Final Report**
  - Tables 1, 2a, 2b, 3 and markdown report.

---

## 5. Setup & Execution

### Prerequisites
- Python 3.11+ (Tested on Python 3.14 on Windows)
- Recommended RAM: 8 GB

### Installation
```powershell
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
```

### Running Tests
```powershell
# Run Ingestion tests
python -m pytest tests/test_ingestion.py -v

# Run Retrieval tests
python -m pytest tests/test_retrieval.py -v
```
