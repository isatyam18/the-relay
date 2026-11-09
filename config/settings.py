"""Configuration settings for The Relay project."""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PaperSpec(BaseModel):
    paper_id: str
    arxiv_id: str
    version: str
    title: str
    filename: str
    url: str

    @property
    def full_arxiv_id(self) -> str:
        return f"{self.arxiv_id}{self.version}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PAPERS_DIR: Path = DATA_DIR / "papers"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    RESULTS_PATH: Path = PROJECT_ROOT / "results.csv"

    # Groq API Configuration
    GROQ_API_KEY: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Stand-in Provider Configuration
    STANDIN_PORT: int = 8000
    STANDIN_HOST: str = "127.0.0.1"
    STANDIN_BASE_URL: str = "http://127.0.0.1:8000/v1"

    # Paper Specifications: Exactly the 3 mandated arXiv papers & versions
    PAPERS: Dict[str, PaperSpec] = {
        "attention": PaperSpec(
            paper_id="attention",
            arxiv_id="1706.03762",
            version="v7",
            title="Attention Is All You Need",
            filename="1706.03762v7.pdf",
            url="https://arxiv.org/pdf/1706.03762v7.pdf"
        ),
        "moshi": PaperSpec(
            paper_id="moshi",
            arxiv_id="2410.00037",
            version="v1",
            title="Moshi: a speech-text foundation model for real-time dialogue",
            filename="2410.00037v1.pdf",
            url="https://arxiv.org/pdf/2410.00037v1.pdf"
        ),
        "dynamic_chunking": PaperSpec(
            paper_id="dynamic_chunking",
            arxiv_id="2507.07955",
            version="v2",
            title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
            filename="2507.07955v2.pdf",
            url="https://arxiv.org/pdf/2507.07955v2.pdf"
        )
    }

    # Default Retrieval & Scheduling limits
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    DEFAULT_TOP_K: int = 4
    DEFAULT_REQUEST_TIMEOUT_S: float = 30.0
    DEFAULT_QUEUE_MAX_SIZE: int = 100
    DEFAULT_MAX_CONCURRENCY: int = 4


settings = Settings()

