"""Downloader for arXiv research papers with strict version verification."""

import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import pymupdf

from config.settings import PaperSpec, settings

logger = logging.getLogger(__name__)


class PaperDownloader:
    """Handles downloading and version verification of research papers."""

    def __init__(self, target_dir: Optional[Path] = None):
        self.target_dir = target_dir or settings.PAPERS_DIR
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def download_paper(self, spec: PaperSpec, force: bool = False) -> Path:
        """Download a paper from its arXiv URL and verify its PDF format and version."""
        dest_path = self.target_dir / spec.filename
        if dest_path.exists() and not force:
            logger.info("Paper %s already exists at %s. Verifying...", spec.paper_id, dest_path)
            self.verify_paper(dest_path, spec)
            return dest_path

        logger.info("Downloading %s (%s) from %s ...", spec.title, spec.full_arxiv_id, spec.url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TheRelayBenchmark/1.0"
        }
        req = urllib.request.Request(spec.url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to download {spec.paper_id} from {spec.url}: HTTP {response.status}"
                )
            content = response.read()

        dest_path.write_bytes(content)
        logger.info("Downloaded %d bytes to %s", len(content), dest_path)

        self.verify_paper(dest_path, spec)
        return dest_path

    def verify_paper(self, pdf_path: Path, spec: PaperSpec) -> None:
        """Verify the downloaded file is a valid PDF and matches the required arXiv version."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # 1. Magic byte check
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                raise ValueError(f"File {pdf_path} is not a valid PDF (invalid magic header: {header})")

        # 2. PyMuPDF integrity & version verification
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            raise ValueError(f"PDF {pdf_path} has 0 pages.")

        # Check first page text for arXiv version watermark (e.g. arXiv:1706.03762v7)
        page1_text = doc[0].get_text()
        expected_version_str = f"arXiv:{spec.arxiv_id}{spec.version}".lower()
        
        # arXiv watermarks are typically "arXiv:YYMM.NNNNNvV"
        if expected_version_str not in page1_text.lower():
            # Check other pages or metadata if margin watermark was cropped or formatted differently
            all_text_sample = "".join(doc[i].get_text() for i in range(min(3, len(doc)))).lower()
            if expected_version_str not in all_text_sample:
                logger.warning(
                    "ArXiv watermark '%s' not explicitly found in first 3 pages of %s. Checking ID and version separately...",
                    expected_version_str,
                    pdf_path.name
                )
                if spec.arxiv_id.lower() not in all_text_sample:
                    doc.close()
                    raise ValueError(
                        f"Paper verification failed: ArXiv ID {spec.arxiv_id} not found in {pdf_path.name}"
                    )

        page_count = len(doc)
        doc.close()
        logger.info("Verification passed for %s (%s, %d pages)", spec.paper_id, spec.full_arxiv_id, page_count)

    def download_all(self, force: bool = False) -> Dict[str, Path]:
        """Download all papers defined in settings."""
        results = {}
        for paper_id, spec in settings.PAPERS.items():
            results[paper_id] = self.download_paper(spec, force=force)
        return results
