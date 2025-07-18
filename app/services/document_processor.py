import os
import email
from pathlib import Path
from typing import List, Tuple
import logging

import PyPDF2
import docx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentProcessor:
    SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".eml"}

    # ---------- single-document helpers ----------
    def _extract_pdf(self, path: str) -> List[Tuple[str, int]]:
        pages = []
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append((text, idx + 1))
        except Exception as e:
            logger.error(f"PDF error {path}: {e}")
        return pages

    def _extract_docx(self, path: str) -> List[Tuple[str, int]]:
        try:
            doc = docx.Document(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [(text, 1)] if text else []
        except Exception as e:
            logger.error(f"DOCX error {path}: {e}")
            return []

    def _extract_txt(self, path: str) -> List[Tuple[str, int]]:
        try:
            with open(path, encoding="utf-8") as f:
                return [(f.read(), 1)]
        except Exception as e:
            logger.error(f"TXT error {path}: {e}")
            return []

    def _extract_html(self, path: str) -> List[Tuple[str, int]]:
        try:
            with open(path, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")
                return [(soup.get_text(" ", strip=True), 1)]
        except Exception as e:
            logger.error(f"HTML error {path}: {e}")
            return []

    def _extract_email(self, path: str) -> List[Tuple[str, int]]:
        try:
            with open(path, encoding="utf-8") as f:
                msg = email.message_from_file(f)
            subject = msg.get("Subject", "")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            full = f"Subject: {subject}\n\n{body}"
            return [(full, 1)]
        except Exception as e:
            logger.error(f"EML error {path}: {e}")
            return []

    # ---------- public ----------
    def process_document(self, path: str) -> List[Tuple[str, int]]:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext == ".docx":
            return self._extract_docx(path)
        elif ext == ".txt":
            return self._extract_txt(path)
        elif ext == ".html":
            return self._extract_html(path)
        elif ext == ".eml":
            return self._extract_email(path)
        else:
            raise ValueError(f"Unsupported format {ext}")

    def batch_process(self, paths: List[str]) -> List[Tuple[str, str, int]]:
        """Return list of (text, source_filename, page_number)"""
        all_docs = []
        for p in paths:
            pages = self.process_document(p)
            src = Path(p).name
            for txt, pg in pages:
                all_docs.append((txt, src, pg))
        return all_docs