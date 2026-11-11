"""Multi-format document loader with rich metadata extraction."""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class LoadedDocument:
    """Represents a loaded document or document page."""
    text: str
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic hash of source and content
            source = str(self.metadata.get("source", "unknown"))
            hash_input = f"{source}:{self.text[:200]}".encode("utf-8")
            self.doc_id = hashlib.sha256(hash_input).hexdigest()[:16]


class DocumentLoader:
    """Loads documents from files, directories, or raw text strings."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".json"}

    def load_file(self, file_path: Union[str, Path]) -> List[LoadedDocument]:
        """Loads a single file and returns a list of LoadedDocument items."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        suffix = path.suffix.lower()
        stat = path.stat()
        base_meta = {
            "source": path.name,
            "file_path": str(path.resolve()),
            "file_type": suffix,
            "file_size_bytes": stat.st_size,
        }

        try:
            if suffix in {".txt", ".md"}:
                return self._load_text(path, base_meta)
            elif suffix == ".pdf":
                return self._load_pdf(path, base_meta)
            elif suffix == ".docx":
                return self._load_docx(path, base_meta)
            elif suffix == ".json":
                return self._load_json(path, base_meta)
            else:
                logger.warning("Unsupported file extension %s; attempting text read", suffix)
                return self._load_text(path, base_meta)
        except Exception as e:
            logger.error("Failed to load file %s: %s", path, e)
            raise

    def load_directory(self, dir_path: Union[str, Path], recursive: bool = True) -> List[LoadedDocument]:
        """Loads all supported files from a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        pattern = "**/*" if recursive else "*"
        all_docs: List[LoadedDocument] = []

        for p in path.glob(pattern):
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load_file(p)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error("Error loading %s: %s", p, e)

        logger.info("Loaded %d documents from %s", len(all_docs), dir_path)
        return all_docs

    def load_text(self, text: str, source_name: str = "direct_input", extra_metadata: Optional[dict] = None) -> List[LoadedDocument]:
        """Loads a document directly from raw string text."""
        cleaned = text.strip()
        if not cleaned:
            return []
        meta = {
            "source": source_name,
            "file_type": "text/plain",
            "char_count": len(cleaned),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        return [LoadedDocument(text=cleaned, metadata=meta)]

    def _load_text(self, path: Path, base_meta: dict) -> List[LoadedDocument]:
        encodings = ["utf-8", "latin-1", "cp1252"]
        content = None
        for enc in encodings:
            try:
                content = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            raise UnicodeDecodeError(f"Unable to decode text file {path}")

        meta = dict(base_meta)
        meta["char_count"] = len(content)
        meta["page_number"] = 1
        return [LoadedDocument(text=content, metadata=meta)]

    def _load_pdf(self, path: Path, base_meta: dict) -> List[LoadedDocument]:
        documents: List[LoadedDocument] = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    meta = dict(base_meta)
                    meta["page_number"] = idx
                    meta["total_pages"] = total_pages
                    meta["char_count"] = len(page_text)
                    documents.append(LoadedDocument(text=page_text, metadata=meta))
        except ImportError:
            logger.warning("pypdf not available; reading as binary strings fallback")
            content = path.read_text(errors="ignore")
            meta = dict(base_meta)
            meta["page_number"] = 1
            documents.append(LoadedDocument(text=content, metadata=meta))
        return documents

    def _load_docx(self, path: Path, base_meta: dict) -> List[LoadedDocument]:
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            meta = dict(base_meta)
            meta["paragraph_count"] = len(paragraphs)
            meta["char_count"] = len(full_text)
            meta["page_number"] = 1
            return [LoadedDocument(text=full_text, metadata=meta)]
        except ImportError:
            content = path.read_text(errors="ignore")
            return [LoadedDocument(text=content, metadata=base_meta)]

    def _load_json(self, path: Path, base_meta: dict) -> List[LoadedDocument]:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        docs: List[LoadedDocument] = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                item_text = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                meta = dict(base_meta)
                meta["item_index"] = i
                docs.append(LoadedDocument(text=item_text, metadata=meta))
        elif isinstance(data, dict):
            # If standard document structure with 'text' or 'content'
            text = data.get("text") or data.get("content") or json.dumps(data, indent=2)
            meta = dict(base_meta)
            meta.update(data.get("metadata", {}))
            docs.append(LoadedDocument(text=str(text), metadata=meta))
        else:
            docs.append(LoadedDocument(text=str(data), metadata=base_meta))
        return docs

