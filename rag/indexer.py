"""
RAG Indexer service for Smart RAG.

Handles document indexing using Docling for parsing and ChromaDB for storage.
Supports scheduled indexing via APScheduler.
"""

import hashlib
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported file extensions (Docling capabilities)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".asciidoc",
    ".adoc",
    # Images (OCR)
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


class RAGIndexer:
    """
    Background service for indexing documents into ChromaDB.

    Uses Docling for document parsing and supports scheduled re-indexing.
    """

    def __init__(self):
        self._scheduler = None
        self._chroma_client = None
        self._running = False
        self._lock = threading.Lock()
        self._indexing_jobs: dict[int, threading.Thread] = {}

    def start(self):
        """Start the indexer service and scheduler."""
        if self._running:
            return

        self._running = True

        # Initialize ChromaDB client
        chroma_url = os.environ.get("CHROMA_URL")
        if chroma_url:
            try:
                import chromadb

                self._chroma_client = chromadb.HttpClient(
                    host=chroma_url.replace("http://", "")
                    .replace("https://", "")
                    .split(":")[0],
                    port=int(chroma_url.split(":")[-1])
                    if ":" in chroma_url.split("/")[-1]
                    else 8000,
                )
                logger.info(f"RAG Indexer connected to ChromaDB at {chroma_url}")
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")
                self._chroma_client = None
        else:
            logger.warning("CHROMA_URL not set - RAG indexing disabled")

        # Initialize scheduler
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()
            self._scheduler.start()
            logger.info("RAG Indexer scheduler started")

            # Schedule existing RAGs
            self._schedule_all_rags()
        except ImportError:
            logger.warning("APScheduler not installed - scheduled indexing disabled")
            self._scheduler = None

    def stop(self):
        """Stop the indexer service."""
        self._running = False

        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        # Wait for any running indexing jobs
        for thread in self._indexing_jobs.values():
            thread.join(timeout=5)

        logger.info("RAG Indexer stopped")

    def _schedule_all_rags(self):
        """Schedule indexing for all RAGs with schedules."""
        from db import get_rags_with_schedule

        rags = get_rags_with_schedule()
        for rag in rags:
            self.schedule_rag(rag.id, rag.index_schedule)

    def schedule_rag(self, rag_id: int, cron_expression: str):
        """
        Schedule periodic indexing for a RAG.

        Args:
            rag_id: ID of the SmartRAG
            cron_expression: Cron expression (e.g., "0 2 * * *" for 2 AM daily)
        """
        if not self._scheduler:
            logger.warning("Scheduler not available - cannot schedule RAG indexing")
            return

        job_id = f"rag_index_{rag_id}"

        # Remove existing job if any
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass

        if not cron_expression:
            return

        try:
            # Parse cron expression (minute hour day month day_of_week)
            parts = cron_expression.split()
            if len(parts) != 5:
                logger.error(f"Invalid cron expression: {cron_expression}")
                return

            from apscheduler.triggers.cron import CronTrigger

            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )

            self._scheduler.add_job(
                self._run_index_job,
                trigger,
                args=[rag_id],
                id=job_id,
                replace_existing=True,
            )
            logger.info(f"Scheduled indexing for RAG {rag_id}: {cron_expression}")
        except Exception as e:
            logger.error(f"Failed to schedule RAG {rag_id}: {e}")

    def unschedule_rag(self, rag_id: int):
        """Remove scheduled indexing for a RAG."""
        if not self._scheduler:
            return

        job_id = f"rag_index_{rag_id}"
        try:
            self._scheduler.remove_job(job_id)
            logger.info(f"Unscheduled indexing for RAG {rag_id}")
        except Exception:
            pass

    def _run_index_job(self, rag_id: int):
        """Run indexing job (called by scheduler)."""
        self.index_rag(rag_id, background=True)

    def cancel_indexing(self, rag_id: int) -> bool:
        """
        Cancel and reset a stuck indexing job.

        This doesn't actually kill the thread (Python threads can't be killed),
        but it resets the status so a new indexing job can be started.
        """
        from db import update_smart_rag_index_status

        with self._lock:
            if rag_id in self._indexing_jobs:
                del self._indexing_jobs[rag_id]

        # Reset status to pending
        update_smart_rag_index_status(rag_id, "pending")
        logger.info(f"Cancelled/reset indexing for RAG {rag_id}")
        return True

    def is_indexing(self, rag_id: int) -> bool:
        """Check if a RAG is currently being indexed."""
        with self._lock:
            if rag_id in self._indexing_jobs:
                return self._indexing_jobs[rag_id].is_alive()
        return False

    def index_rag(self, rag_id: int, background: bool = False) -> bool:
        """
        Index all documents for a SmartRAG.

        Args:
            rag_id: ID of the SmartRAG to index
            background: If True, run in background thread

        Returns:
            True if indexing started/completed successfully
        """
        if not self._chroma_client:
            logger.error("ChromaDB not available - cannot index")
            return False

        if background:
            # Check if already indexing
            with self._lock:
                if rag_id in self._indexing_jobs:
                    thread = self._indexing_jobs[rag_id]
                    if thread.is_alive():
                        logger.warning(f"RAG {rag_id} is already being indexed")
                        return False

                thread = threading.Thread(
                    target=self._index_rag_impl,
                    args=[rag_id],
                    daemon=True,
                )
                self._indexing_jobs[rag_id] = thread
                thread.start()
            return True
        else:
            return self._index_rag_impl(rag_id)

    def _index_rag_impl(self, rag_id: int) -> bool:
        """Implementation of RAG indexing."""
        from db import get_smart_rag_by_id, update_smart_rag_index_status

        from .retriever import _get_embedding_provider_for_rag

        # Get RAG config
        rag = get_smart_rag_by_id(rag_id)
        if not rag:
            logger.error(f"RAG {rag_id} not found")
            return False

        source_desc = rag.source_path if rag.source_type == "local" else f"MCP:{rag.mcp_server_config.get('name', 'unknown') if rag.mcp_server_config else 'unknown'}"
        logger.info(f"Starting indexing for RAG '{rag.name}' from {source_desc}")

        # Update status to indexing
        update_smart_rag_index_status(rag_id, "indexing")

        try:
            # Get document source based on source_type
            from mcp.sources import get_document_source

            source = get_document_source(
                source_type=rag.source_type,
                source_path=rag.source_path,
                mcp_config=rag.mcp_server_config,
            )

            if not source.is_available():
                raise ValueError(f"Document source not available: {source_desc}")

            # Get embedding provider
            embedding_provider = _get_embedding_provider_for_rag(
                rag.embedding_provider,
                rag.embedding_model,
                rag.ollama_url,
            )

            if not embedding_provider.is_available():
                raise ValueError(
                    f"Embedding provider '{rag.embedding_provider}' is not available"
                )

            # Get or create ChromaDB collection
            collection_name = rag.collection_name or f"smartrag_{rag_id}"
            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Clear existing documents
            try:
                collection.delete(where={})
            except Exception:
                # Collection might be empty
                pass

            # Find and process documents
            documents = []
            doc_count = 0

            # List all documents from the source
            all_docs = list(source.list_documents())
            total_files = len(all_docs)
            logger.info(f"Found {total_files} documents to index")

            for file_idx, doc_info in enumerate(all_docs):
                logger.info(
                    f"Processing document {file_idx + 1}/{total_files}: {doc_info.name}"
                )
                try:
                    # Process document based on source type
                    if rag.source_type == "local":
                        # For local files, use the existing file-based processing
                        chunks = self._process_document(
                            Path(doc_info.uri),
                            rag.chunk_size,
                            rag.chunk_overlap,
                            vision_provider=rag.vision_provider,
                            vision_model=rag.vision_model,
                            vision_ollama_url=rag.vision_ollama_url,
                        )
                    else:
                        # For MCP sources, read content and process
                        content = source.read_document(doc_info.uri)
                        if content:
                            chunks = self._process_content(
                                content,
                                rag.chunk_size,
                                rag.chunk_overlap,
                            )
                        else:
                            chunks = []

                    for chunk in chunks:
                        chunk["source_file"] = doc_info.name
                    documents.extend(chunks)
                    doc_count += 1
                    logger.info(
                        f"Completed {doc_info.name}: {len(chunks)} chunks "
                        f"(total: {len(documents)} chunks from {doc_count} docs)"
                    )

                    # Update progress after each document
                    update_smart_rag_index_status(
                        rag_id,
                        "indexing",
                        document_count=doc_count,
                        chunk_count=len(documents),
                    )
                except Exception as e:
                    logger.warning(f"Failed to process {doc_info.uri}: {e}")

            if not documents:
                logger.warning(f"No documents found in {source_desc}")
                update_smart_rag_index_status(
                    rag_id,
                    "ready",
                    document_count=0,
                    chunk_count=0,
                    collection_name=collection_name,
                )
                return True

            # Generate embeddings and store in ChromaDB
            logger.info(f"Generating embeddings for {len(documents)} chunks...")

            # Process in batches
            batch_size = 100
            total_chunks = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                texts = [doc["content"] for doc in batch]

                # Generate embeddings
                embed_result = embedding_provider.embed(texts)

                # Prepare for ChromaDB
                ids = [
                    hashlib.md5(
                        f"{doc['source_file']}:{doc['chunk_index']}".encode()
                    ).hexdigest()
                    for doc in batch
                ]
                metadatas = [
                    {
                        "source_file": doc["source_file"],
                        "chunk_index": doc["chunk_index"],
                        "indexed_at": datetime.utcnow().isoformat(),
                    }
                    for doc in batch
                ]

                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embed_result.embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

                total_chunks += len(batch)
                logger.debug(f"Indexed {total_chunks}/{len(documents)} chunks")

            # Update status
            update_smart_rag_index_status(
                rag_id,
                "ready",
                document_count=doc_count,
                chunk_count=total_chunks,
                collection_name=collection_name,
            )

            logger.info(
                f"Indexing complete for RAG '{rag.name}': "
                f"{doc_count} documents, {total_chunks} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Indexing failed for RAG {rag_id}: {e}")
            update_smart_rag_index_status(rag_id, "error", error=str(e))
            return False

        finally:
            # Clean up job reference
            with self._lock:
                self._indexing_jobs.pop(rag_id, None)

    def _find_documents(self, source_path: Path):
        """Find all supported documents in a directory."""
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path

    def _process_document(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        vision_provider: str = "local",
        vision_model: Optional[str] = None,
        vision_ollama_url: Optional[str] = None,
    ) -> list[dict]:
        """
        Process a document and return chunks.

        Uses Docling for parsing if available, falls back to simple text extraction.

        Args:
            file_path: Path to the document
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            vision_provider: Vision model provider ("local", "ollama:<instance>", or provider name)
            vision_model: Vision model name for remote processing
            vision_ollama_url: Ollama URL for vision processing
        """
        suffix = file_path.suffix.lower()

        # Try Docling first
        try:
            return self._process_with_docling(
                file_path,
                chunk_size,
                chunk_overlap,
                vision_provider=vision_provider,
                vision_model=vision_model,
                vision_ollama_url=vision_ollama_url,
            )
        except ImportError:
            logger.debug("Docling not available, using fallback parser")
        except Exception as e:
            logger.warning(f"Docling failed for {file_path}: {e}, trying fallback")

        # Fallback to simple parsing
        return self._process_simple(file_path, suffix, chunk_size, chunk_overlap)

    def _process_with_docling(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        vision_provider: str = "local",
        vision_model: Optional[str] = None,
        vision_ollama_url: Optional[str] = None,
    ) -> list[dict]:
        """Process document using Docling with configurable vision model."""
        from .vision import get_document_converter

        converter = get_document_converter(
            vision_provider=vision_provider,
            vision_model=vision_model,
            vision_ollama_url=vision_ollama_url,
        )
        result = converter.convert(str(file_path))

        # Get text content
        text = result.document.export_to_markdown()

        # Chunk the text
        return self._chunk_text(text, chunk_size, chunk_overlap)

    def _process_simple(
        self, file_path: Path, suffix: str, chunk_size: int, chunk_overlap: int
    ) -> list[dict]:
        """Simple fallback document processing."""
        text = ""

        if suffix in {".txt", ".md", ".asciidoc", ".adoc"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        elif suffix in {".html", ".htm"}:
            try:
                from bs4 import BeautifulSoup

                html = file_path.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
            except ImportError:
                # Just strip tags crudely
                import re

                html = file_path.read_text(encoding="utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", "", html)

        elif suffix == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise ValueError("pypdf required for PDF processing without Docling")

        elif suffix == ".docx":
            try:
                from docx import Document

                doc = Document(str(file_path))
                text = "\n".join(para.text for para in doc.paragraphs)
            except ImportError:
                raise ValueError(
                    "python-docx required for DOCX processing without Docling"
                )

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if not text.strip():
            return []

        return self._chunk_text(text, chunk_size, chunk_overlap)

    def _process_content(
        self,
        content: "DocumentContent",
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Process document content from an MCP source.

        Handles both text and binary content.

        Args:
            content: DocumentContent from MCP source
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk dicts with 'content' and 'chunk_index'
        """
        from mcp.sources import DocumentContent

        # If we have text content, chunk it directly
        if content.text:
            text = content.text

            # Handle HTML content
            if content.mime_type and "html" in content.mime_type.lower():
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(text, "html.parser")
                    text = soup.get_text(separator="\n")
                except ImportError:
                    import re
                    text = re.sub(r"<[^>]+>", "", text)

            return self._chunk_text(text, chunk_size, chunk_overlap)

        # If we have binary content, try to extract text
        if content.binary:
            mime = content.mime_type or ""

            # PDF
            if "pdf" in mime.lower():
                try:
                    import io
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(content.binary))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    return self._chunk_text(text, chunk_size, chunk_overlap)
                except Exception as e:
                    logger.warning(f"Failed to extract text from PDF: {e}")
                    return []

            # DOCX
            if "wordprocessingml" in mime.lower() or content.name.endswith(".docx"):
                try:
                    import io
                    from docx import Document
                    doc = Document(io.BytesIO(content.binary))
                    text = "\n".join(para.text for para in doc.paragraphs)
                    return self._chunk_text(text, chunk_size, chunk_overlap)
                except Exception as e:
                    logger.warning(f"Failed to extract text from DOCX: {e}")
                    return []

            # For images and other binary formats, we can't extract text
            # without vision models - skip for now
            logger.debug(f"Skipping binary content with mime type: {mime}")
            return []

        return []

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
        """
        Split text into overlapping chunks.

        Simple character-based chunking with sentence awareness.
        """
        if not text.strip():
            return []

        # Normalize whitespace
        text = " ".join(text.split())

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "content": chunk_text,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            # Move start with overlap
            start = end - chunk_overlap
            if start >= len(text) - chunk_overlap:
                break

        return chunks

    def delete_collection(self, rag_id: int) -> bool:
        """Delete the ChromaDB collection for a RAG."""
        if not self._chroma_client:
            return False

        from db import get_smart_rag_by_id

        rag = get_smart_rag_by_id(rag_id)
        if not rag or not rag.collection_name:
            return False

        try:
            self._chroma_client.delete_collection(rag.collection_name)
            logger.info(f"Deleted collection {rag.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_stats(self, rag_id: int) -> Optional[dict]:
        """Get statistics for a RAG's ChromaDB collection."""
        if not self._chroma_client:
            return None

        from db import get_smart_rag_by_id

        rag = get_smart_rag_by_id(rag_id)
        if not rag or not rag.collection_name:
            return None

        try:
            collection = self._chroma_client.get_collection(rag.collection_name)
            count = collection.count()
            return {
                "collection_name": rag.collection_name,
                "chunk_count": count,
            }
        except Exception:
            return None


# Global indexer instance
_indexer: Optional[RAGIndexer] = None


def get_indexer() -> RAGIndexer:
    """Get or create the global RAG indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = RAGIndexer()
        _indexer.start()
    return _indexer


def start_indexer():
    """Start the global RAG indexer."""
    indexer = get_indexer()
    indexer.start()


def stop_indexer():
    """Stop the global RAG indexer."""
    global _indexer
    if _indexer:
        _indexer.stop()
        _indexer = None
