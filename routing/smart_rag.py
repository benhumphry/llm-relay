"""
Smart RAG Engine for document-based context augmentation.

Retrieves relevant document chunks from ChromaDB and injects them into the system prompt.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from db.models import SmartRAG
    from providers.registry import ProviderRegistry, ResolvedModel

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result of a RAG retrieval and augmentation."""

    resolved: "ResolvedModel"
    rag_id: int  # For stats updates
    rag_name: str
    rag_tags: list[str]

    # Augmented content
    augmented_system: str | None = None  # Modified system prompt with injected context
    augmented_messages: list[dict] = field(default_factory=list)  # Usually unchanged

    # RAG metadata
    chunks_retrieved: int = 0
    sources: list[str] = field(default_factory=list)  # Source files used
    context_injected: bool = False

    # Embedding usage tracking
    embedding_usage: dict | None = None  # {"prompt_tokens": N, "total_tokens": N}
    embedding_model: str | None = None
    embedding_provider: str | None = None


class SmartRAGEngine:
    """
    Engine for document-based context augmentation using RAG.

    The engine:
    1. Extracts the user's query from messages
    2. Retrieves relevant document chunks from ChromaDB
    3. Injects the context into the system prompt
    4. Returns the augmented request for forwarding to the target model
    """

    def __init__(self, rag: "SmartRAG", registry: "ProviderRegistry"):
        """
        Initialize the RAG engine.

        Args:
            rag: SmartRAG configuration
            registry: Provider registry for model resolution
        """
        self.rag = rag
        self.registry = registry

    def augment(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> RAGResult:
        """
        Augment a request with document context.

        Args:
            messages: List of message dicts
            system: Optional system prompt

        Returns:
            RAGResult with augmented system/messages and metadata
        """
        from providers.registry import ResolvedModel

        # Resolve target model first
        try:
            target_resolved = self.registry._resolve_actual_model(self.rag.target_model)
            resolved = ResolvedModel(
                provider=target_resolved.provider,
                model_id=target_resolved.model_id,
                alias_name=self.rag.name,
                alias_tags=self.rag.tags,
            )
        except ValueError as e:
            logger.error(f"RAG '{self.rag.name}' target model not available: {e}")
            raise

        # Check if RAG is ready
        if self.rag.index_status != "ready":
            logger.warning(
                f"RAG '{self.rag.name}' is not ready (status: {self.rag.index_status})"
            )
            return RAGResult(
                resolved=resolved,
                rag_id=self.rag.id,
                rag_name=self.rag.name,
                rag_tags=self.rag.tags,
                augmented_system=system,
                augmented_messages=messages,
                context_injected=False,
            )

        # Get the user's query
        query = self._get_query(messages)
        if not query:
            logger.debug(f"RAG '{self.rag.name}': No query found in messages")
            return RAGResult(
                resolved=resolved,
                rag_id=self.rag.id,
                rag_name=self.rag.name,
                rag_tags=self.rag.tags,
                augmented_system=system,
                augmented_messages=messages,
                context_injected=False,
            )

        # Retrieve relevant chunks
        try:
            from rag import get_retriever

            retriever = get_retriever()
            result = retriever.retrieve(
                collection_name=self.rag.collection_name,
                query=query,
                embedding_provider=self.rag.embedding_provider,
                embedding_model=self.rag.embedding_model,
                ollama_url=self.rag.ollama_url,
                max_results=self.rag.max_results,
                similarity_threshold=self.rag.similarity_threshold,
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return RAGResult(
                resolved=resolved,
                rag_id=self.rag.id,
                rag_name=self.rag.name,
                rag_tags=self.rag.tags,
                augmented_system=system,
                augmented_messages=messages,
                context_injected=False,
            )

        # If no relevant chunks found, return unchanged but include embedding usage
        if not result.chunks:
            logger.debug(f"RAG '{self.rag.name}': No relevant chunks found")
            return RAGResult(
                resolved=resolved,
                rag_id=self.rag.id,
                rag_name=self.rag.name,
                rag_tags=self.rag.tags,
                augmented_system=system,
                augmented_messages=messages,
                context_injected=False,
                embedding_usage=result.embedding_usage,
                embedding_model=result.embedding_model,
                embedding_provider=result.embedding_provider,
            )

        # Format context
        from rag import get_retriever

        retriever = get_retriever()
        context = retriever.format_context(
            result,
            max_tokens=self.rag.max_context_tokens,
            include_sources=True,
        )

        # Get unique sources
        sources = list(set(chunk.source_file for chunk in result.chunks))

        # Inject context into system prompt
        augmented_system = self._inject_context(system, context)

        logger.info(
            f"RAG '{self.rag.name}': Injected {len(result.chunks)} chunks from {len(sources)} sources"
        )

        return RAGResult(
            resolved=resolved,
            rag_id=self.rag.id,
            rag_name=self.rag.name,
            rag_tags=self.rag.tags,
            augmented_system=augmented_system,
            augmented_messages=messages,
            chunks_retrieved=len(result.chunks),
            sources=sources,
            context_injected=True,
            embedding_usage=result.embedding_usage,
            embedding_model=result.embedding_model,
            embedding_provider=result.embedding_provider,
        )

    def _get_query(self, messages: list[dict], max_chars: int = 2000) -> str:
        """Extract the user's query from messages."""
        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:max_chars]
                elif isinstance(content, list):
                    # Extract text from content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    return " ".join(texts)[:max_chars]
        return ""

    def _inject_context(self, original_system: str | None, context: str) -> str:
        """Inject document context into the system prompt."""
        if not context.strip():
            return original_system or ""

        context_block = f"""
<document_context>
The following information was retrieved from the document collection to help answer the user's question.
Use this information to provide an accurate response. Cite the source files when relevant.

{context.strip()}
</document_context>
"""

        if original_system:
            return original_system + "\n\n" + context_block
        else:
            return context_block.strip()
