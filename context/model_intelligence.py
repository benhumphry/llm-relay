"""
Model Intelligence for Smart Routers.

Gathers balanced, real-world model assessments via web search and caches
them in ChromaDB. Used to enhance Smart Router designator decisions with
comparative information about model strengths and weaknesses.

Usage:
    from context.model_intelligence import ModelIntelligence

    mi = ModelIntelligence()

    # Refresh intelligence for specific models
    mi.refresh_models(["anthropic/claude-sonnet-4", "openai/gpt-4o"])

    # Get cached intelligence for a model
    intel = mi.get_intelligence("anthropic/claude-sonnet-4")

    # Get intelligence for multiple models (for router prompt)
    intel_map = mi.get_intelligence_for_candidates([
        {"model": "anthropic/claude-sonnet-4"},
        {"model": "openai/gpt-4o"},
    ])
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Collection name for model intelligence
COLLECTION_NAME = "model_intelligence"

# Default TTL for cached intelligence (days)
DEFAULT_TTL_DAYS = 7

# Search queries to gather model information
# More specific queries yield better, more actionable results
SEARCH_QUERY_TEMPLATES = [
    "{model_name} strengths weaknesses use cases",
    "{model_name} benchmark comparison review 2024 2025 2026",
    "{model_name} best for what tasks limitations",
]


@dataclass
class ModelIntel:
    """Cached intelligence for a single model."""

    model_id: str  # e.g., "anthropic/claude-sonnet-4"
    intelligence: str  # Balanced assessment text
    strengths: list[str]
    weaknesses: list[str]
    best_for: list[str]
    avoid_for: list[str]
    sources: list[str]  # URLs used
    generated_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def to_prompt_text(self) -> str:
        """Format for inclusion in designator prompt."""
        return self.intelligence


class ModelIntelligence:
    """
    Service for gathering and caching model intelligence.

    Uses web search to gather recent model comparisons, benchmarks, and user
    feedback, then summarizes into balanced assessments cached in ChromaDB.
    """

    def __init__(
        self,
        summarizer_model: str | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ):
        """
        Initialize the Model Intelligence service.

        Args:
            summarizer_model: Model to use for summarizing search results (required for refresh operations)
            ttl_days: How long to cache intelligence before refreshing
        """
        self.summarizer_model = summarizer_model
        self.ttl_days = ttl_days
        self._collection = None

    @property
    def collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            from context import CollectionWrapper, is_chroma_available

            if not is_chroma_available():
                raise RuntimeError("ChromaDB is not available")
            self._collection = CollectionWrapper(COLLECTION_NAME)
        return self._collection

    def _model_to_id(self, model_id: str) -> str:
        """Convert model ID to a ChromaDB document ID."""
        return hashlib.md5(model_id.lower().encode()).hexdigest()

    def _model_to_search_name(self, model_id: str) -> str:
        """Convert model ID to a human-readable search name."""
        # "anthropic/claude-sonnet-4-20250514" -> "Claude Sonnet 4"
        # "openai/gpt-4o" -> "GPT-4o"
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Remove date suffixes
        import re

        name = re.sub(r"-\d{8}$", "", name)  # -20250514
        name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)  # -2025-01-01

        # Common replacements for readability
        replacements = {
            "claude-": "Claude ",
            "gpt-": "GPT-",
            "gemini-": "Gemini ",
            "llama-": "Llama ",
            "mistral-": "Mistral ",
            "deepseek-": "DeepSeek ",
            "-": " ",
        }
        for old, new in replacements.items():
            name = name.replace(old, new)

        return name.strip()

    def get_intelligence(self, model_id: str) -> Optional[ModelIntel]:
        """
        Get cached intelligence for a model.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4")

        Returns:
            ModelIntel object if found and not expired, None otherwise
        """
        try:
            doc_id = self._model_to_id(model_id)
            results = self.collection.get(ids=[doc_id])

            if not results or not results.get("ids"):
                return None

            metadata = results["metadatas"][0] if results.get("metadatas") else {}

            # Check expiry
            expires_at_str = metadata.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.utcnow() > expires_at:
                    logger.debug(f"Intelligence for {model_id} has expired")
                    return None

            return ModelIntel(
                model_id=model_id,
                intelligence=metadata.get("intelligence", ""),
                strengths=metadata.get("strengths", "").split("|")
                if metadata.get("strengths")
                else [],
                weaknesses=metadata.get("weaknesses", "").split("|")
                if metadata.get("weaknesses")
                else [],
                best_for=metadata.get("best_for", "").split("|")
                if metadata.get("best_for")
                else [],
                avoid_for=metadata.get("avoid_for", "").split("|")
                if metadata.get("avoid_for")
                else [],
                sources=metadata.get("sources", "").split("|")
                if metadata.get("sources")
                else [],
                generated_at=datetime.fromisoformat(metadata["generated_at"])
                if metadata.get("generated_at")
                else datetime.utcnow(),
                expires_at=expires_at
                if expires_at_str
                else datetime.utcnow() + timedelta(days=self.ttl_days),
            )

        except Exception as e:
            logger.warning(f"Error getting intelligence for {model_id}: {e}")
            return None

    def get_intelligence_for_candidates(
        self, candidates: list[dict]
    ) -> dict[str, ModelIntel]:
        """
        Get intelligence for multiple candidate models.

        Args:
            candidates: List of candidate dicts with "model" key

        Returns:
            Dict mapping model_id to ModelIntel (only includes models with intel)
        """
        result = {}
        for candidate in candidates:
            model_id = candidate.get("model", "")
            if model_id:
                intel = self.get_intelligence(model_id)
                if intel:
                    result[model_id] = intel
        return result

    def refresh_model(self, model_id: str, force: bool = False) -> Optional[ModelIntel]:
        """
        Refresh intelligence for a single model.

        Args:
            model_id: Model identifier
            force: If True, refresh even if not expired

        Returns:
            Updated ModelIntel or None if refresh failed

        Raises:
            ValueError: If no summarizer_model is configured
        """
        if not self.summarizer_model:
            raise ValueError("No summarizer model configured for Model Intelligence")

        # Check if we need to refresh
        if not force:
            existing = self.get_intelligence(model_id)
            if existing and not existing.is_expired():
                logger.debug(f"Intelligence for {model_id} is still valid")
                return existing

        logger.info(f"Refreshing intelligence for {model_id}")

        try:
            # 1. Search for model information
            search_results = self._search_for_model(model_id)
            if not search_results:
                logger.warning(f"No search results for {model_id}")
                return None

            # 2. Summarize into balanced assessment
            intel = self._summarize_results(model_id, search_results)
            if not intel:
                logger.warning(f"Failed to summarize results for {model_id}")
                return None

            # 3. Store in ChromaDB
            self._store_intelligence(intel)

            return intel

        except Exception as e:
            logger.error(f"Error refreshing intelligence for {model_id}: {e}")
            return None

    def refresh_models(
        self, model_ids: list[str], force: bool = False
    ) -> dict[str, ModelIntel]:
        """
        Refresh intelligence for multiple models.

        Args:
            model_ids: List of model identifiers
            force: If True, refresh all even if not expired

        Returns:
            Dict mapping model_id to updated ModelIntel
        """
        results = {}
        for model_id in model_ids:
            intel = self.refresh_model(model_id, force=force)
            if intel:
                results[model_id] = intel
        return results

    def refresh_models_comparative(
        self, model_ids: list[str], force: bool = False
    ) -> dict[str, ModelIntel]:
        """
        Refresh intelligence for multiple models with comparative analysis.

        Unlike refresh_models(), this gathers search results for all models
        and creates comparative assessments highlighting relative strengths
        and weaknesses between the specific models.

        Args:
            model_ids: List of model identifiers to compare
            force: If True, refresh all even if not expired

        Returns:
            Dict mapping model_id to updated ModelIntel with comparative assessments

        Raises:
            ValueError: If no summarizer_model is configured
        """
        if not self.summarizer_model:
            raise ValueError("No summarizer model configured for Model Intelligence")

        if len(model_ids) < 2:
            # Fall back to individual refresh for single model
            return self.refresh_models(model_ids, force=force)

        # Check if we need to refresh any
        if not force:
            all_valid = True
            for model_id in model_ids:
                existing = self.get_intelligence(model_id)
                if not existing or existing.is_expired():
                    all_valid = False
                    break
            if all_valid:
                logger.debug("All model intelligence is still valid")
                return self.get_intelligence_for_candidates(
                    [{"model": m} for m in model_ids]
                )

        logger.info(f"Refreshing comparative intelligence for {len(model_ids)} models")

        try:
            # 1. Search for each model individually
            all_search_results = {}
            for model_id in model_ids:
                results = self._search_for_model(model_id)
                if results:
                    all_search_results[model_id] = results

            # 2. Search for direct model comparisons ("model a vs model b")
            comparison_results = self._search_model_comparisons(model_ids)

            # Add comparison results to each model's search results
            for model_id in model_ids:
                if model_id not in all_search_results:
                    all_search_results[model_id] = []
                all_search_results[model_id].extend(comparison_results)

            if not all_search_results:
                logger.warning("No search results for any models")
                return {}

            # 3. Create comparative summaries
            intel_results = self._summarize_comparative(model_ids, all_search_results)

            # 4. Store each model's intelligence
            for intel in intel_results.values():
                self._store_intelligence(intel)

            return intel_results

        except Exception as e:
            logger.error(f"Error refreshing comparative intelligence: {e}")
            return {}

    def _search_model_comparisons(self, model_ids: list[str]) -> list[dict]:
        """Search for direct comparisons between models (e.g., 'Claude vs GPT-4')."""
        from itertools import combinations

        from augmentation.search import get_configured_search_provider

        # Use globally configured search provider
        provider = get_configured_search_provider()

        if not provider:
            logger.warning("No search provider available for model comparisons")
            return []

        all_results = []
        search_names = {mid: self._model_to_search_name(mid) for mid in model_ids}

        # Search for pairwise comparisons
        for model_a, model_b in combinations(model_ids, 2):
            name_a = search_names[model_a]
            name_b = search_names[model_b]
            query = f"{name_a} vs {name_b} comparison benchmark"
            try:
                results = provider.search(query, max_results=2)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Comparison search failed for '{query}': {e}")

        # If there are 3+ models, also search for multi-model comparisons
        if len(model_ids) >= 3:
            # Take up to first 4 model names for the comparison query
            names = list(search_names.values())[:4]
            query = " vs ".join(names) + " comparison"
            try:
                results = provider.search(query, max_results=3)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Multi-model comparison search failed: {e}")

        # Deduplicate by URL and convert SearchResult objects to dicts
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.url if hasattr(r, "url") else r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                if hasattr(r, "url"):
                    unique_results.append(
                        {
                            "title": r.title,
                            "url": r.url,
                            "content": r.snippet,
                        }
                    )
                else:
                    unique_results.append(r)

        logger.info(f"Found {len(unique_results)} comparison search results")
        return unique_results

    def _summarize_comparative(
        self, model_ids: list[str], search_results_by_model: dict[str, list[dict]]
    ) -> dict[str, ModelIntel]:
        """Create comparative assessments for multiple models."""
        from providers import registry

        # Format all search results grouped by model
        all_results_text = ""
        for model_id in model_ids:
            search_name = self._model_to_search_name(model_id)
            results = search_results_by_model.get(model_id, [])
            if results:
                results_text = "\n".join(
                    f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}"
                    for r in results[:5]
                )
                all_results_text += f"\n\n### {search_name}\n{results_text}"

        model_names = [self._model_to_search_name(m) for m in model_ids]
        model_list = ", ".join(model_names)

        prompt = f"""You are comparing these specific models for a smart router that will choose between them: {model_list}

SEARCH RESULTS FOR EACH MODEL:
{all_results_text}

For EACH model, create a comparative assessment that helps a routing system choose the right model for a task. Focus on RELATIVE differences between these specific models.

For each model, provide:
1. ASSESSMENT: 2-3 sentences describing when to choose THIS model over the others in this group. Be specific about relative strengths.
2. STRENGTHS: 2-3 key advantages COMPARED TO the other models listed
3. WEAKNESSES: 2-3 disadvantages COMPARED TO the other models listed
4. BEST_FOR: 2-3 specific task types where this model beats the others
5. AVOID_FOR: 1-2 task types where another model in this group would be better

Format your response EXACTLY like this for EACH model:

MODEL: [model name]
ASSESSMENT: [comparative assessment focusing on when to choose this over others]
STRENGTHS: [strength vs others 1, strength vs others 2, ...]
WEAKNESSES: [weakness vs others 1, weakness vs others 2, ...]
BEST_FOR: [task type 1, task type 2, ...]
AVOID_FOR: [task type 1, ...]

---

Provide this for each model: {model_list}"""

        try:
            resolved = registry.resolve_model(self.summarizer_model)
            response = resolved.provider.chat_completion(
                model=resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 1500, "temperature": 0.3},
            )

            content = response.get("content", "")
            if not content:
                return {}

            # Parse the comparative response
            return self._parse_comparative_response(
                model_ids, content, search_results_by_model
            )

        except Exception as e:
            logger.error(f"Error in comparative summarization: {e}")
            return {}

    def _parse_comparative_response(
        self,
        model_ids: list[str],
        response: str,
        search_results_by_model: dict[str, list[dict]],
    ) -> dict[str, ModelIntel]:
        """Parse comparative LLM response into ModelIntel objects."""
        import re

        results = {}
        now = datetime.utcnow()
        expires = now + timedelta(days=self.ttl_days)

        # Split response by MODEL: sections
        sections = re.split(r"\n(?=MODEL:)", response)

        for section in sections:
            if not section.strip():
                continue

            # Extract model name from section
            model_match = re.search(r"MODEL:\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
            if not model_match:
                continue

            model_name = model_match.group(1).strip()

            # Find matching model_id
            matched_model_id = None
            for model_id in model_ids:
                search_name = self._model_to_search_name(model_id)
                if (
                    search_name.lower() in model_name.lower()
                    or model_name.lower() in search_name.lower()
                    or model_id.lower() in model_name.lower()
                ):
                    matched_model_id = model_id
                    break

            if not matched_model_id:
                logger.warning(
                    f"Could not match model name '{model_name}' to any model_id"
                )
                continue

            # Extract fields
            def extract_field(field_name: str) -> str:
                pattern = rf"{field_name}:\s*(.+?)(?:\n[A-Z_]+:|---|$)"
                match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
                return match.group(1).strip() if match else ""

            def split_list(text: str) -> list[str]:
                items = [item.strip() for item in text.split(",")]
                return [item for item in items if item]

            assessment = extract_field("ASSESSMENT")
            if not assessment:
                continue

            # Get sources from search results
            sources = [
                r.get("url", "")
                for r in search_results_by_model.get(matched_model_id, [])[:5]
                if r.get("url")
            ]

            results[matched_model_id] = ModelIntel(
                model_id=matched_model_id,
                intelligence=assessment,
                strengths=split_list(extract_field("STRENGTHS")),
                weaknesses=split_list(extract_field("WEAKNESSES")),
                best_for=split_list(extract_field("BEST_FOR")),
                avoid_for=split_list(extract_field("AVOID_FOR")),
                sources=sources,
                generated_at=now,
                expires_at=expires,
            )

        return results

    def _search_for_model(self, model_id: str) -> list[dict]:
        """Search for model reviews and comparisons."""
        from augmentation.search import get_configured_search_provider

        search_name = self._model_to_search_name(model_id)
        all_results = []

        # Use globally configured search provider
        provider = get_configured_search_provider()

        if not provider:
            logger.warning("No search provider available for model intelligence")
            return []

        for template in SEARCH_QUERY_TEMPLATES:
            query = template.format(model_name=search_name)
            try:
                results = provider.search(query, max_results=3)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.url if hasattr(r, "url") else r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Convert SearchResult to dict for downstream compatibility
                if hasattr(r, "url"):
                    unique_results.append(
                        {
                            "title": r.title,
                            "url": r.url,
                            "content": r.snippet,
                        }
                    )
                else:
                    unique_results.append(r)

        return unique_results[:10]  # Limit total results

    def _summarize_results(
        self, model_id: str, search_results: list[dict]
    ) -> Optional[ModelIntel]:
        """Use LLM to summarize search results into balanced assessment."""
        from providers import registry

        # Format search results for the prompt
        results_text = "\n\n".join(
            f"**{r.get('title', 'Untitled')}**\n{r.get('url', '')}\n{r.get('content', '')}"
            for r in search_results
        )

        search_name = self._model_to_search_name(model_id)

        prompt = f"""Analyze these search results about {search_name} and create a balanced, objective assessment.

SEARCH RESULTS:
{results_text}

Create a concise assessment (2-3 sentences) that:
1. Summarizes the model's key strengths
2. Notes any weaknesses or limitations
3. Compares to alternatives where relevant
4. Is balanced and objective (avoid marketing language)

Also extract:
- STRENGTHS: List 2-4 key strengths (comma-separated)
- WEAKNESSES: List 2-4 weaknesses or limitations (comma-separated)
- BEST_FOR: List 2-3 use cases where this model excels (comma-separated)
- AVOID_FOR: List 1-2 use cases where other models might be better (comma-separated)

Format your response EXACTLY like this:
ASSESSMENT: [Your 2-3 sentence balanced assessment]
STRENGTHS: [strength1, strength2, ...]
WEAKNESSES: [weakness1, weakness2, ...]
BEST_FOR: [use case1, use case2, ...]
AVOID_FOR: [use case1, use case2, ...]"""

        try:
            resolved = registry.resolve_model(self.summarizer_model)
            response = resolved.provider.chat_completion(
                model=resolved.model_id,
                messages=[{"role": "user", "content": prompt}],
                system=None,
                options={"max_tokens": 1000, "temperature": 0.3},
            )

            content = response.get("content", "")
            if not content:
                return None

            # Parse the response
            intel = self._parse_summary_response(model_id, content, search_results)
            return intel

        except Exception as e:
            logger.error(f"Error summarizing for {model_id}: {e}")
            return None

    def _parse_summary_response(
        self, model_id: str, response: str, search_results: list[dict]
    ) -> Optional[ModelIntel]:
        """Parse LLM response into ModelIntel object."""
        import re

        def extract_field(field_name: str) -> str:
            # Match field until the next FIELD_NAME: or end of string
            # Handle both newline-separated and same-line formats
            pattern = rf"{field_name}:\s*(.+?)(?=\n\s*[A-Z_]+:|$)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            # Try alternate pattern for fields on same line
            pattern2 = rf"{field_name}:\s*([^\n]+)"
            match2 = re.search(pattern2, response, re.IGNORECASE)
            return match2.group(1).strip() if match2 else ""

        def split_list(text: str) -> list[str]:
            # Split on comma, clean up
            items = [item.strip() for item in text.split(",")]
            return [item for item in items if item]

        assessment = extract_field("ASSESSMENT")
        if not assessment:
            # Try to use the whole response as assessment
            assessment = response.strip()[:1000]

        now = datetime.utcnow()

        return ModelIntel(
            model_id=model_id,
            intelligence=assessment,
            strengths=split_list(extract_field("STRENGTHS")),
            weaknesses=split_list(extract_field("WEAKNESSES")),
            best_for=split_list(extract_field("BEST_FOR")),
            avoid_for=split_list(extract_field("AVOID_FOR")),
            sources=[r.get("url", "") for r in search_results if r.get("url")],
            generated_at=now,
            expires_at=now + timedelta(days=self.ttl_days),
        )

    def _store_intelligence(self, intel: ModelIntel) -> None:
        """Store intelligence in ChromaDB."""
        doc_id = self._model_to_id(intel.model_id)

        # ChromaDB metadata must be strings, ints, floats, or bools
        metadata = {
            "model_id": intel.model_id,
            "intelligence": intel.intelligence,
            "strengths": "|".join(intel.strengths),
            "weaknesses": "|".join(intel.weaknesses),
            "best_for": "|".join(intel.best_for),
            "avoid_for": "|".join(intel.avoid_for),
            "sources": "|".join(intel.sources[:5]),  # Limit stored sources
            "generated_at": intel.generated_at.isoformat(),
            "expires_at": intel.expires_at.isoformat(),
        }

        # Use upsert pattern (delete then add)
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass  # May not exist

        # Document content is used for semantic search (not really needed here)
        self.collection.add(
            ids=[doc_id],
            documents=[f"{intel.model_id}: {intel.intelligence}"],
            metadatas=[metadata],
        )

        logger.info(f"Stored intelligence for {intel.model_id}")

    def delete_intelligence(self, model_id: str) -> bool:
        """Delete cached intelligence for a model."""
        try:
            doc_id = self._model_to_id(model_id)
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted intelligence for {model_id}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting intelligence for {model_id}: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all cached intelligence."""
        try:
            from context import delete_collection

            delete_collection(COLLECTION_NAME)
            self._collection = None
            logger.info("Cleared all model intelligence")
            return True
        except Exception as e:
            logger.warning(f"Error clearing intelligence: {e}")
            return False

    def get_all_cached(self) -> list[ModelIntel]:
        """Get all cached intelligence entries."""
        try:
            # Get all documents from the collection
            results = self.collection.collection.get()

            if not results or not results.get("ids"):
                return []

            intel_list = []
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                model_id = metadata.get("model_id", "")

                if not model_id:
                    continue

                expires_at_str = metadata.get("expires_at")
                expires_at = (
                    datetime.fromisoformat(expires_at_str)
                    if expires_at_str
                    else datetime.utcnow() + timedelta(days=self.ttl_days)
                )

                intel_list.append(
                    ModelIntel(
                        model_id=model_id,
                        intelligence=metadata.get("intelligence", ""),
                        strengths=metadata.get("strengths", "").split("|")
                        if metadata.get("strengths")
                        else [],
                        weaknesses=metadata.get("weaknesses", "").split("|")
                        if metadata.get("weaknesses")
                        else [],
                        best_for=metadata.get("best_for", "").split("|")
                        if metadata.get("best_for")
                        else [],
                        avoid_for=metadata.get("avoid_for", "").split("|")
                        if metadata.get("avoid_for")
                        else [],
                        sources=metadata.get("sources", "").split("|")
                        if metadata.get("sources")
                        else [],
                        generated_at=datetime.fromisoformat(metadata["generated_at"])
                        if metadata.get("generated_at")
                        else datetime.utcnow(),
                        expires_at=expires_at,
                    )
                )

            return intel_list

        except Exception as e:
            logger.warning(f"Error getting all cached intelligence: {e}")
            return []
