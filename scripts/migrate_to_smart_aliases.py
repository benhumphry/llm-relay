#!/usr/bin/env python3
"""
Migration script: Migrate Aliases, SmartRouters, and SmartEnrichers to unified SmartAliases.

This script migrates existing data from the legacy tables to the new unified SmartAlias table.
Run this script ONCE after updating to the version that includes SmartAliases.

Usage:
    python scripts/migrate_to_smart_aliases.py [--dry-run] [--delete-legacy]

Options:
    --dry-run       Show what would be migrated without making changes
    --delete-legacy Delete legacy table data after successful migration (use with caution!)

The script will:
1. Migrate all Aliases to SmartAliases (simple aliases with optional cache)
2. Migrate all SmartRouters to SmartAliases (with use_routing=True)
3. Migrate all SmartEnrichers to SmartAliases (with use_rag/use_web flags)
4. Handle document store relationships for enrichers

Note: The script checks for name conflicts and skips duplicates.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.connection import get_db_context
from db.models import (
    Alias,
    SmartAlias,
    SmartEnricher,
    SmartRouter,
    smart_alias_stores,
    smart_enricher_stores,
)


def migrate_alias(session, alias, dry_run=False):
    """Migrate a single Alias to SmartAlias."""
    # Check if already exists
    existing = session.query(SmartAlias).filter(SmartAlias.name == alias.name).first()
    if existing:
        print(f"  [SKIP] Alias '{alias.name}' - already exists as SmartAlias")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would migrate Alias '{alias.name}' -> SmartAlias")
        return True

    # Create SmartAlias
    smart_alias = SmartAlias(
        name=alias.name,
        target_model=alias.target_model,
        # Feature toggles (all False for simple alias)
        use_routing=False,
        use_rag=False,
        use_web=False,
        use_cache=alias.use_cache,
        # Cache settings
        cache_similarity_threshold=alias.cache_similarity_threshold,
        cache_match_system_prompt=alias.cache_match_system_prompt,
        cache_match_last_message_only=alias.cache_match_last_message_only,
        cache_ttl_hours=alias.cache_ttl_hours,
        cache_min_tokens=alias.cache_min_tokens,
        cache_max_tokens=alias.cache_max_tokens,
        cache_collection=alias.cache_collection,
        # Cache stats
        cache_hits=alias.cache_hits,
        cache_tokens_saved=alias.cache_tokens_saved,
        cache_cost_saved=alias.cache_cost_saved,
        # Metadata
        tags_json=alias.tags_json,
        description=alias.description,
        enabled=alias.enabled,
    )
    session.add(smart_alias)
    print(f"  [OK] Migrated Alias '{alias.name}' -> SmartAlias")
    return True


def migrate_router(session, router, dry_run=False):
    """Migrate a single SmartRouter to SmartAlias."""
    # Check if already exists
    existing = session.query(SmartAlias).filter(SmartAlias.name == router.name).first()
    if existing:
        print(f"  [SKIP] SmartRouter '{router.name}' - already exists as SmartAlias")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would migrate SmartRouter '{router.name}' -> SmartAlias")
        return True

    # Create SmartAlias with routing enabled
    smart_alias = SmartAlias(
        name=router.name,
        target_model=router.fallback_model,  # Fallback becomes the target
        # Feature toggles
        use_routing=True,
        use_rag=False,
        use_web=False,
        use_cache=router.use_cache,
        # Routing settings
        designator_model=router.designator_model,
        purpose=router.purpose,
        candidates_json=router.candidates_json,
        fallback_model=router.fallback_model,
        routing_strategy=router.strategy,
        session_ttl=router.session_ttl,
        # Model intelligence
        use_model_intelligence=router.use_model_intelligence,
        search_provider=router.search_provider,
        intelligence_model=router.intelligence_model,
        # Cache settings
        cache_similarity_threshold=router.cache_similarity_threshold,
        cache_match_system_prompt=router.cache_match_system_prompt,
        cache_match_last_message_only=router.cache_match_last_message_only,
        cache_ttl_hours=router.cache_ttl_hours,
        cache_min_tokens=router.cache_min_tokens,
        cache_max_tokens=router.cache_max_tokens,
        cache_collection=router.cache_collection,
        # Cache stats
        cache_hits=router.cache_hits,
        cache_tokens_saved=router.cache_tokens_saved,
        cache_cost_saved=router.cache_cost_saved,
        # Metadata
        tags_json=router.tags_json,
        description=router.description,
        enabled=router.enabled,
    )
    session.add(smart_alias)
    print(f"  [OK] Migrated SmartRouter '{router.name}' -> SmartAlias (with routing)")
    return True


def migrate_enricher(session, enricher, dry_run=False):
    """Migrate a single SmartEnricher to SmartAlias."""
    # Check if already exists
    existing = (
        session.query(SmartAlias).filter(SmartAlias.name == enricher.name).first()
    )
    if existing:
        print(
            f"  [SKIP] SmartEnricher '{enricher.name}' - already exists as SmartAlias"
        )
        return False

    if dry_run:
        print(
            f"  [DRY-RUN] Would migrate SmartEnricher '{enricher.name}' -> SmartAlias"
        )
        return True

    # Create SmartAlias with enrichment settings
    smart_alias = SmartAlias(
        name=enricher.name,
        target_model=enricher.target_model,
        # Feature toggles
        use_routing=False,
        use_rag=enricher.use_rag,
        use_web=enricher.use_web,
        use_cache=enricher.use_cache if not enricher.use_web else False,
        # Designator (for web query optimization)
        designator_model=enricher.designator_model,
        # RAG settings
        max_results=enricher.max_results,
        similarity_threshold=enricher.similarity_threshold,
        # Web settings
        max_search_results=enricher.max_search_results,
        max_scrape_urls=enricher.max_scrape_urls,
        # Common enrichment settings
        max_context_tokens=enricher.max_context_tokens,
        rerank_provider=enricher.rerank_provider,
        rerank_model=enricher.rerank_model,
        rerank_top_n=enricher.rerank_top_n,
        # Statistics
        total_requests=enricher.total_requests,
        context_injections=enricher.context_injections,
        search_requests=enricher.search_requests,
        scrape_requests=enricher.scrape_requests,
        # Cache settings
        cache_similarity_threshold=enricher.cache_similarity_threshold,
        cache_match_system_prompt=enricher.cache_match_system_prompt,
        cache_match_last_message_only=enricher.cache_match_last_message_only,
        cache_ttl_hours=enricher.cache_ttl_hours,
        cache_min_tokens=enricher.cache_min_tokens,
        cache_max_tokens=enricher.cache_max_tokens,
        cache_collection=enricher.cache_collection,
        # Cache stats
        cache_hits=enricher.cache_hits,
        cache_tokens_saved=enricher.cache_tokens_saved,
        cache_cost_saved=enricher.cache_cost_saved,
        # Metadata
        tags_json=enricher.tags_json,
        description=enricher.description,
        enabled=enricher.enabled,
    )
    session.add(smart_alias)
    session.flush()  # Get the ID for junction table

    # Migrate document store relationships
    if enricher.use_rag:
        # Get existing document store IDs from junction table
        store_ids = session.execute(
            smart_enricher_stores.select().where(
                smart_enricher_stores.c.smart_enricher_id == enricher.id
            )
        ).fetchall()

        for store_row in store_ids:
            store_id = store_row.document_store_id
            session.execute(
                smart_alias_stores.insert().values(
                    smart_alias_id=smart_alias.id, document_store_id=store_id
                )
            )
            print(f"    -> Linked document store ID {store_id}")

    features = []
    if enricher.use_rag:
        features.append("RAG")
    if enricher.use_web:
        features.append("Web")
    feature_str = "+".join(features) if features else "none"
    print(
        f"  [OK] Migrated SmartEnricher '{enricher.name}' -> SmartAlias ({feature_str})"
    )
    return True


def run_migration(dry_run=False, delete_legacy=False):
    """Run the full migration."""
    print("=" * 60)
    print("Smart Alias Migration Script")
    print("=" * 60)

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    with get_db_context() as session:
        stats = {"aliases": 0, "routers": 0, "enrichers": 0, "skipped": 0}

        # Migrate Aliases
        print("\n[1/3] Migrating Aliases...")
        aliases = session.query(Alias).all()
        print(f"  Found {len(aliases)} aliases")
        for alias in aliases:
            if migrate_alias(session, alias, dry_run):
                stats["aliases"] += 1
            else:
                stats["skipped"] += 1

        # Migrate SmartRouters
        print("\n[2/3] Migrating Smart Routers...")
        routers = session.query(SmartRouter).all()
        print(f"  Found {len(routers)} smart routers")
        for router in routers:
            if migrate_router(session, router, dry_run):
                stats["routers"] += 1
            else:
                stats["skipped"] += 1

        # Migrate SmartEnrichers
        print("\n[3/3] Migrating Smart Enrichers...")
        enrichers = session.query(SmartEnricher).all()
        print(f"  Found {len(enrichers)} smart enrichers")
        for enricher in enrichers:
            if migrate_enricher(session, enricher, dry_run):
                stats["enrichers"] += 1
            else:
                stats["skipped"] += 1

        # Commit or rollback
        if dry_run:
            print("\n*** DRY RUN - Rolling back all changes ***")
            session.rollback()
        else:
            session.commit()
            print("\n[OK] Migration committed successfully!")

        # Delete legacy data if requested
        if delete_legacy and not dry_run:
            print("\n[4/4] Deleting legacy data...")
            # Note: We don't actually delete the tables, just the data
            # The tables can be dropped in a later version
            deleted_aliases = session.query(Alias).delete()
            deleted_routers = session.query(SmartRouter).delete()
            deleted_enrichers = session.query(SmartEnricher).delete()
            session.commit()
            print(
                f"  Deleted {deleted_aliases} aliases, {deleted_routers} routers, {deleted_enrichers} enrichers"
            )
        elif delete_legacy and dry_run:
            print("\n[DRY-RUN] Would delete legacy data from:")
            print(f"  - aliases table: {len(aliases)} rows")
            print(f"  - smart_routers table: {len(routers)} rows")
            print(f"  - smart_enrichers table: {len(enrichers)} rows")

        # Print summary
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"  Aliases migrated:        {stats['aliases']}")
        print(f"  Smart Routers migrated:  {stats['routers']}")
        print(f"  Smart Enrichers migrated:{stats['enrichers']}")
        print(f"  Skipped (duplicates):    {stats['skipped']}")
        print(
            f"  Total SmartAliases:      {stats['aliases'] + stats['routers'] + stats['enrichers']}"
        )

        if not dry_run:
            print("\nMigration complete! You can now use Smart Aliases.")
            print("The legacy pages are marked as (legacy) in the navigation.")
            if not delete_legacy:
                print("\nTo clean up legacy data, run again with --delete-legacy flag.")
        else:
            print("\nThis was a dry run. Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Aliases, SmartRouters, and SmartEnrichers to SmartAliases"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete legacy table data after successful migration",
    )
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run, delete_legacy=args.delete_legacy)


if __name__ == "__main__":
    main()
