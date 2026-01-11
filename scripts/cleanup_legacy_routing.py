#!/usr/bin/env python3
"""
Cleanup script: Remove legacy routing files after SmartAlias migration.

Run this script ONLY AFTER:
1. Running migrate_to_smart_aliases.py successfully
2. Confirming SmartAliases work correctly in the admin UI
3. Testing that all features (routing, RAG, web, cache) work

Usage:
    python scripts/cleanup_legacy_routing.py [--dry-run] [--force]

Options:
    --dry-run  Show what would be deleted without making changes
    --force    Skip confirmation prompt

This script will:
1. Delete legacy Python files (db/aliases.py, db/smart_routers.py, etc.)
2. Delete legacy templates (aliases.html, routers.html, enrichers.html)
3. Update base.html to remove legacy navigation items

Note: This is a destructive operation. Make sure you have a backup!
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent

# Files to delete
LEGACY_FILES = [
    # Database CRUD files
    "db/aliases.py",
    "db/smart_routers.py",
    "db/smart_enrichers.py",
    # Routing engine files
    "routing/smart_router.py",
    "routing/smart_enricher.py",
    # Template files
    "admin/templates/aliases.html",
    "admin/templates/routers.html",
    "admin/templates/enrichers.html",
]


def remove_legacy_nav_items(dry_run=False):
    """Remove legacy navigation items from base.html."""
    base_html = project_root / "admin/templates/base.html"

    if not base_html.exists():
        print(f"  [WARN] base.html not found at {base_html}")
        return False

    content = base_html.read_text()

    # Pattern to match the legacy navigation block
    # This matches from "<!-- Legacy items" to the closing </li> of Smart Enrichers (legacy)
    legacy_pattern = r"\s*<!-- Legacy items \(will be removed after migration\) -->.*?Smart Enrichers \(legacy\)\s*</a>\s*</li>"

    if re.search(legacy_pattern, content, re.DOTALL):
        if dry_run:
            print("  [DRY-RUN] Would remove legacy nav items from base.html")
        else:
            new_content = re.sub(legacy_pattern, "", content, flags=re.DOTALL)
            base_html.write_text(new_content)
            print("  [OK] Removed legacy nav items from base.html")
        return True
    else:
        print("  [SKIP] Legacy nav items not found in base.html (already removed?)")
        return False


def cleanup_legacy_files(dry_run=False, force=False):
    """Remove legacy routing files."""
    print("=" * 60)
    print("Legacy Routing Cleanup Script")
    print("=" * 60)

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Check which files exist
    existing_files = []
    for rel_path in LEGACY_FILES:
        full_path = project_root / rel_path
        if full_path.exists():
            existing_files.append(rel_path)

    if not existing_files:
        print("\nNo legacy files found. Cleanup may have already been done.")
        return

    print(f"\nFound {len(existing_files)} legacy files to delete:")
    for f in existing_files:
        print(f"  - {f}")

    # Confirmation
    if not dry_run and not force:
        print("\n*** WARNING: This will permanently delete the above files! ***")
        response = input("Are you sure you want to continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Delete files
    print("\n[1/2] Deleting legacy files...")
    deleted = 0
    for rel_path in existing_files:
        full_path = project_root / rel_path
        if dry_run:
            print(f"  [DRY-RUN] Would delete {rel_path}")
            deleted += 1
        else:
            try:
                full_path.unlink()
                print(f"  [OK] Deleted {rel_path}")
                deleted += 1
            except Exception as e:
                print(f"  [ERROR] Failed to delete {rel_path}: {e}")

    # Update base.html navigation
    print("\n[2/2] Updating navigation...")
    remove_legacy_nav_items(dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary")
    print("=" * 60)
    print(f"  Files deleted: {deleted}/{len(existing_files)}")

    if not dry_run:
        print("\nCleanup complete!")
        print("\nNOTE: You may also want to:")
        print("  1. Update db/__init__.py to remove legacy exports")
        print("  2. Update routing/__init__.py to remove legacy exports")
        print("  3. Restart the application to apply changes")
    else:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up legacy routing files after SmartAlias migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    cleanup_legacy_files(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
