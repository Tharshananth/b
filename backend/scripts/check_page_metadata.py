#!/usr/bin/env python3
"""
Inspect vector store chunks for page metadata (page / page_label).
Reports, per source file, how many chunks have page metadata.
If any PDF chunks lack pages, prints:
    "Re-index required: POST /api/documents/refresh as an admin"
"""
import os
import sys
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def check_metadata_via_chroma() -> Dict[str, Tuple[int, int]]:
    """Try to read chunk metadata via vector_db store / Chroma client."""
    from vector_db.retriever import get_retriever
    retriever = get_retriever()
    store = retriever.get_vector_store()
    data = store.vectordb.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    stats: Dict[str, Dict[str, int]] = {}
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source", "Unknown")
        if source not in stats:
            stats[source] = {"total": 0, "with_page": 0}
        stats[source]["total"] += 1
        if "page" in meta or "page_label" in meta:
            stats[source]["with_page"] += 1

    return {k: (v["with_page"], v["total"]) for k, v in stats.items()}


def check_metadata_via_sqlite() -> Dict[str, Tuple[int, int]]:
    """Fallback: read chunk metadata directly from chroma.sqlite3."""
    from config import get_config
    cfg = get_config()
    db_path = Path(cfg.vector_db.persist_directory) / "chroma.sqlite3"
    if not db_path.is_file():
        # Try relative to backend
        db_path = BACKEND_DIR / cfg.vector_db.persist_directory / "chroma.sqlite3"
    if not db_path.is_file():
        # Search for chroma.sqlite3
        for candidate in [BACKEND_DIR / "chroma_db" / "chroma.sqlite3", Path("chroma_db/chroma.sqlite3")]:
            if candidate.is_file():
                db_path = candidate
                break

    if not db_path.is_file():
        raise FileNotFoundError(f"Chroma DB sqlite file not found at {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            m_source.string_value AS source,
            COUNT(DISTINCT m_source.id) AS total_chunks,
            COUNT(DISTINCT CASE WHEN m_page.id IS NOT NULL OR m_plabel.id IS NOT NULL THEN m_source.id END) AS with_page
        FROM embedding_metadata m_source
        LEFT JOIN embedding_metadata m_page 
            ON m_source.id = m_page.id AND m_page.key = 'page'
        LEFT JOIN embedding_metadata m_plabel 
            ON m_source.id = m_plabel.id AND m_plabel.key = 'page_label'
        WHERE m_source.key = 'source'
        GROUP BY m_source.string_value
    """)
    rows = cur.fetchall()
    conn.close()
    return {row[0]: (row[2], row[1]) for row in rows}


def main():
    try:
        stats = check_metadata_via_chroma()
    except Exception:
        stats = check_metadata_via_sqlite()

    if not stats:
        print("Vector store contains no indexed chunks.")
        return 0

    reindex_required = False
    print("=" * 70)
    print("Vector Store Page Metadata Report:")
    print("=" * 70)

    for source in sorted(stats.keys()):
        with_page, total = stats[source]
        is_pdf = source.lower().endswith(".pdf")
        pct = (with_page / total * 100) if total else 0
        status = "OK" if with_page == total else "PARTIAL" if with_page > 0 else "NO PAGES"
        print(f"  [{status:<8}] {source}: {with_page}/{total} chunks have page info ({pct:.0f}%)")

        if is_pdf and with_page < total:
            reindex_required = True

    print("=" * 70)
    if reindex_required:
        print("Re-index required: POST /api/documents/refresh as an admin")
    else:
        print("All PDF chunks contain page metadata. No re-index required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
