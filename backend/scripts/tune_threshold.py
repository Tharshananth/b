"""tune_threshold.py — Read-only script to tune the vector search score threshold.

Run from the backend/ directory:
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --queries extra_queries.txt --k 8

Prints the top-k cosine scores for a set of realistic KB questions and shows
a distribution summary and keep-counts at several candidate thresholds.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# ── Path setup (run from backend/) ────────────────────────────────────────────
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# Suppress noisy init logs during the script run; show WARNING+ only.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Default query list (15 realistic questions about the VoxelBox KB) ─────────
DEFAULT_QUERIES = [
    "How do I upload a document to VoxelBox?",
    "How do I download files from VoxelBox?",
    "How does VoxelBox integrate with PACS systems?",
    "What versions of VoxelBox are currently supported?",
    "What new features were introduced in the latest VoxelBox release?",
    "How do I export neuronavigation data from VoxelBox?",
    "What formats are supported for neuronavigation export?",
    "How is resting-state fMRI processed in VoxelBox?",
    "Where can I find the validation report for VoxelBox?",
    "What TMS data types does VoxelBox support?",
    "How do clinicians access patient results in VoxelBox?",
    "What are the system requirements for running VoxelBox?",
    "How do I configure DICOM settings in VoxelBox?",
    "What quality checks are applied during rs-fMRI processing?",
    "How do I contact VoxelBox support or submit a bug report?",
]

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]


def percentile(sorted_vals: list, p: float) -> float:
    """Return the p-th percentile (0-100) of a sorted list."""
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune the vector search score threshold for the VoxelBox KB."
    )
    p.add_argument(
        "--queries",
        metavar="FILE",
        help="Path to a text file with additional queries (one per line).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top results to retrieve per query (default: 5).",
    )
    return p.parse_args()


def load_extra_queries(path: str) -> list:
    queries = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    queries.append(line)
    except OSError as exc:
        print(f"WARNING: could not read --queries file: {exc}", file=sys.stderr)
    return queries


def main() -> None:
    args = parse_args()

    queries = list(DEFAULT_QUERIES)
    if args.queries:
        extra = load_extra_queries(args.queries)
        queries.extend(extra)
        print(f"[+] Loaded {len(extra)} extra queries from {args.queries}")

    print(f"\n{'='*72}")
    print(f"  VoxelBox Vector Search Threshold Tuner")
    print(f"  k={args.k}  |  queries={len(queries)}")
    print(f"{'='*72}\n")

    # Import AFTER sys.path is set
    try:
        from vector_db.store import VectorStore, distance_to_cosine
    except ImportError as exc:
        sys.exit(f"ERROR: Could not import VectorStore: {exc}")

    try:
        store = VectorStore()
    except Exception as exc:
        sys.exit(f"ERROR: Could not initialize VectorStore: {exc}")

    # Read the threshold that is currently configured
    current_threshold = store.config.vector_db.search.score_threshold

    all_scores: list[float] = []

    for qi, query in enumerate(queries, 1):
        print(f"[{qi:02d}/{len(queries):02d}] {query}")
        print(f"  {'COSINE':>7}  {'STATUS':>8}  {'SOURCE':<30}  {'PG':>3}  SNIPPET")
        print(f"  {'-'*7}  {'-'*8}  {'-'*30}  {'-'*3}  {'-'*42}")

        try:
            # Use similarity_search_with_score directly (read-only)
            raw = store.vectordb.similarity_search_with_score(query=query, k=args.k)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            print()
            continue

        for doc, distance in raw:
            cosine = distance_to_cosine(float(distance), store._distance_space)
            all_scores.append(cosine)

            status = "KEPT   " if cosine >= current_threshold else "dropped"
            source = str(doc.metadata.get("source", doc.metadata.get("file_name", "??")))
            source = Path(source).name[:30]
            page = str(doc.metadata.get("page", doc.metadata.get("page_number", "?")))[:3]
            snippet = doc.page_content.replace("\n", " ").strip()[:70]

            print(f"  {cosine:7.3f}  {status:>8}  {source:<30}  {page:>3}  {snippet}")

        print()

    # ── Score distribution summary ─────────────────────────────────────────────
    if not all_scores:
        print("No scores collected — is the knowledge base empty?")
        return

    all_scores.sort()
    print(f"{'='*72}")
    print(f"  Score Distribution  ({len(all_scores)} results across {len(queries)} queries)")
    print(f"{'='*72}")
    print(f"  Min    : {all_scores[0]:.3f}")
    print(f"  25th % : {percentile(all_scores, 25):.3f}")
    print(f"  Median : {percentile(all_scores, 50):.3f}")
    print(f"  75th % : {percentile(all_scores, 75):.3f}")
    print(f"  Max    : {all_scores[-1]:.3f}")
    print()

    # ── Keep-count table at candidate thresholds ───────────────────────────────
    print(f"  {'Threshold':>10}  {'Kept':>6}  {'Dropped':>8}  {'% Kept':>8}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}")
    for t in CANDIDATE_THRESHOLDS:
        kept = sum(1 for s in all_scores if s >= t)
        dropped = len(all_scores) - kept
        pct = 100.0 * kept / len(all_scores)
        marker = "  ← current" if abs(t - current_threshold) < 1e-6 else ""
        print(f"  {t:10.2f}  {kept:6d}  {dropped:8d}  {pct:7.1f}%{marker}")
    print()


if __name__ == "__main__":
    main()
