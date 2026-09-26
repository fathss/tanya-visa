"""Run the retrieval eval set and report the 9 relevance metrics.

Prints per-query results plus the score separation used to calibrate
SIMILARITY_THRESHOLD. Re-run after any chunking or embedding change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.embedder import embed_queries
from rag.retriever import Retriever


def _is_relevant_hit(query: dict, hit) -> bool:
    expected_source = query.get("expected_source")
    expected_section = query.get("expected_section")

    if expected_source and expected_source in hit.chunk.source:
        return True

    return bool(expected_section) and expected_section in str(hit.chunk.locator_value)


def evaluate(
    retriever: Retriever, queries: list[dict], vectors: list[list[float]], top_k: int
) -> list[dict]:
    rows = []
    for query, vector in zip(queries, vectors):
        hits = retriever.search(vector, top_k=top_k, threshold=None)
        top1 = hits[0].score if hits else None

        if query["kind"] == "relevant":
            passed = any(_is_relevant_hit(query, hit) for hit in hits)
        else:
            passed = None

        rows.append({"query": query, "hits": hits, "top1": top1, "passed": passed})

    return rows


def main() -> None:
    payload = json.loads(Path(config.EVAL_FILE).read_text(encoding="utf-8"))
    queries = payload["queries"]
    retriever = Retriever.load()

    print(f"index: {retriever.index.ntotal} vectors, dim {retriever.index.d}, top_k={config.TOP_K_RETRIEVAL}\n")

    vectors = embed_queries([query["query"] for query in queries])
    rows = evaluate(retriever, queries, vectors, config.TOP_K_RETRIEVAL)

    for row in rows:
        query = row["query"]
        mark = "  " if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        top1 = f"{row['top1']:.4f}" if row["top1"] is not None else "  none"
        print(f"[{mark}] {query['id']}  top1={top1}  {query['query']}")
        if row["passed"] is False:
            for hit in row["hits"][:3]:
                print(f"         -> {hit.score:.4f}  {hit.chunk.source[:40]} | {hit.chunk.locator_label or '(dokumen)'}")

    relevant = [r for r in rows if r["query"]["kind"] == "relevant"]
    off_topic = [r for r in rows if r["query"]["kind"] == "off_topic"]

    n_pass = sum(1 for r in relevant if r["passed"])
    pct = 100 * n_pass / len(relevant) if relevant else 0.0
    print(f"\nretrieval relevance: {n_pass}/{len(relevant)} = {pct:.0f}%   (target >= 80%)")

    rel_scores = [r["top1"] for r in relevant if r["top1"] is not None]
    off_scores = [r["top1"] for r in off_topic if r["top1"] is not None]
    if rel_scores:
        print(f"relevant top1:  min={min(rel_scores):.4f}  max={max(rel_scores):.4f}")
    if off_scores:
        print(f"off_topic top1: min={min(off_scores):.4f}  max={max(off_scores):.4f}")
    if rel_scores and off_scores:
        lo, hi = max(off_scores), min(rel_scores)
        if lo < hi:
            print(f"separation: {lo:.4f} < {hi:.4f}  -> suggested threshold ~{(lo + hi) / 2:.2f}")
        else:
            print(f"NO separation: off_topic max {lo:.4f} >= relevant min {hi:.4f}")


if __name__ == "__main__":
    main()
