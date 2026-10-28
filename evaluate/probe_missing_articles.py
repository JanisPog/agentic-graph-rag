"""Probe Neo4j for the 49 expert-set article_ids that didn't resolve.

Reads `evaluate/results/appendix/per_query.jsonl`, collects the gold
`article_ids` of every expert query with `resolved_doc_count == 0`, and tries
to locate them in the graph under three matching modes:

  1. Exact match on Document.id or Document.source_id
  2. STARTS WITH — the missing id is a prefix of an indexed Document id
     (catches "ingest stores concatenated form, lookup uses component")
  3. Component split — for double-hash ids (>64 chars), split into the two
     64-char halves and try each as a Document.id / source_id
     (catches "ingest stores ONE half, WixQA stores both concatenated")

Outputs a markdown report at `evaluate/results/appendix/missing_articles.md`
plus a JSON blob at `evaluate/results/appendix/missing_articles.json`.

Run inside the container:

    sudo docker compose exec -T graph-rag-agent \\
        python evaluate/probe_missing_articles.py
"""
from __future__ import annotations
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402,F401
from graph.neo4j_client import Neo4jClient  # noqa: E402

log = logging.getLogger("probe_missing")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

JSONL_PATH = Path("evaluate/results/appendix/per_query.jsonl")
OUT_MD     = Path("evaluate/results/appendix/missing_articles.md")
OUT_JSON   = Path("evaluate/results/appendix/missing_articles.json")


# ── Cypher probes ────────────────────────────────────────────────────────────

# 1. Exact match on Document.id OR Document.source_id
_Q_EXACT = """
MATCH (d:Document)
WHERE d.id IN $ids OR d.source_id IN $ids
RETURN d.id AS d_id, d.source_id AS d_source_id, d.title AS d_title
"""

# 2. The missing id is a PREFIX of an indexed Document id/source_id
#    (e.g. WixQA snapshot stored the first 64 chars only, ingest stored full)
_Q_PREFIX_OF = """
UNWIND $ids AS qid
MATCH (d:Document)
WHERE d.id STARTS WITH qid OR d.source_id STARTS WITH qid
RETURN qid AS query_id, d.id AS d_id, d.source_id AS d_source_id,
       d.title AS d_title
"""

# 3. An indexed Document id/source_id is a PREFIX of the missing id
#    (the inverse: WixQA stored concatenated form, ingest stored a component)
_Q_PREFIX_FOR = """
UNWIND $ids AS qid
MATCH (d:Document)
WHERE qid STARTS WITH d.id OR qid STARTS WITH d.source_id
RETURN qid AS query_id, d.id AS d_id, d.source_id AS d_source_id,
       d.title AS d_title
"""

# 4. For double-hash ids: split into two 64-char halves and check each.
#    Returns rows keyed by the original (long) id so we can attribute findings.
_Q_HALVES = """
UNWIND $halves AS pair          // {long_id, half}
MATCH (d:Document)
WHERE d.id = pair.half OR d.source_id = pair.half
RETURN pair.long_id AS long_id, pair.half AS half,
       d.id AS d_id, d.source_id AS d_source_id, d.title AS d_title
"""


def _load_missing_ids() -> tuple[list[str], dict[str, list[str]]]:
    """Return (distinct_missing_ids, query_to_ids) for expert zero-resolved rows."""
    if not JSONL_PATH.exists():
        log.error("per_query.jsonl not found at %s", JSONL_PATH)
        sys.exit(1)

    by_query: dict[str, list[str]] = {}
    all_missing: list[str] = []
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dataset") != "expert":
            continue
        if r.get("resolved_doc_count", 0) != 0:
            continue
        ids = r.get("gold_article_ids") or []
        by_query[r["qid"]] = ids
        all_missing.extend(ids)

    distinct = sorted(set(all_missing))
    log.info("Loaded %d expert queries with zero resolved gold, %d distinct missing ids",
             len(by_query), len(distinct))
    return distinct, by_query


def _make_halves(ids: list[str]) -> list[dict]:
    """For ids > 64 chars, split into 64-char halves and produce probe rows."""
    halves: list[dict] = []
    for aid in ids:
        if len(aid) >= 128 and len(aid) % 64 == 0:
            # Could be 2x64 or 3x64; try each 64-char window
            for i in range(0, len(aid), 64):
                half = aid[i:i + 64]
                if len(half) == 64:
                    halves.append({"long_id": aid, "half": half})
    return halves


def _summarise_lengths(ids: list[str]) -> dict[int, int]:
    return dict(Counter(len(i) for i in ids))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    missing_ids, by_query = _load_missing_ids()
    length_hist = _summarise_lengths(missing_ids)
    log.info("ID-length histogram: %s", length_hist)

    halves = _make_halves(missing_ids)
    log.info("Long-id components to probe: %d (from %d long ids)",
             len(halves),
             sum(1 for i in missing_ids if len(i) > 64))

    with Neo4jClient() as client:
        log.info("Probe 1/4: exact id / source_id match …")
        r_exact = client.run(_Q_EXACT, {"ids": missing_ids})
        log.info("  exact matches: %d", len(r_exact))

        log.info("Probe 2/4: missing id is PREFIX of an indexed doc id …")
        r_pref_of = client.run(_Q_PREFIX_OF, {"ids": missing_ids})
        log.info("  prefix-of matches: %d", len(r_pref_of))

        log.info("Probe 3/4: an indexed doc id is PREFIX of the missing id …")
        r_pref_for = client.run(_Q_PREFIX_FOR, {"ids": missing_ids})
        log.info("  prefix-for matches: %d", len(r_pref_for))

        log.info("Probe 4/4: split long ids into 64-char halves …")
        r_halves = client.run(_Q_HALVES, {"halves": halves}) if halves else []
        log.info("  half-id matches: %d", len(r_halves))

    # ── Aggregate which missing ids became recoverable ──────────────────────
    recovered: dict[str, list[dict]] = {}

    def _add(qid: str, source: str, row: dict) -> None:
        recovered.setdefault(qid, []).append({
            "match_mode": source,
            "indexed_id": row.get("d_id"),
            "indexed_source_id": row.get("d_source_id"),
            "title": row.get("d_title"),
        })

    # Exact: the row.d_id or row.d_source_id equals one of missing_ids
    for row in r_exact:
        d_id  = row.get("d_id")
        d_src = row.get("d_source_id")
        for qid in missing_ids:
            if qid == d_id or qid == d_src:
                _add(qid, "exact", row)

    for row in r_pref_of:
        _add(row["query_id"], "missing_is_prefix_of_indexed", row)

    for row in r_pref_for:
        _add(row["query_id"], "indexed_is_prefix_of_missing", row)

    for row in r_halves:
        _add(row["long_id"], f"half_match[{row['half'][:8]}…]", row)

    truly_missing = [aid for aid in missing_ids if aid not in recovered]
    log.info("Recoverable via at least one match mode: %d / %d",
             len(recovered), len(missing_ids))
    log.info("Truly missing (no match in any mode): %d / %d",
             len(truly_missing), len(missing_ids))

    # ── Write JSON + Markdown reports ───────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "expert_queries_zero_resolved": len(by_query),
        "distinct_missing_ids":         len(missing_ids),
        "id_length_histogram":          length_hist,
        "recovered":                    recovered,
        "truly_missing":                truly_missing,
    }, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON)

    lines: list[str] = []
    lines.append("# Missing expert article_ids — Neo4j probe")
    lines.append("")
    lines.append(f"- Expert queries with zero resolved gold articles: **{len(by_query)}**")
    lines.append(f"- Distinct missing `article_ids`: **{len(missing_ids)}**")
    lines.append(f"- ID-length histogram: `{length_hist}`")
    lines.append(f"- Recoverable via some match mode: **{len(recovered)}**")
    lines.append(f"- Truly missing (no match in any mode): **{len(truly_missing)}**")
    lines.append("")

    if recovered:
        lines.append("## Recoverable IDs")
        lines.append("")
        lines.append("| missing id (head…tail) | match mode | indexed id (head…tail) | title |")
        lines.append("|---|---|---|---|")
        for aid, hits in sorted(recovered.items()):
            for h in hits:
                ix = h.get("indexed_id") or h.get("indexed_source_id") or "—"
                title = (h.get("title") or "").replace("|", "\\|")[:60]
                lines.append(
                    f"| `{aid[:10]}…{aid[-6:]}` | {h['match_mode']} | "
                    f"`{(ix or '')[:10]}…{(ix or '')[-6:]}` | {title} |"
                )
        lines.append("")

    if truly_missing:
        lines.append("## Truly missing IDs")
        lines.append("")
        lines.append("These reference articles that are not present in any form in the indexed graph.")
        lines.append("")
        for aid in truly_missing:
            lines.append(f"- `{aid}`")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", OUT_MD)


if __name__ == "__main__":
    main()
