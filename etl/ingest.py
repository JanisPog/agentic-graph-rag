"""ETL ingestion pipeline v2 — flexible, metadata-rich, SAGE-enabled.

Supports any dataset with question/answer/document structure. Schema is detected
automatically via the LLM-based SchemaDetectorAgent; use --archetype to override.

Two-file mode: pass --secondary to ingest a paired dataset (e.g. QA + KB articles).
The primary dataset's ref_field is used to create REFERENCES edges between documents.

Usage::

    py etl/ingest.py --clear --limit 1000                                   # single file
    py etl/ingest.py --primary data/wixqa_expertwritten.csv \\
                     --secondary data/wix_kb_corpus.csv --clear             # paired ingest
    py etl/ingest.py --primary data/wixqa_expertwritten.csv \\
                     --secondary data/wix_kb_corpus.csv --articles --clear  # only referenced KB articles
    py etl/ingest.py --primary data/wixqa_expertwritten.csv \\
                     --secondary data/wix_kb_corpus.csv --articles \\
                     --distractor-sample 2000 --clear                       # gold + 2000 distractor articles
    py etl/ingest.py --detect-only                                          # schema only
    py etl/ingest.py --no-embed                                             # BM25 only
    py etl/ingest.py --no-sage                                              # skip SAGE
    py etl/ingest.py --csv data/faq.json --archetype qa
    py etl/ingest.py --entity-mode fast                                     # regex only
"""
import argparse
import ast
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import config
from etl.schema_detector import detect_schema, detect_paired_schemas, load_raw, ColumnMapping, DatasetProfile
from etl.chunker_v3 import make_chunks, ChunkData
from etl.ner_v2 import extract_entities, generate_surface_forms, generate_synonyms_llm
from etl.metadata_builder import build_metadata_nodes, build_concept_nodes, build_belongs_to_edges
from etl.embedder import embed_texts
from graph.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def _doc_type_for(archetype: str) -> str:
    # kb_article is a chunking strategy; stored as doc_type="document" to match planner vocab.
    return "document" if archetype == "kb_article" else archetype


SOURCE_SYSTEM_BY_DATASOURCE = {
    "customer_support_tickets_200k":  "customer_support",
    "customer_support_tickets_test":          "customer_support",
    "customer_support_tickets_known_issues":  "customer_support",
    "wixqa_expertwritten":            "expertQA",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _batches(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _filter_df(df: pd.DataFrame, mapping: ColumnMapping, limit: int | None) -> pd.DataFrame:
    """Apply status filter, deduplicate on id_field, and optionally limit rows."""
    if mapping.status_field and mapping.status_keep:
        before = len(df)
        df = df[df[mapping.status_field].isin(mapping.status_keep)].copy()
        log.info("  Status filter (%s ∈ %s): %d → %d rows",
                 mapping.status_field, mapping.status_keep, before, len(df))

    if mapping.id_field and mapping.id_field in df.columns:
        df = df.drop_duplicates(subset=mapping.id_field).reset_index(drop=True)
        log.info("  After dedup on '%s': %d rows", mapping.id_field, len(df))

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if limit:
        df = df.head(limit)
        log.info("  Limiting to %d rows (--limit)", limit)

    return df


# A token made only of 64-char hex ids plus separators (comma / whitespace / quotes).
_HEX_TOKEN_RE = re.compile(r"^[0-9a-fA-F,\s'\"]+$")
_HEX_ID_RE = re.compile(r"[0-9a-fA-F]{64}")


def _split_concat_ids(token: str) -> list[str]:
    """Extract individual 64-char SHA-256 IDs from a WixQA multi-article ref token."""
    t = token.strip()
    if len(t) > 64 and _HEX_TOKEN_RE.match(t):
        ids = _HEX_ID_RE.findall(t)
        if ids:
            return ids
    return [t]


def _parse_ref_ids(raw: str) -> list[str]:
    """Parse a ref_field value — handles Python list literals and plain string IDs."""
    raw = raw.strip()
    items: list[str]
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            items = [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            items = [raw]
    else:
        items = [raw] if raw and raw.lower() not in ("nan", "none", "") else []
    return [part for item in items for part in _split_concat_ids(item)]


# ── Document date extraction ────────────────────────────────────────────────

_CREATED_CANDIDATES = [
    "created_at", "created", "creation_date",
    "ticket_created_date", "date", "created_date",
]
_UPDATED_CANDIDATES = [
    "updated_at", "updated", "modified", "last_modified",
    "ticket_resolved_date", "resolved_at", "closed_at",
]


def _normalise_date(raw: str) -> str:
    """Try to parse a date string into ISO-8601 format. Return '' on failure."""
    raw = raw.strip()
    if not raw or raw.lower() in ("nan", "none", "null", "nat", ""):
        return ""
    # Already ISO?
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    log.debug("Could not parse date value %r — leaving empty", raw)
    return ""


def _extract_document_dates(
    row_dict: dict,
    mapping,  # ColumnMapping
) -> tuple[str, str]:
    """Return (created_at_iso, updated_at_iso); updated_at falls back to created_at."""
    all_fields = set(row_dict.keys())
    if mapping.metadata_fields:
        all_fields.update(mapping.metadata_fields)

    def _find(candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate in all_fields:
                val = str(row_dict.get(candidate, "")).strip()
                normalised = _normalise_date(val)
                if normalised:
                    return normalised
        return ""

    created_at = _find(_CREATED_CANDIDATES)
    updated_at = _find(_UPDATED_CANDIDATES)

    if not updated_at and created_at:
        updated_at = created_at

    return created_at, updated_at


# ── Per-dataset processing ─────────────────────────────────────────────────────

def _process_dataset(
    profile: DatasetProfile,
    df: pd.DataFrame,
    entity_mode: str,
) -> tuple[list[ChunkData], list[dict], dict, list[dict], list[dict], dict, list[dict]]:
    """Build all in-memory structures for one dataset."""
    mapping = profile.mapping
    datasource_id = f"source_{profile.file_type}_{mapping.archetype}"
    file_stem = Path(profile.file_path).stem
    resolved_source_system = SOURCE_SYSTEM_BY_DATASOURCE.get(file_stem, file_stem.lower())

    all_chunks:    list[ChunkData] = []
    doc_records:   list[dict]      = []
    entity_map:    dict[str, dict] = {}
    mentions:      list[dict]      = []
    meta_nodes:    list[dict]      = []
    concept_map:   dict[str, dict] = {}
    belongs_edges: list[dict]      = []

    _DIAG_ROWS = 3   # log per-stage timing for the first N rows only

    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row")):
        diag = row_idx < _DIAG_ROWS
        t0 = time.perf_counter() if diag else 0.0

        row_dict = row.to_dict()

        if mapping.id_field and mapping.id_field in row_dict:
            raw_id = str(row_dict[mapping.id_field])
            doc_id = str(int(float(raw_id))) if raw_id.replace(".", "").isdigit() else raw_id
        else:
            doc_id = str(hash(str(row_dict)))

        t1 = time.perf_counter() if diag else 0.0
        chunks = make_chunks(row_dict, mapping)
        all_chunks.extend(chunks)
        t2 = time.perf_counter() if diag else 0.0

        title_val = str(row_dict.get(mapping.title_field, "")).strip() if mapping.title_field else ""
        created_at, updated_at = _extract_document_dates(row_dict, mapping)
        doc_records.append({
            "id":          doc_id,
            "source_id":   datasource_id,
            "source_system": resolved_source_system,
            "title":       title_val or doc_id,
            "doc_type":    _doc_type_for(mapping.archetype),
            "status":      str(row_dict.get(mapping.status_field, "")).strip() if mapping.status_field else "",
            "created_at":  created_at,
            "updated_at":  updated_at,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })

        t3 = time.perf_counter() if diag else 0.0
        meta_nodes.extend(build_metadata_nodes(doc_id, row_dict, mapping))
        t4 = time.perf_counter() if diag else 0.0

        entities = extract_entities(row_dict, mapping)
        t5 = time.perf_counter() if diag else 0.0

        for e in entities:
            entity_map[e.id] = {"id": e.id, "name": e.name, "type": e.type,
                                 "source": e.source, "confidence": e.confidence}
            for chunk in chunks:
                mentions.append({"chunk_id": chunk.id, "entity_id": e.id,
                                  "confidence": e.confidence, "method": e.source})

        for c in build_concept_nodes(entities):
            concept_map[c["id"]] = c
        belongs_edges.extend(build_belongs_to_edges(entities))
        t6 = time.perf_counter() if diag else 0.0

        if diag:
            log.info(
                "ROW %d timings — id:%.3fs  chunks:%.3fs  doc:%.3fs  "
                "meta:%.3fs  ner:%.3fs  concepts:%.3fs  TOTAL:%.3fs",
                row_idx,
                t1 - t0, t2 - t1, t3 - t2,
                t4 - t3, t5 - t4, t6 - t5,
                t6 - t0,
            )

    return all_chunks, doc_records, entity_map, mentions, meta_nodes, concept_map, belongs_edges


def _build_references_edges(
    primary_df: pd.DataFrame,
    primary_mapping: ColumnMapping,
    secondary_mapping: ColumnMapping,
    secondary_doc_ids: set[str],
) -> list[dict]:
    """Build REFERENCES edges from primary documents to secondary documents via ref_field."""
    if not primary_mapping.ref_field or primary_mapping.ref_field not in primary_df.columns:
        return []

    edges = []
    for _, row in primary_df.iterrows():
        raw_id = str(row.get(primary_mapping.id_field, "")) if primary_mapping.id_field else ""
        src_id = str(int(float(raw_id))) if raw_id.replace(".", "").isdigit() else raw_id

        ref_raw = str(row.get(primary_mapping.ref_field, ""))
        for target_id in _parse_ref_ids(ref_raw):
            if target_id in secondary_doc_ids:
                edges.append({"src_id": src_id, "dst_id": target_id})

    log.info("REFERENCES edges: %d primary→secondary links built (ref_field=%s)",
             len(edges), primary_mapping.ref_field)
    return edges


# ── KnowledgeBaseContext helpers ───────────────────────────────────────────────

def _detect_kb_context(sample_texts: list[str]) -> dict:
    """Ask the LLM to infer organisation name and domain from a sample of chunk texts."""
    from agents.llm import chat_json  # noqa: PLC0415

    combined = "\n---\n".join(sample_texts[:30])
    prompt = f"""\
You are analysing a customer-support knowledge base to extract its organisational context.

Read the following sample passages and return a JSON object with:
{{
  "organisation_name": "<name of the product or company the KB is about, e.g. 'Wix'>",
  "domain_description": "<one sentence describing what the product/service does>"
}}

Sample passages:
{combined}"""

    try:
        result = chat_json(prompt)
        if not isinstance(result, dict):
            return {}
        return result
    except Exception as exc:
        log.warning("KB context detection failed (%s); skipping context node", exc)
        return {}


def _write_kb_context(client, kb_ctx: dict) -> None:
    """MERGE a single KnowledgeBaseContext node into Neo4j."""
    if not kb_ctx:
        return
    client.run(
        """
        MERGE (k:KnowledgeBaseContext {id: 'kb_context'})
        SET k.organisation_name  = $org_name,
            k.domain_description = $domain
        """,
        {
            "org_name": kb_ctx.get("organisation_name", ""),
            "domain":   kb_ctx.get("domain_description", ""),
        },
    )
    log.info(
        "KnowledgeBaseContext written — org=%r  domain=%r",
        kb_ctx.get("organisation_name", ""),
        kb_ctx.get("domain_description", ""),
    )


# ── Neo4j write helpers ────────────────────────────────────────────────────────

def _write_datasource(client, profile: DatasetProfile) -> None:
    mapping = profile.mapping
    datasource_id = f"source_{profile.file_type}_{mapping.archetype}"
    client.run(
        """
        MERGE (s:DataSource {id: $id})
        SET s.name            = $name,
            s.type            = $type,
            s.schema_detected = $schema_detected
        """,
        {"id": datasource_id, "name": profile.file_path,
         "type": mapping.archetype, "schema_detected": True},
    )


def _write_documents(client, doc_records: list[dict]) -> None:
    log.info("Writing %d Document nodes...", len(doc_records))
    for batch in _batches(doc_records, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d.source_id    = row.source_id,
                d.source_system = row.source_system,
                d.title        = row.title,
                d.doc_type     = row.doc_type,
                d.status       = row.status,
                d.created_at   = row.created_at,
                d.updated_at   = row.updated_at,
                d.ingested_at  = row.ingested_at
            WITH d, row
            MATCH (s:DataSource {id: row.source_id})
            MERGE (d)-[:FROM_SOURCE]->(s)
            """,
            {"rows": batch},
        )


def _write_metadata(client, meta_nodes: list[dict]) -> None:
    log.info("Writing %d Metadata nodes...", len(meta_nodes))
    for batch in _batches(meta_nodes, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $nodes AS m
            MERGE (meta:Metadata {id: m.id})
            SET meta.key        = m.key,
                meta.value      = m.value,
                meta.value_type = m.value_type
            WITH meta, m
            MATCH (d:Document {id: m.doc_id})
            MERGE (d)-[:HAS_METADATA]->(meta)
            """,
            {"nodes": batch},
        )


def _write_chunks(client, all_chunks: list[ChunkData]) -> None:
    log.info("Writing %d Chunk nodes...", len(all_chunks))
    chunk_dicts = [
        {
            "id":             c.id,
            "text":           c.text,
            "chunk_type":     c.chunk_type,
            "chunk_index":    c.chunk_index,
            "ticket_id":      c.ticket_id,
            "context_prefix": c.context_prefix,
            "token_count":    c.token_count,
            "embedding":      c.embedding,
        }
        for c in all_chunks
    ]
    batches = list(_batches(chunk_dicts, config.NEO4J_BATCH_SIZE))
    for batch in tqdm(batches, desc="Writing chunks", unit="batch"):
        client.run(
            """
            UNWIND $chunks AS c
            MERGE (chunk:Chunk {id: c.id})
            SET chunk.text           = c.text,
                chunk.chunk_type     = c.chunk_type,
                chunk.chunk_index    = c.chunk_index,
                chunk.context_prefix = c.context_prefix,
                chunk.token_count    = c.token_count,
                chunk.embedding      = c.embedding
            WITH chunk, c
            MATCH (d:Document {id: c.ticket_id})
            MERGE (chunk)-[:PART_OF]->(d)
            """,
            {"chunks": batch},
        )


def _write_next_chunk_edges(client, all_chunks: list[ChunkData]) -> None:
    log.info("Writing NEXT_CHUNK edges...")
    ticket_chunks: dict[str, list[ChunkData]] = defaultdict(list)
    for c in all_chunks:
        ticket_chunks[c.ticket_id].append(c)

    pairs = []
    for tid, clist in ticket_chunks.items():
        sorted_chunks = sorted(clist, key=lambda c: c.chunk_index)
        for a, b in zip(sorted_chunks, sorted_chunks[1:]):
            rel_type = "problem_resolution" if (
                a.chunk_type == "problem" and b.chunk_type == "resolution"
            ) else "qa" if (
                a.chunk_type == "question" and b.chunk_type == "answer"
            ) else "sequential"
            pairs.append({"src": a.id, "dst": b.id, "relation_type": rel_type})

    for batch in _batches(pairs, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $pairs AS p
            MATCH (a:Chunk {id: p.src})
            MATCH (b:Chunk {id: p.dst})
            MERGE (a)-[:NEXT_CHUNK {relation_type: p.relation_type}]->(b)
            """,
            {"pairs": batch},
        )


def _write_entities(client, entity_map: dict, mentions: list[dict]) -> None:
    entities_list = list(entity_map.values())
    log.info("Writing %d Entity nodes...", len(entities_list))
    for batch in _batches(entities_list, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $entities AS e
            MERGE (entity:Entity {id: e.id})
            SET entity.name       = e.name,
                entity.type       = e.type,
                entity.source     = e.source,
                entity.confidence = e.confidence
            """,
            {"entities": batch},
        )
    log.info("Writing %d MENTIONS edges...", len(mentions))
    for batch in _batches(mentions, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $mentions AS m
            MATCH (c:Chunk  {id: m.chunk_id})
            MATCH (e:Entity {id: m.entity_id})
            MERGE (c)-[:MENTIONS {confidence: m.confidence, method: m.method}]->(e)
            """,
            {"mentions": batch},
        )


def _write_surface_forms(client, entity_map: dict, entity_mode: str) -> None:
    """Write SurfaceForm nodes + ALIASES edges; LLM synonyms only for entity_mode='full'."""
    from etl.ner_v2 import ExtractedEntity  # noqa: PLC0415

    entities = [
        ExtractedEntity(
            id=e["id"], name=e["name"], type=e["type"],
            source=e["source"], confidence=e["confidence"],
        )
        for e in entity_map.values()
    ]

    form_records: list[dict] = []
    for entity in entities:
        for form in generate_surface_forms(entity):
            form_records.append({
                "id":        f"sf_{entity.id}_{re.sub(r'[^a-z0-9]', '_', form)}",
                "form":      form,
                "entity_id": entity.id,
            })

    if entity_mode == "full" and entities:
        log.info("Generating LLM synonyms for %d unique entities...", len(entities))
        synonym_map = generate_synonyms_llm(entities)
        for entity in entities:
            for form in synonym_map.get(entity.id, []):
                form_records.append({
                    "id":        f"sf_{entity.id}_{re.sub(r'[^a-z0-9]', '_', form)}",
                    "form":      form,
                    "entity_id": entity.id,
                })

    seen: set[str] = set()
    unique_forms = []
    for r in form_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique_forms.append(r)

    log.info("Writing %d SurfaceForm nodes...", len(unique_forms))
    for batch in _batches(unique_forms, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $forms AS f
            MERGE (s:SurfaceForm {id: f.id})
            SET s.form = f.form
            WITH s, f
            MATCH (e:Entity {id: f.entity_id})
            MERGE (s)-[:ALIASES]->(e)
            """,
            {"forms": batch},
        )


def _write_concepts(client, concept_map: dict, belongs_edges: list[dict]) -> None:
    concepts_list = list(concept_map.values())
    log.info("Writing %d Concept nodes...", len(concepts_list))
    for batch in _batches(concepts_list, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $concepts AS c
            MERGE (concept:Concept {id: c.id})
            SET concept.label       = c.label,
                concept.description = c.description
            """,
            {"concepts": batch},
        )
    for batch in _batches(belongs_edges, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $edges AS e
            MATCH (ent:Entity  {id: e.entity_id})
            MATCH (con:Concept {id: e.concept_id})
            MERGE (ent)-[:BELONGS_TO {confidence: e.confidence}]->(con)
            """,
            {"edges": batch},
        )


def _write_references_edges(client, edges: list[dict]) -> None:
    if not edges:
        return
    log.info("Writing %d REFERENCES edges (QA → KB)...", len(edges))
    for batch in _batches(edges, config.NEO4J_BATCH_SIZE):
        client.run(
            """
            UNWIND $edges AS e
            MATCH (a:Document {id: e.src_id})
            MATCH (b:Document {id: e.dst_id})
            MERGE (a)-[:REFERENCES]->(b)
            """,
            {"edges": batch},
        )


def _write_sage(client, all_chunks: list[ChunkData]) -> None:
    from etl.sage_builder import build_sage_edges  # noqa: PLC0415
    sage_edges = build_sage_edges(all_chunks)
    if sage_edges:
        log.info("Writing %d SAGE RELATED_TO edges...", len(sage_edges))
        for batch in _batches(sage_edges, config.NEO4J_BATCH_SIZE):
            client.run(
                """
                UNWIND $edges AS e
                MATCH (a:Chunk {id: e.src_id})
                MATCH (b:Chunk {id: e.dst_id})
                MERGE (a)-[:RELATED_TO {score: e.score, method: e.method}]->(b)
                """,
                {"edges": batch},
            )


def _aggregate_known_issues(
    all_chunks: list[ChunkData],
    source_system: str,
    meta_nodes: list[dict] | None = None,
) -> None:
    """Cluster customer_support tickets into KnownIssue nodes after ingest."""
    if source_system != "customer_support":
        return

    from etl.known_issue_builder import find_similar_ticket_cluster, upsert_known_issue  # noqa: PLC0415

    product_by_doc: dict[str, str] = {}
    for m in (meta_nodes or []):
        if m.get("key") == "product" and m.get("value"):
            product_by_doc[m["doc_id"]] = m["value"]

    question_chunks: dict[str, ChunkData] = {}
    resolution_by_doc: dict[str, str]     = {}
    for c in all_chunks:
        if c.chunk_type == "problem" and c.embedding and c.ticket_id not in question_chunks:
            question_chunks[c.ticket_id] = c
        if c.chunk_type == "resolution" and c.ticket_id not in resolution_by_doc:
            resolution_by_doc[c.ticket_id] = c.text

    if not question_chunks:
        log.info("KnownIssue aggregation: no question chunks with embeddings — skipping")
        return

    log.info("KnownIssue aggregation: checking %d tickets for clusters...", len(question_chunks))
    matched = 0

    for doc_id, qchunk in question_chunks.items():
        try:
            product = product_by_doc.get(doc_id)
            cluster = find_similar_ticket_cluster(doc_id, qchunk.embedding, product=product)
            if not cluster:
                continue
            matched += 1
            resolution = resolution_by_doc.get(doc_id, "")
            upsert_known_issue(cluster, doc_id, resolution_notes=resolution)
        except Exception:
            log.exception("KnownIssue aggregation: unexpected error for doc_id=%s — skipping", doc_id)

    log.info("KnownIssue aggregation: %d/%d tickets triggered cluster upsert", matched, len(question_chunks))


def _write_summary(client) -> None:
    log.info("─" * 60)
    log.info("Ingestion complete:")
    for row in client.run(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label"
    ):
        log.info("  %-14s  %s nodes", row["label"], f"{row['cnt']:,}")
    for row in client.run(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY rel"
    ):
        log.info("  [%-16s]  %s edges", row["rel"], f"{row['cnt']:,}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _collect_referenced_ids(
    primary_df: pd.DataFrame,
    ref_field: str,
) -> set[str]:
    """Collect all article/document IDs referenced by the primary dataset's ref_field."""
    ids: set[str] = set()
    for raw in primary_df[ref_field].dropna().astype(str):
        ids.update(_parse_ref_ids(raw))
    return ids


def ingest(
    csv_path: str,
    *,
    secondary_path: str | None = None,
    skip_embed: bool = False,
    skip_sage:  bool = False,
    clear:      bool = False,
    limit:      int | None = None,
    secondary_limit: int | None = None,
    archetype_override: str | None = None,
    secondary_archetype: str | None = None,
    entity_mode: str = "full",
    articles_only: bool = False,
    distractor_sample: int = 0,
    distractor_seed: int = 42,
) -> None:
    os.environ["ENTITY_EXTRACT_MODE"] = entity_mode

    # ── Schema detection ───────────────────────────────────────────────────────
    if secondary_path:
        primary_profile, secondary_profile = detect_paired_schemas(
            csv_path, secondary_path,
            primary_archetype=archetype_override,
            secondary_archetype=secondary_archetype,
        )
    else:
        primary_profile   = detect_schema(csv_path, archetype_override=archetype_override)
        secondary_profile = None

    # ── Load & filter ──────────────────────────────────────────────────────────
    primary_df = _filter_df(load_raw(csv_path), primary_profile.mapping, limit)
    if primary_df.empty:
        log.error("No rows after filtering primary dataset.")
        sys.exit(1)

    secondary_df = None
    if secondary_profile:
        secondary_df = _filter_df(load_raw(secondary_path), secondary_profile.mapping, limit=None)
        if secondary_df.empty:
            log.warning("Secondary dataset is empty after filtering.")

        # ── --articles: restrict secondary to only IDs referenced by primary ──
        if articles_only and secondary_df is not None and not secondary_df.empty:
            ref_field = primary_profile.mapping.ref_field
            id_field  = secondary_profile.mapping.id_field
            if not ref_field:
                log.warning("--articles: no ref_field detected in primary dataset — loading all secondary rows.")
            elif not id_field:
                log.warning("--articles: no id_field detected in secondary dataset — loading all secondary rows.")
            else:
                referenced_ids = _collect_referenced_ids(primary_df, ref_field)
                before = len(secondary_df)
                ids_series = secondary_df[id_field].astype(str).str.strip()
                is_ref = ids_series.isin(referenced_ids)
                gold_df = secondary_df[is_ref]

                if distractor_sample > 0:
                    pool = secondary_df[~is_ref]
                    n_take = min(distractor_sample, len(pool))
                    distractors = pool.sample(n=n_take, random_state=distractor_seed)
                    secondary_df = pd.concat([gold_df, distractors]).reset_index(drop=True)
                    log.info(
                        "--articles + distractors: %d gold + %d distractors (seed=%d) "
                        "= %d secondary rows  (from %d, %d unique IDs referenced)",
                        len(gold_df), len(distractors), distractor_seed,
                        len(secondary_df), before, len(referenced_ids),
                    )
                    if n_take < distractor_sample:
                        log.warning("--distractor-sample %d capped to %d (pool exhausted).",
                                    distractor_sample, n_take)
                else:
                    secondary_df = gold_df.reset_index(drop=True)
                    log.info(
                        "--articles filter: %d → %d secondary rows  (%d unique IDs referenced by primary)",
                        before, len(secondary_df), len(referenced_ids),
                    )
                if secondary_df.empty:
                    log.warning("--articles filter removed all secondary rows — check ref_field values.")
        elif distractor_sample > 0:
            log.warning("--distractor-sample is only applied together with --articles — ignoring.")

        if secondary_limit is not None and secondary_df is not None:
            secondary_df = secondary_df.head(secondary_limit)
            log.info("  Secondary limited to %d rows (--secondary-limit)", secondary_limit)

    # ── Build in-memory structures ─────────────────────────────────────────────
    log.info("Processing primary dataset (%d rows)...", len(primary_df))
    p_chunks, p_docs, p_ents, p_mentions, p_meta, p_concepts, p_belongs = _process_dataset(
        primary_profile, primary_df, entity_mode
    )

    s_chunks, s_docs, s_ents, s_mentions, s_meta, s_concepts, s_belongs = [], [], {}, [], [], {}, []
    if secondary_df is not None:
        log.info("Processing secondary dataset (%d rows)...", len(secondary_df))
        s_chunks, s_docs, s_ents, s_mentions, s_meta, s_concepts, s_belongs = _process_dataset(
            secondary_profile, secondary_df, entity_mode
        )

    all_chunks    = p_chunks + s_chunks
    all_entities  = {**p_ents,    **s_ents}
    all_mentions  = p_mentions  + s_mentions
    all_meta      = p_meta      + s_meta
    all_concepts  = {**p_concepts, **s_concepts}
    all_belongs   = p_belongs   + s_belongs

    log.info(
        "Built: %d docs | %d chunks | %d entities | %d metadata nodes | %d concepts",
        len(p_docs) + len(s_docs), len(all_chunks), len(all_entities),
        len(all_meta), len(all_concepts),
    )

    # ── REFERENCES edges (cross-dataset join) ──────────────────────────────────
    references_edges = []
    if secondary_df is not None and not secondary_df.empty:
        secondary_doc_ids = {
            str(row[secondary_profile.mapping.id_field]).strip()
            for _, row in secondary_df.iterrows()
            if secondary_profile.mapping.id_field
        }
        references_edges = _build_references_edges(
            primary_df, primary_profile.mapping,
            secondary_profile.mapping, secondary_doc_ids,
        )

    # ── Embed ──────────────────────────────────────────────────────────────────
    with Neo4jClient() as client:
        if not client.test_connection():
            sys.exit(1)

        if clear:
            log.info("Clearing database...")
            client.run("MATCH (n) DETACH DELETE n")

        log.info("Applying schema...")
        client.setup_schema()

        if not skip_embed:
            log.info("Embedding %d chunks...", len(all_chunks))
            texts   = [c.embed_text for c in all_chunks]
            vectors = embed_texts(texts)
            for chunk, vec in tqdm(zip(all_chunks, vectors), total=len(all_chunks),
                                   desc="Assigning embeddings", unit="chunk"):
                chunk.embedding = vec
        else:
            log.info("Embedding skipped (--no-embed)")

        # ── KnowledgeBaseContext (org/domain context for retrieval prompts) ───
        kb_ctx = _detect_kb_context([c.text for c in all_chunks])
        _write_kb_context(client, kb_ctx)

        # ── Write primary dataset ──────────────────────────────────────────────
        log.info("Writing primary DataSource...")
        _write_datasource(client, primary_profile)
        _write_documents(client, p_docs)
        _write_metadata(client, p_meta)
        _write_chunks(client, p_chunks)
        _write_next_chunk_edges(client, p_chunks)

        # ── Known-Issue aggregation (customer_support only, non-fatal) ────────
        if not skip_embed:
            primary_source_system = SOURCE_SYSTEM_BY_DATASOURCE.get(
                Path(primary_profile.file_path).stem,
                Path(primary_profile.file_path).stem.lower(),
            )
            _aggregate_known_issues(p_chunks, primary_source_system, meta_nodes=p_meta)

        # ── Write secondary dataset ────────────────────────────────────────────
        if secondary_profile and s_docs:
            log.info("Writing secondary DataSource...")
            _write_datasource(client, secondary_profile)
            _write_documents(client, s_docs)
            _write_metadata(client, s_meta)
            _write_chunks(client, s_chunks)
            _write_next_chunk_edges(client, s_chunks)

        # ── Write shared structures ────────────────────────────────────────────
        _write_entities(client, all_entities, all_mentions)
        _write_surface_forms(client, all_entities, entity_mode)
        _write_concepts(client, all_concepts, all_belongs)
        _write_references_edges(client, references_edges)

        # ── SAGE ──────────────────────────────────────────────────────────────
        if not skip_sage and not skip_embed and config.SAGE_ENABLED:
            _write_sage(client, all_chunks)
        else:
            log.info("SAGE edge construction skipped.")

        _write_summary(client)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline v2 — ingest into Neo4j")

    # Primary file — accept either --csv (legacy) or --primary
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--primary",  default=None, help="Primary input file (CSV/JSON/JSONL/MD)")
    file_group.add_argument("--csv",      default=None, help="Alias for --primary (legacy)")

    parser.add_argument("--secondary",           default=None, help="Secondary/paired input file")
    parser.add_argument("--no-embed",            action="store_true", help="Skip embedding step")
    parser.add_argument("--no-sage",             action="store_true", help="Skip SAGE edge construction")
    parser.add_argument("--clear",               action="store_true", help="Wipe database first")
    parser.add_argument("--limit",               type=int, default=1000, help="Max rows for primary dataset (default 1000)")
    parser.add_argument("--secondary-limit",     type=int, default=None, help="Max rows for secondary dataset (default: all)")
    parser.add_argument("--archetype",           default=None, help="Override primary archetype: ticket|qa|kb_article")
    parser.add_argument("--secondary-archetype", default=None, help="Override secondary archetype")
    parser.add_argument("--entity-mode",         default="full", choices=["full", "fast"],
                        help="full=LLM+regex, fast=regex+metadata only")
    parser.add_argument("--articles",            action="store_true",
                        help="Only ingest secondary (KB) articles referenced by the primary dataset's ref_field")
    parser.add_argument("--distractor-sample",   type=int, default=0,
                        help="With --articles: also ingest N randomly-sampled non-referenced KB rows "
                             "as distractors (keeps a realistic haystack). 0 = gold only.")
    parser.add_argument("--distractor-seed",     type=int, default=42,
                        help="Random seed for --distractor-sample (reproducibility). Default 42.")
    parser.add_argument("--detect-only",         action="store_true", help="Print detected schema(s) and exit")
    args = parser.parse_args()

    primary_path = args.primary or args.csv or config.CSV_PATH

    if args.detect_only:
        import json
        from dataclasses import asdict
        if args.secondary:
            p, s = detect_paired_schemas(primary_path, args.secondary,
                                         primary_archetype=args.archetype,
                                         secondary_archetype=args.secondary_archetype)
            print("=== Primary ===")
            print(json.dumps(asdict(p.mapping), indent=2))
            print("=== Secondary ===")
            print(json.dumps(asdict(s.mapping), indent=2))
        else:
            profile = detect_schema(primary_path, archetype_override=args.archetype)
            print(json.dumps(asdict(profile.mapping), indent=2))
        sys.exit(0)

    ingest(
        primary_path,
        secondary_path       = args.secondary,
        skip_embed           = args.no_embed,
        skip_sage            = args.no_sage,
        clear                = args.clear,
        limit                = args.limit,
        secondary_limit      = args.secondary_limit,
        archetype_override   = args.archetype,
        secondary_archetype  = args.secondary_archetype,
        entity_mode          = args.entity_mode,
        articles_only        = args.articles,
        distractor_sample    = args.distractor_sample,
        distractor_seed      = args.distractor_seed,
    )
