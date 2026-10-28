"""Paperless ETL orchestrator — ingests PDF documents from Paperless-ngx into Neo4j.

Usage::
    py etl/paperless_ingest.py --clear --limit 50
    py etl/paperless_ingest.py --no-enrich --no-embed
    py etl/paperless_ingest.py --tag "wix-kb" --limit 100
    py etl/paperless_ingest.py --id 42
    py etl/paperless_ingest.py --detect-only --limit 10
    py etl/paperless_ingest.py --no-incremental --clear
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import config
from etl.paperless_client import PaperlessClient, PaperlessDocument
from etl.pdf_parser import parse_pdf
from etl.doc_enrichment import (
    enrich_document, build_chunk_lineage,
    SectionSummaryRecord, DocumentSummaryRecord,
)
from etl.chunker_v3 import ChunkData, _split_long_text, _build_prefix_document
from etl.schema_detector import ColumnMapping
from etl.ner_v2 import extract_entities, ExtractedEntity, _extract_regex, _extract_llm, _dedup
from etl.metadata_builder import build_concept_nodes, build_belongs_to_edges
from etl.embedder import embed_texts
from graph.neo4j_client import Neo4jClient
from agents.llm import chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Metadata builder for Paperless documents ──────────────────────────────────

def _build_paperless_metadata(doc_id: str, pdoc: PaperlessDocument) -> list[dict]:
    """Build Metadata node dicts from Paperless fields (auto-discovery)."""
    nodes = []
    def _add(key, val, vtype="str"):
        if val and str(val).strip() and str(val).lower() not in ("nan", "none", ""):
            nodes.append({"id": f"meta_{doc_id}_{key}", "doc_id": doc_id,
                          "key": key, "value": str(val).strip(), "value_type": vtype})
    _add("correspondent", pdoc.correspondent)
    _add("document_type", pdoc.document_type)
    for i, tag in enumerate(pdoc.tags):
        _add(f"tag_{i}", tag)
    _add("created", pdoc.created, "date")
    _add("modified", pdoc.modified, "date")
    for k, v in pdoc.custom_fields.items():
        _add(k, v)
    return nodes


# ── Entity extraction for document text ───────────────────────────────────────

def _extract_doc_entities(full_text: str, entity_mode: str) -> list[ExtractedEntity]:
    entities = list(_extract_regex(full_text))
    if entity_mode == "full" and full_text.strip():
        entities.extend(_extract_llm(full_text[:2000]))
    return _dedup(entities)


# ── Neo4j write helpers ──────────────────────────────────────────────────────

def _write_datasource(client, datasource_id: str):
    client.run("""
        MERGE (s:DataSource {id: $id})
        SET s.name = 'paperless-ngx', s.type = 'document', s.schema_detected = true
    """, {"id": datasource_id})

def _write_documents(client, docs):
    log.info("Writing %d Document nodes...", len(docs))
    for batch in _batches(docs, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d.source_id = row.source_id, d.title = row.title,
                d.source_system = row.source_system,
                d.doc_type = row.doc_type, d.status = row.status,
                d.created_at = row.created_at, d.updated_at = row.updated_at,
                d.ingested_at = row.ingested_at
            WITH d, row
            MATCH (s:DataSource {id: row.source_id})
            MERGE (d)-[:FROM_SOURCE]->(s)
        """, {"rows": batch})

def _write_metadata(client, meta_nodes):
    log.info("Writing %d Metadata nodes...", len(meta_nodes))
    for batch in _batches(meta_nodes, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $nodes AS m
            MERGE (meta:Metadata {id: m.id})
            SET meta.key = m.key, meta.value = m.value, meta.value_type = m.value_type
            WITH meta, m
            MATCH (d:Document {id: m.doc_id})
            MERGE (d)-[:HAS_METADATA]->(meta)
        """, {"nodes": batch})

def _write_sections(client, sections):
    log.info("Writing %d Section nodes...", len(sections))
    for batch in _batches(sections, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $sections AS s
            MERGE (sec:Section {id: s.id})
            SET sec.doc_id = s.doc_id, sec.heading = s.heading,
                sec.heading_path = s.heading_path, sec.level = s.level,
                sec.position = s.position
            WITH sec, s
            MATCH (d:Document {id: s.doc_id})
            MERGE (sec)-[:PART_OF {position: s.position}]->(d)
        """, {"sections": batch})

def _write_chunks(client, all_chunks):
    log.info("Writing %d Chunk nodes...", len(all_chunks))
    chunk_dicts = [{
        "id": c.id, "text": c.text, "chunk_type": c.chunk_type,
        "chunk_index": c.chunk_index, "ticket_id": c.ticket_id,
        "context_prefix": c.context_prefix, "token_count": c.token_count,
        "embedding": c.embedding, "lineage": c.lineage,
    } for c in all_chunks]
    for batch in tqdm(list(_batches(chunk_dicts, config.NEO4J_BATCH_SIZE)), desc="Writing chunks"):
        client.run("""
            UNWIND $chunks AS c
            MERGE (chunk:Chunk {id: c.id})
            SET chunk.text = c.text, chunk.chunk_type = c.chunk_type,
                chunk.chunk_index = c.chunk_index,
                chunk.context_prefix = c.context_prefix,
                chunk.token_count = c.token_count,
                chunk.embedding = c.embedding, chunk.lineage = c.lineage
            WITH chunk, c
            MATCH (d:Document {id: c.ticket_id})
            MERGE (chunk)-[:PART_OF]->(d)
        """, {"chunks": batch})

def _write_chunk_section_edges(client, all_chunks, section_map):
    """Link chunks to their parent Section nodes."""
    edges = []
    for c in all_chunks:
        if c.chunk_type == "section" and c.lineage:
            parts = c.lineage.split(" > ")
            if len(parts) >= 2:
                doc_part = parts[0].replace("Document:", "")
                sec_part = parts[1].replace("Section:", "")
                # Find section by heading match
                for sid, sdata in section_map.items():
                    if sdata["doc_id"] == doc_part and sdata["heading"] == sec_part:
                        edges.append({"chunk_id": c.id, "section_id": sid})
                        break
    if edges:
        log.info("Writing %d Chunk→Section PART_OF edges...", len(edges))
        for batch in _batches(edges, config.NEO4J_BATCH_SIZE):
            client.run("""
                UNWIND $edges AS e
                MATCH (c:Chunk {id: e.chunk_id})
                MATCH (s:Section {id: e.section_id})
                MERGE (c)-[:PART_OF]->(s)
            """, {"edges": batch})

def _write_section_summaries(client, summaries):
    if not summaries:
        return
    log.info("Writing %d section summary Chunk nodes...", len(summaries))
    chunk_dicts = [{
        "id": s.id, "text": s.text, "context_prefix": s.context_prefix,
        "token_count": s.token_count, "embedding": s.embedding,
        "lineage": s.lineage, "section_id": s.section_id,
    } for s in summaries]
    for batch in _batches(chunk_dicts, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $chunks AS c
            MERGE (chunk:Chunk {id: c.id})
            SET chunk.text = c.text, chunk.chunk_type = 'section_summary',
                chunk.chunk_index = -1, chunk.context_prefix = c.context_prefix,
                chunk.token_count = c.token_count, chunk.embedding = c.embedding,
                chunk.lineage = c.lineage
            WITH chunk, c
            MATCH (s:Section {id: c.section_id})
            MATCH (s)-[:PART_OF]->(d:Document)
            MERGE (chunk)-[:PART_OF]->(s)
            MERGE (chunk)-[:PART_OF]->(d)
        """, {"chunks": batch})

def _write_document_summaries(client, summaries):
    if not summaries:
        return
    log.info("Writing %d document summary Chunk nodes...", len(summaries))
    chunk_dicts = [{
        "id": s.id, "text": s.text, "context_prefix": s.context_prefix,
        "token_count": s.token_count, "embedding": s.embedding,
        "lineage": s.lineage, "doc_id": s.doc_id,
    } for s in summaries]
    for batch in _batches(chunk_dicts, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $chunks AS c
            MERGE (chunk:Chunk {id: c.id})
            SET chunk.text = c.text, chunk.chunk_type = 'document_summary',
                chunk.chunk_index = -1, chunk.context_prefix = c.context_prefix,
                chunk.token_count = c.token_count, chunk.embedding = c.embedding,
                chunk.lineage = c.lineage
            WITH chunk, c
            MATCH (d:Document {id: c.doc_id})
            MERGE (chunk)-[:PART_OF]->(d)
        """, {"chunks": batch})

def _write_next_chunk_edges(client, all_chunks):
    log.info("Writing NEXT_CHUNK edges...")
    doc_chunks = defaultdict(list)
    for c in all_chunks:
        doc_chunks[c.ticket_id].append(c)
    pairs = []
    for clist in doc_chunks.values():
        sorted_c = sorted(clist, key=lambda c: c.chunk_index)
        for a, b in zip(sorted_c, sorted_c[1:]):
            pairs.append({"src": a.id, "dst": b.id, "relation_type": "sequential"})
    for batch in _batches(pairs, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $pairs AS p
            MATCH (a:Chunk {id: p.src}) MATCH (b:Chunk {id: p.dst})
            MERGE (a)-[:NEXT_CHUNK {relation_type: p.relation_type}]->(b)
        """, {"pairs": batch})

def _write_entities(client, entity_map, mentions):
    entities_list = list(entity_map.values())
    log.info("Writing %d Entity nodes...", len(entities_list))
    for batch in _batches(entities_list, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $entities AS e
            MERGE (entity:Entity {id: e.id})
            SET entity.name = e.name, entity.type = e.type,
                entity.source = e.source, entity.confidence = e.confidence
        """, {"entities": batch})
    log.info("Writing %d MENTIONS edges...", len(mentions))
    for batch in _batches(mentions, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $mentions AS m
            MATCH (c:Chunk {id: m.chunk_id}) MATCH (e:Entity {id: m.entity_id})
            MERGE (c)-[:MENTIONS {confidence: m.confidence, method: m.method}]->(e)
        """, {"mentions": batch})

def _write_concepts(client, concept_map, belongs_edges):
    concepts_list = list(concept_map.values())
    log.info("Writing %d Concept nodes...", len(concepts_list))
    for batch in _batches(concepts_list, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $concepts AS c
            MERGE (concept:Concept {id: c.id})
            SET concept.label = c.label, concept.description = c.description
        """, {"concepts": batch})
    for batch in _batches(belongs_edges, config.NEO4J_BATCH_SIZE):
        client.run("""
            UNWIND $edges AS e
            MATCH (ent:Entity {id: e.entity_id}) MATCH (con:Concept {id: e.concept_id})
            MERGE (ent)-[:BELONGS_TO {confidence: e.confidence}]->(con)
        """, {"edges": batch})

def _write_sage(client, all_chunks):
    from etl.sage_builder import build_sage_edges
    sage_edges = build_sage_edges(all_chunks)
    if sage_edges:
        log.info("Writing %d SAGE RELATED_TO edges...", len(sage_edges))
        for batch in _batches(sage_edges, config.NEO4J_BATCH_SIZE):
            client.run("""
                UNWIND $edges AS e
                MATCH (a:Chunk {id: e.src_id}) MATCH (b:Chunk {id: e.dst_id})
                MERGE (a)-[:RELATED_TO {score: e.score, method: e.method}]->(b)
            """, {"edges": batch})

def _write_summary(client):
    log.info("─" * 60)
    log.info("Ingestion complete:")
    for row in client.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label"):
        log.info("  %-14s  %s nodes", row["label"], f"{row['cnt']:,}")
    for row in client.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY rel"):
        log.info("  [%-16s]  %s edges", row["rel"], f"{row['cnt']:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_paperless(
    *, skip_embed=False, skip_sage=False, skip_enrich=False,
    skip_summaries=False, clear=False, limit=None,
    entity_mode="full", tag_filter=None, doc_type_filter=None,
    single_id=None, incremental=True, detect_only=False,
):
    os.environ["ENTITY_EXTRACT_MODE"] = entity_mode
    datasource_id = "source_document_paperless"

    # 1. Connect to Paperless
    client = PaperlessClient()
    if single_id:
        doc_list = [client.get_document(single_id)]
    else:
        doc_list = client.list_documents(
            tag_filter=tag_filter, doc_type_filter=doc_type_filter, limit=limit,
        )

    if not doc_list:
        log.error("No documents found in Paperless.")
        return

    # Incremental: filter out already-ingested docs
    if incremental and not clear and not single_id:
        existing_ids = client.get_ingested_ids()
        if existing_ids:
            before = len(doc_list)
            doc_list = [d for d in doc_list if d["id"] not in existing_ids]
            log.info("Incremental: %d already ingested, %d new → processing %d",
                     len(existing_ids), before - len(existing_ids) + len(doc_list), len(doc_list))
            if not doc_list:
                log.info("All documents already ingested. Nothing to do.")
                return

    if detect_only:
        log.info("Detect-only mode: %d documents would be processed", len(doc_list))
        for d in doc_list[:10]:
            log.info("  id=%d  title=%s", d["id"], d.get("title", "?")[:60])
        return

    # 2. Process each document
    all_chunks: list[ChunkData] = []
    all_docs: list[dict] = []
    all_sections: list[dict] = []
    all_section_summaries: list[SectionSummaryRecord] = []
    all_doc_summaries: list[DocumentSummaryRecord] = []
    all_meta: list[dict] = []
    entity_map: dict[str, dict] = {}
    mentions: list[dict] = []
    concept_map: dict[str, dict] = {}
    belongs_edges: list[dict] = []
    section_map: dict[str, dict] = {}  # section_id → section dict

    for raw_doc in tqdm(doc_list, desc="Processing documents", unit="doc"):
        pdoc = client.build_paperless_document(raw_doc)
        doc_id = f"paperless_{pdoc.paperless_id}"

        # Parse PDF
        if pdoc.pdf_bytes:
            title, sections = parse_pdf(pdoc.pdf_bytes)
        else:
            log.warning("No PDF for doc %d, using OCR content", pdoc.paperless_id)
            title = pdoc.title
            sections = []

        if not title:
            title = pdoc.title

        resolved_source_system = pdoc.custom_fields.get("source_system", "paperless")

        # Document record
        all_docs.append({
            "id": doc_id, "source_id": datasource_id,
            "source_system": resolved_source_system,
            "title": title, "doc_type": "document", "status": "",
            "created_at": pdoc.created, "updated_at": pdoc.modified,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })

        # Enrichment
        if not skip_enrich and sections:
            enrichment = enrich_document(
                doc_id=doc_id, title=title, sections=sections,
                metadata={"document_type": pdoc.document_type, **pdoc.custom_fields},
                llm_fn=chat, skip_summaries=skip_summaries,
            )
            if not skip_summaries:
                from etl.chunker_v3 import build_context_prefix_v3
                prefix_row = {"title": title, "source_system": "paperless", "document_type": pdoc.document_type, "tags": pdoc.tags, **pdoc.custom_fields}
                
                for s in enrichment.section_summaries:
                    sec_obj = next((sec for sec in sections if f"{doc_id}_sec_{sec.position}" == s.section_id), None)
                    s.context_prefix = build_context_prefix_v3(prefix_row, "document", "section_summary", doc_id, sec_obj)
                all_section_summaries.extend(enrichment.section_summaries)
                
                if enrichment.document_summary:
                    enrichment.document_summary.context_prefix = build_context_prefix_v3(prefix_row, "document", "document_summary", doc_id)
                    all_doc_summaries.append(enrichment.document_summary)

        # Section records + chunks
        doc_mapping = ColumnMapping(
            archetype="document", id_field=None, question_field=None,
            answer_field=None, title_field=None,
        )
        for section in sections:
            section_id = f"{doc_id}_sec_{section.position}"
            sec_dict = {
                "id": section_id, "doc_id": doc_id,
                "heading": section.heading, "heading_path": section.heading_path,
                "level": section.level, "position": section.position,
            }
            all_sections.append(sec_dict)
            section_map[section_id] = sec_dict

            # Build chunks for this section
            section_text = "\n\n".join(
                b.text for b in section.blocks if b.block_type in ("paragraph", "list_item")
            )
            if not section_text.strip():
                continue

            idx = len(all_chunks)
            for part in _split_long_text(section_text):
                if not part.strip():
                    continue
                from etl.chunker_v3 import build_context_prefix_v3
                prefix_row = {"title": title, "source_system": "paperless", "document_type": pdoc.document_type, "tags": pdoc.tags, **pdoc.custom_fields}
                prefix = build_context_prefix_v3(prefix_row, "document", "section", doc_id, section)
                lineage = build_chunk_lineage(doc_id, section.heading, len(all_chunks))
                all_chunks.append(ChunkData(
                    id=f"{doc_id}_{len(all_chunks)}", text=part,
                    embed_text=prefix + "\n\n" + part, chunk_type="section",
                    chunk_index=len(all_chunks), ticket_id=doc_id,
                    context_prefix=prefix, token_count=len(part.split()),
                    lineage=lineage,
                ))

        # Metadata
        all_meta.extend(_build_paperless_metadata(doc_id, pdoc))

        # NER
        full_text = " ".join(
            b.text for s in sections for b in s.blocks if b.block_type in ("paragraph", "list_item")
        )
        if not full_text and pdoc.content:
            full_text = pdoc.content[:5000]

        entities = _extract_doc_entities(full_text, entity_mode)
        for e in entities:
            entity_map[e.id] = {"id": e.id, "name": e.name, "type": e.type,
                                "source": e.source, "confidence": e.confidence}
            for chunk in all_chunks[-20:]:  # link to recent chunks from this doc
                if chunk.ticket_id == doc_id:
                    mentions.append({"chunk_id": chunk.id, "entity_id": e.id,
                                     "confidence": e.confidence, "method": e.source})
        for c in build_concept_nodes(entities):
            concept_map[c["id"]] = c
        belongs_edges.extend(build_belongs_to_edges(entities))

    log.info("Built: %d docs | %d sections | %d chunks | %d entities | %d meta",
             len(all_docs), len(all_sections), len(all_chunks), len(entity_map), len(all_meta))

    # 3. Embed
    if not skip_embed and all_chunks:
        log.info("Embedding %d chunks...", len(all_chunks))
        texts = [c.embed_text for c in all_chunks]
        vectors = embed_texts(texts)
        for chunk, vec in zip(all_chunks, vectors):
            chunk.embedding = vec

        if all_section_summaries:
            log.info("Embedding %d section summaries...", len(all_section_summaries))
            stexts = [s.context_prefix + "\n\n" + s.text for s in all_section_summaries]
            svecs = embed_texts(stexts)
            for s, v in zip(all_section_summaries, svecs):
                s.embedding = v

        if all_doc_summaries:
            log.info("Embedding %d document summaries...", len(all_doc_summaries))
            dtexts = [s.context_prefix + "\n\n" + s.text for s in all_doc_summaries]
            dvecs = embed_texts(dtexts)
            for s, v in zip(all_doc_summaries, dvecs):
                s.embedding = v

    # 4. Write to Neo4j
    with Neo4jClient() as neo4j:
        if not neo4j.test_connection():
            sys.exit(1)
        if clear:
            log.info("Clearing database...")
            neo4j.run("MATCH (n) DETACH DELETE n")
        neo4j.setup_schema()

        _write_datasource(neo4j, datasource_id)
        _write_documents(neo4j, all_docs)
        _write_document_summaries(neo4j, all_doc_summaries)
        _write_metadata(neo4j, all_meta)
        _write_sections(neo4j, all_sections)
        _write_section_summaries(neo4j, all_section_summaries)
        _write_chunks(neo4j, all_chunks)
        _write_chunk_section_edges(neo4j, all_chunks, section_map)
        _write_next_chunk_edges(neo4j, all_chunks)
        _write_entities(neo4j, entity_map, mentions)
        
        # P1 fix: Add SurfaceForm nodes + ALIASES edges for Paperless entities
        from etl.ingest import _write_surface_forms
        _write_surface_forms(neo4j, entity_map, entity_mode)
        
        _write_concepts(neo4j, concept_map, belongs_edges)

        if not skip_sage and not skip_embed and config.SAGE_ENABLED:
            _write_sage(neo4j, all_chunks)
        else:
            log.info("SAGE edge construction skipped.")

        _write_summary(neo4j)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paperless ETL — ingest PDFs into Neo4j")
    p.add_argument("--clear", action="store_true", help="Wipe database first")
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")
    p.add_argument("--id", type=int, default=None, help="Single Paperless document ID")
    p.add_argument("--tag", default=None, help="Filter by Paperless tag name")
    p.add_argument("--doc-type", default=None, help="Filter by Paperless document type")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding")
    p.add_argument("--no-sage", action="store_true", help="Skip SAGE edges")
    p.add_argument("--no-enrich", action="store_true", help="Skip all enrichment")
    p.add_argument("--no-summaries", action="store_true", help="Skip summaries only")
    p.add_argument("--no-incremental", action="store_true", help="Disable incremental mode")
    p.add_argument("--entity-mode", default="full", choices=["full", "fast"])
    p.add_argument("--detect-only", action="store_true", help="List docs and exit")
    args = p.parse_args()

    ingest_paperless(
        skip_embed=args.no_embed, skip_sage=args.no_sage,
        skip_enrich=args.no_enrich, skip_summaries=args.no_summaries,
        clear=args.clear, limit=args.limit,
        entity_mode=args.entity_mode, tag_filter=args.tag,
        doc_type_filter=args.doc_type, single_id=args.id,
        incremental=not args.no_incremental, detect_only=args.detect_only,
    )
