// schema.cypher v2 — Index and constraint definitions for the v2 graph schema.
//
// WARNING: Do NOT apply this file directly to Neo4j for production use.
//          The vector index dimension is written here for documentation only.
//          The authoritative value lives in config.py (EMBED_DIM) and is
//          applied at runtime by neo4j_client.setup_schema().
//          Applying this file directly may create a mismatched index.
//
// Current dimension values by backend:
//   qwen3:8b (Ollama, default) → 4096 dims  (OLLAMA_EMBED_DIM in .env)
//   nomic-embed-text (Ollama)  → 768 dims
//   gemini-embedding-001       → 768 dims   (GEMINI_EMBED_DIM in config.py)

// ── Uniqueness constraints ─────────────────────────────────────────────────────

CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT metadata_id_unique IF NOT EXISTS
FOR (m:Metadata) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

// ── Vector index ───────────────────────────────────────────────────────────────
// Adjust vector.dimensions to match your embedding model (see note above).

// Dimension shown here is for qwen3:8b (Ollama default).
// The actual value applied at runtime is config.EMBED_DIM.
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 4096,
  `vector.similarity_function`: 'cosine'
}};

// ── Full-text index ────────────────────────────────────────────────────────────
// Includes context_prefix so BM25 benefits from Contextual Retrieval enrichment.

CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text, c.context_prefix];

// ── Composite and B-tree indexes ──────────────────────────────────────────────

CREATE INDEX chunk_hop_index IF NOT EXISTS
FOR (c:Chunk) ON (c.id, c.chunk_index);

CREATE INDEX metadata_lookup IF NOT EXISTS
FOR (m:Metadata) ON (m.key, m.value);

CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity) ON (e.name);

// ── Cross-dataset reference index ─────────────────────────────────────────────
// Supports fast traversal of REFERENCES edges (e.g. QA Document → KB Article).
CREATE INDEX doc_type_index IF NOT EXISTS
FOR (d:Document) ON (d.doc_type);

// ── KnownIssue — aggregated cluster node (TODO 2, Memory Integration) ─────────
// Created by etl/known_issue_builder.py during customer_support ingest when
// ≥3 tickets share a problem_embedding cosine similarity ≥ 0.85.
//
// (:Document)-[:SAME_ISSUE {similarity_score}]->(:KnownIssue)
//
// WARNING: problem_embedding dimension must match EMBED_DIM (see top of file).

CREATE CONSTRAINT known_issue_id IF NOT EXISTS
FOR (k:KnownIssue) REQUIRE k.id IS UNIQUE;

CREATE VECTOR INDEX known_issue_embedding IF NOT EXISTS
FOR (k:KnownIssue) ON (k.problem_embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 4096,
  `vector.similarity_function`: 'cosine'
}};

CREATE INDEX known_issue_ticket_count IF NOT EXISTS
FOR (k:KnownIssue) ON (k.ticket_count);
