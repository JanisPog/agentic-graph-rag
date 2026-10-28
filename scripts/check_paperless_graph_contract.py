"""Script to verify that the Paperless ingestion graph contract is fulfilled."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.neo4j_client import Neo4jClient

def check_paperless_graph():
    with Neo4jClient() as client:
        print("Checking Paperless graph contract...\n")
        
        # 1. Documents with/without ingested_at
        res = client.run("""
            MATCH (d:Document)
            WHERE d.id STARTS WITH 'paperless_'
            RETURN 
                count(d) AS total, 
                sum(CASE WHEN d.ingested_at IS NOT NULL THEN 1 ELSE 0 END) AS with_ingested_at,
                sum(CASE WHEN d.ingested_at IS NULL THEN 1 ELSE 0 END) AS without_ingested_at
        """)
        row = res[0]
        print(f"Paperless Documents:")
        print(f"  Total: {row['total']}")
        print(f"  With ingested_at: {row['with_ingested_at']}")
        print(f"  Without ingested_at: {row['without_ingested_at']}")
        
        # 2. section_summary chunks
        res = client.run("""
            MATCH (c:Chunk {chunk_type: 'section_summary'})
            WHERE c.id STARTS WITH 'paperless_'
            RETURN count(c) AS total
        """)
        total_section_summary = res[0]["total"]
        print(f"\nSection Summary Chunks (Paperless):")
        print(f"  Total: {total_section_summary}")
        
        # 3. PART_OF edges for section_summary chunks
        res = client.run("""
            MATCH (c:Chunk {chunk_type: 'section_summary'})
            WHERE c.id STARTS WITH 'paperless_'
            RETURN 
                sum(CASE WHEN EXISTS((c)-[:PART_OF]->(:Section)) THEN 1 ELSE 0 END) AS part_of_section,
                sum(CASE WHEN EXISTS((c)-[:PART_OF]->(:Document)) THEN 1 ELSE 0 END) AS part_of_document
        """)
        row = res[0]
        print(f"  With PART_OF -> Section: {row['part_of_section']}")
        print(f"  With PART_OF -> Document: {row['part_of_document']}")
        
        # 4. Chunk-Nodes ohne PART_OF Document (alle Typen)
        res = client.run("""
            MATCH (c:Chunk)
            WHERE c.id STARTS WITH 'paperless_' AND NOT EXISTS((c)-[:PART_OF]->(:Document))
            RETURN count(c) AS missing_doc_edge, collect(DISTINCT c.chunk_type) AS types
        """)
        row = res[0]
        print(f"\nAll Paperless Chunks missing direct PART_OF -> Document edge:")
        print(f"  Count: {row['missing_doc_edge']} (Types: {row['types']})")
        
        # 5. Metadata nodes with source_system key on Paperless docs
        res = client.run("""
            MATCH (d:Document)-[:HAS_METADATA]->(m:Metadata {key: 'source_system'})
            WHERE d.id STARTS WITH 'paperless_'
            RETURN count(m) AS total
        """)
        print(f"\nMetadata nodes with key='source_system' on Paperless docs:")
        print(f"  Total: {res[0]['total']} (Should be 0 if cleaned up)")
        
if __name__ == "__main__":
    check_paperless_graph()
