"""Smoke test for BM25 and Graph metadata filter logic.

Usage: python scripts/check_metadata_filters.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.bm25_retriever import bm25_retriever
from agents.graph_retriever import graph_retriever
from agents.semantic_retriever import semantic_retriever

PASS = "✅"
FAIL = "❌"
all_ok = True

def run_test(label: str, filters: dict, expected_fields: dict | None = None, assert_hits: bool = True):
    global all_ok
    print(f"\n--- [ {label} ] ---")
    state = {
        "original_query": "billing issue reset password",
        "expanded_queries": ["billing issue reset password"],
        "metadata_filters": filters,
        "semantic_hits": [], # Graph needs these
    }
    
    # 1. Semantic (just for seeds for Graph)
    sem_res = semantic_retriever(state)
    state["semantic_hits"] = sem_res.get("semantic_hits", [])
    
    # 2. BM25
    bm25_res = bm25_retriever(state)
    bm25_hits = bm25_res.get("bm25_hits", [])
    print(f"BM25 Hits: {len(bm25_hits)}")
    
    # 3. Graph
    graph_res = graph_retriever(state)
    graph_hits = graph_res.get("graph_hits", [])
    print(f"Graph Hits: {len(graph_hits)}")
    
    if assert_hits and (not bm25_hits and not graph_hits):
        print(f"  {FAIL} No hits found but expected hits!")
        all_ok = False
        return
        
    for name, hits in [("BM25", bm25_hits), ("Graph", graph_hits)]:
        if not hits:
            continue
            
        for h in hits:
            meta = h.metadata or {}
            
            # If we expect specific field values
            if expected_fields:
                for k, allowed_vals in expected_fields.items():
                    actual = str(meta.get(k, "")).lower()
                    if actual not in [str(v).lower() for v in allowed_vals]:
                        print(f"  {FAIL} {name} Hit {h.id} failed filter check! {k}={actual} not in {allowed_vals}")
                        all_ok = False
                        
        print(f"  {PASS} {name} valid ({len(hits)} hits). Sample meta keys: {list(hits[0].metadata.keys()) if hits else []}")


print("=" * 60)
print("  METADATA FILTER SMOKE TEST")
print("=" * 60)

# 1. No filters
print("--- [ No Filters ] ---")
state = {
    "original_query": "billing issue reset password",
    "expanded_queries": ["billing issue reset password"],
    "metadata_filters": {},
}
sem_res = semantic_retriever(state)
state["semantic_hits"] = sem_res.get("semantic_hits", [])
bm25_res = bm25_retriever(state)
graph_res = graph_retriever(state)

sample_status = "resolved"
sample_category = "billing"
if bm25_res.get("bm25_hits"):
    meta = bm25_res["bm25_hits"][0].metadata
    sample_status = meta.get("status", "resolved")
    sample_category = meta.get("category", "billing")
    print(f"Top hit actual status: {sample_status}")
    print(f"Top hit actual category: {sample_category}")

# 2. Status
run_test(f"Status = {sample_status}", {"status": [sample_status]}, {"status": [sample_status]})

# 3. Category
run_test(f"Category = {sample_category}", {"category": [sample_category]}, {"category": [sample_category]})

# 4. Status + Category
run_test("Status + Category", {"status": [sample_status], "category": [sample_category]}, 
         {"status": [sample_status], "category": [sample_category]}, assert_hits=False)

# 5. Language filter explicit
run_test("Language explicit", {"language": ["en"]}, {"language": ["en"]}, assert_hits=False)

print("\n" + "=" * 60)
if all_ok:
    print(f"  {PASS} ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(f"  {FAIL} SOME CHECKS FAILED")
    sys.exit(1)
