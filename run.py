"""CLI entrypoint — run a single query through the pipeline.

Usage:
    py run.py "Why is my login failing after the last update?"
    py run.py --query "How do I reset my password?"
"""
import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Silence noisy third-party HTTP / download loggers
for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "sentence_transformers",
    "filelock",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from graph_builder import graph


def run_query(query: str) -> dict:
    initial_state = {
        "original_query": query,
        "expanded_queries": [],
        "intent": "factual",
        "semantic_hits": [],
        "bm25_hits": [],
        "graph_hits": [],
        "web_hits": [],
        "fused_hits": [],
        "reranked_hits": [],
        "react_iterations": 0,
        "react_scratchpad": [],
        "goal_achieved": False,
        "pivot_required": False,
        "prev_sub_queries": [],
        # ReflAct fields (ignored when USE_REFLACT = False in graph_builder)
        "reflact_scratchpad": [],
        "reflection_summary": "",
        "_last_new_hits": [],
        "context_quality": 0.0,
        "crag_verdict": "",
        "crag_retries": 0,
        "crag_missing_hint": "",
        "final_answer": "",
        "cited_chunk_ids": [],
        "confidence": 0.0,
    }
    return graph.invoke(initial_state)


def main():
    parser = argparse.ArgumentParser(description="Agentic Graph-RAG query runner")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--query", "-q", dest="query_flag", help="Query string (alternative)")
    parser.add_argument("--json", action="store_true", help="Output full state as JSON")
    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.print_help()
        sys.exit(1)

    print(f"\nQuery: {query}\n{'─' * 60}")
    result = run_query(query)

    if args.json:
        # Dataclasses are not JSON-serialisable by default — convert manually
        def serialise(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            raise TypeError(f"Not serialisable: {type(obj)}")
        print(json.dumps(result, default=serialise, indent=2, ensure_ascii=False))
    else:
        print(f"\nAnswer:\n{result.get('final_answer', '(no answer)')}")
        print(f"\nConfidence: {result.get('confidence', 0):.2f}")
        cited = result.get("cited_chunk_ids", [])
        if cited:
            print(f"Cited chunks ({len(cited)}):")
            for chunk_id in cited:
                print(f"  • {chunk_id}")


if __name__ == "__main__":
    main()
