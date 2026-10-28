"""Debug script to test context_prefix v2 output for different scenarios."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl.chunker_v3 import build_context_prefix_v3

class DummySection:
    def __init__(self, heading, heading_path):
        self.heading = heading
        self.heading_path = heading_path

def test_prefix():
    print("Testing context_prefix v3...\n")
    
    cases = [
        {
            "name": "ticket problem",
            "row": {"source_system": "customer_support", "status": "closed", "priority": "high", "category": "billing"},
            "doc_type": "ticket",
            "chunk_type": "problem",
        },
        {
            "name": "ticket body",
            "row": {"source_system": "customer_support", "status": "open", "priority": "low", "category": "tech support", "product": "Cloud", "severity": "minor"},
            "doc_type": "ticket",
            "chunk_type": "body",
        },
        {
            "name": "qa question",
            "row": {"source_system": "wixqa", "title": "Account recovery"},
            "doc_type": "qa",
            "chunk_type": "question",
        },
        {
            "name": "document section",
            "row": {"source_system": "paperless", "title": "Employee Handbook", "document_type": "policy", "tags": ["HR", "Internal"]},
            "doc_type": "document",
            "chunk_type": "section",
            "section": DummySection("Leave Policy", "Policies > Leave Policy")
        },
        {
            "name": "section_summary",
            "row": {"source_system": "wixqa", "title": "Account recovery guide"},
            "doc_type": "document",
            "chunk_type": "section_summary",
            "section": DummySection("Steps", "Main > Steps")
        },
        {
            "name": "document_summary",
            "row": {"source_system": "wixqa", "title": "Account recovery guide"},
            "doc_type": "document",
            "chunk_type": "document_summary",
        }
    ]
    
    for case in cases:
        print(f"--- {case['name'].upper()} ---")
        print(f"doc_type: {case['doc_type']}, chunk_type: {case['chunk_type']}")
        prefix = build_context_prefix_v3(case["row"], case["doc_type"], case["chunk_type"], doc_id="TEST-123", section=case.get("section"))
        print(f"Prefix:\n{prefix}")
        print(f"Embed preview:\n{prefix}\n\n<chunk text>\n")

if __name__ == "__main__":
    test_prefix()
