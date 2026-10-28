"""Download WixQA datasets from the Hugging Face Hub into ``data/``.

Normalises every QA subset to the schema already used by
``data/wixqa_expertwritten.csv`` so the existing evaluation runner can consume
any subset without changes::

    question, answer, article_ids

``article_ids`` is written as a Python list-literal string (e.g. ``['ab12...']``)
to match the existing file and the parser in ``evaluate/eval.py``.

Subsets (friendly name → tried Hugging Face configs):
    expertwritten  → expert_written / expertwritten / WixQA-ExpertWritten
    simulated      → simulated / WixQA-Simulated
    synthetic      → synthetic / WixQA-Synthetic

The KB corpus (``--kb``) is normalised to the schema used by
``data/wix_kb_corpus.csv``::

    id, url, contents, title, html_content, article_type

Usage::

    py etl/download_wixqa.py --subset synthetic
    py etl/download_wixqa.py --subset expertwritten --force
    py etl/download_wixqa.py --kb                       # (re)download KB corpus
    py etl/download_wixqa.py --subset synthetic --kb    # both in one go
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_wixqa")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

HF_REPO = "Wix/WixQA"

# Friendly subset name → ordered list of candidate HF config names to try.
# The live Wix/WixQA repo uses the ``wixqa_`` prefix; older mirrors used other
# spellings, so several candidates are tried in order.
_SUBSET_CONFIGS: dict[str, list[str]] = {
    "expertwritten": ["wixqa_expertwritten", "expert_written", "WixQA-ExpertWritten"],
    "simulated":     ["wixqa_simulated", "simulated", "WixQA-Simulated"],
    "synthetic":     ["wixqa_synthetic", "synthetic", "WixQA-Synthetic"],
}

# Candidate HF configs / repos for the KB corpus.
_KB_CONFIGS = ["wix_kb_corpus", "kb", "documents", "corpus"]
_KB_REPO_FALLBACKS = ["Wix/wix-kb", "Wix/WixKB"]

# Column-name detection (case-insensitive) for QA subsets.
_QUESTION_KEYS = ["question", "query", "user_query", "user_question"]
_ANSWER_KEYS   = ["answer", "response", "gold_answer", "ground_truth"]
_ARTICLE_KEYS  = [
    "article_ids", "article_id", "source_article_ids", "relevant_article_ids",
    "doc_ids", "document_ids", "source_ids", "article_urls", "source_urls", "urls",
]

# Column detection for the KB corpus.
_KB_ID_KEYS       = ["id", "article_id", "doc_id", "_id"]
_KB_URL_KEYS      = ["url", "link", "source_url"]
_KB_CONTENT_KEYS  = ["contents", "content", "text", "body", "article_text"]
_KB_TITLE_KEYS    = ["title", "name", "heading"]
_KB_HTML_KEYS     = ["html_content", "html", "raw_html"]
_KB_TYPE_KEYS     = ["article_type", "type", "category", "doc_type"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick(colnames: list[str], candidates: list[str]) -> str | None:
    """Return the first dataset column matching any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _normalise_article_ids(value, url_to_id: dict[str, str]) -> str:
    """Coerce an article reference into a Python list-literal string of KB ids.

    Accepts: a list, a single id/url string, or a list-literal string. URLs are
    mapped back to KB ids via ``url_to_id`` when a mapping is available.
    """
    if value is None:
        return "[]"

    # Already a list (HF often returns sequence columns as Python lists).
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        s = str(value).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return "[]"
        if s.startswith("["):
            # list-literal string → parse leniently
            import ast
            try:
                parsed = ast.literal_eval(s)
                items = [str(v).strip() for v in parsed if str(v).strip()]
            except (ValueError, SyntaxError):
                items = [s]
        else:
            items = [s]

    # Map URLs → KB ids where possible (keeps grounding aligned with Document.id).
    mapped: list[str] = []
    for it in items:
        if it.startswith("http") and url_to_id:
            mapped.append(url_to_id.get(it.rstrip("/"), it))
        else:
            mapped.append(it)
    return repr(mapped)


def _load_url_to_id_map() -> dict[str, str]:
    """Build a url→id map from an existing KB corpus CSV, if present."""
    kb_path = DATA_DIR / "wix_kb_corpus.csv"
    if not kb_path.exists():
        return {}
    mapping: dict[str, str] = {}
    try:
        with open(kb_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip().rstrip("/")
                _id = (row.get("id") or "").strip()
                if url and _id:
                    mapping[url] = _id
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read KB corpus for url→id mapping: %s", exc)
    if mapping:
        log.info("Loaded url→id map from existing KB corpus (%d entries)", len(mapping))
    return mapping


def _load_hf_config(repo: str, candidates: list[str]):
    """Try each candidate config name; return (dataset, config_name) or raise."""
    from datasets import get_dataset_config_names, load_dataset

    available: list[str] = []
    try:
        available = get_dataset_config_names(repo)
        log.info("Available configs for %s: %s", repo, available)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not list configs for %s (%s) — trying candidates blind", repo, exc)

    # Safe fallback: only consider *available* configs that share a token with a
    # requested candidate (e.g. "synthetic"). Never load an arbitrary config —
    # that previously caused the wrong subset to be written silently.
    keywords = {tok for cand in candidates for tok in cand.lower().replace("-", "_").split("_") if tok}
    keywords.discard("wixqa")
    keywords.discard("wix")
    extra = [
        c for c in available
        if c not in candidates and any(kw in c.lower() for kw in keywords)
    ]

    tried: list[str] = []
    for cfg in candidates + extra:
        tried.append(cfg)
        try:
            ds = load_dataset(repo, cfg, split="train")
            log.info("Loaded %s config=%r  (%d rows)", repo, cfg, len(ds))
            return ds, cfg
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        f"None of the candidate configs matched for {repo}. Tried: {tried}. "
        f"Available: {available or 'unknown'}"
    )


# ── QA subset download ───────────────────────────────────────────────────────

def download_subset(subset: str, *, force: bool, limit: int | None) -> Path:
    out_path = DATA_DIR / f"wixqa_{subset}.csv"
    if out_path.exists() and not force:
        log.info("%s already exists — skip (use --force to overwrite)", out_path.name)
        return out_path

    candidates = _SUBSET_CONFIGS[subset]
    ds, cfg = _load_hf_config(HF_REPO, candidates)

    cols = list(ds.column_names)
    q_col = _pick(cols, _QUESTION_KEYS)
    a_col = _pick(cols, _ANSWER_KEYS)
    art_col = _pick(cols, _ARTICLE_KEYS)
    log.info("Column mapping → question=%r  answer=%r  article_ids=%r  (from %s)",
             q_col, a_col, art_col, cols)
    if not q_col or not a_col:
        raise RuntimeError(f"Could not detect question/answer columns in {cols}")

    url_to_id = _load_url_to_id_map()

    n_with_articles = 0
    rows_out: list[dict] = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        art_raw = row.get(art_col) if art_col else None
        art_norm = _normalise_article_ids(art_raw, url_to_id)
        if art_norm != "[]":
            n_with_articles += 1
        rows_out.append({
            "question":    str(row.get(q_col, "")).strip(),
            "answer":      str(row.get(a_col, "")).strip(),
            "article_ids": art_norm,
        })

    DATA_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", "article_ids"])
        w.writeheader()
        w.writerows(rows_out)

    log.info("Wrote %s  (%d rows, %d with article grounding, config=%r)",
             out_path.name, len(rows_out), n_with_articles, cfg)
    if art_col is None:
        log.warning("No article-id column detected — retrieval metrics (Layer 1) "
                    "will be empty for this subset. Answer/routing layers still work.")
    return out_path


# ── KB corpus download ───────────────────────────────────────────────────────

def download_kb(*, force: bool) -> Path:
    out_path = DATA_DIR / "wix_kb_corpus.csv"
    if out_path.exists() and not force:
        log.info("%s already exists — skip (use --force to overwrite)", out_path.name)
        return out_path

    ds = None
    # Try as a config within the main repo first, then standalone repos.
    for repo, cands in [(HF_REPO, _KB_CONFIGS)] + [(r, [None]) for r in _KB_REPO_FALLBACKS]:
        try:
            if cands == [None]:
                from datasets import load_dataset
                ds = load_dataset(repo, split="train")
                log.info("Loaded KB corpus from %s (%d rows)", repo, len(ds))
            else:
                ds, _ = _load_hf_config(repo, cands)
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("KB load attempt failed for %s: %s", repo, exc)
            ds = None

    if ds is None:
        raise RuntimeError("Could not download the Wix KB corpus from any known location.")

    cols = list(ds.column_names)
    id_col    = _pick(cols, _KB_ID_KEYS)
    url_col   = _pick(cols, _KB_URL_KEYS)
    cont_col  = _pick(cols, _KB_CONTENT_KEYS)
    title_col = _pick(cols, _KB_TITLE_KEYS)
    html_col  = _pick(cols, _KB_HTML_KEYS)
    type_col  = _pick(cols, _KB_TYPE_KEYS)
    log.info("KB column mapping → id=%r url=%r contents=%r title=%r html=%r type=%r",
             id_col, url_col, cont_col, title_col, html_col, type_col)
    if not id_col or not cont_col:
        raise RuntimeError(f"Could not detect id/contents columns in KB corpus: {cols}")

    rows_out = []
    for row in ds:
        rows_out.append({
            "id":           str(row.get(id_col, "")).strip(),
            "url":          str(row.get(url_col, "")).strip() if url_col else "",
            "contents":     str(row.get(cont_col, "")).strip(),
            "title":        str(row.get(title_col, "")).strip() if title_col else "",
            "html_content": str(row.get(html_col, "")).strip() if html_col else "",
            "article_type": str(row.get(type_col, "")).strip() if type_col else "",
        })

    DATA_DIR.mkdir(exist_ok=True)
    fields = ["id", "url", "contents", "title", "html_content", "article_type"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    log.info("Wrote %s  (%d KB articles)", out_path.name, len(rows_out))
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download WixQA datasets into data/")
    parser.add_argument(
        "--subset", choices=sorted(_SUBSET_CONFIGS), action="append", default=None,
        help="QA subset to download (repeatable). Default: synthetic if neither "
             "--subset nor --kb is given.",
    )
    parser.add_argument("--kb", action="store_true", help="Also download the KB corpus.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows per QA subset.")
    args = parser.parse_args()

    subsets = args.subset
    if not subsets and not args.kb:
        subsets = ["synthetic"]  # the explicit request: pull synthetic by default

    written: list[Path] = []
    for sub in (subsets or []):
        written.append(download_subset(sub, force=args.force, limit=args.limit))
    if args.kb:
        written.append(download_kb(force=args.force))

    log.info("─" * 60)
    log.info("Done. Files ready:")
    for p in written:
        log.info("  %s", p)


if __name__ == "__main__":
    main()
