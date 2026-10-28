"""
load_wixqa.py – Load and clean the WixQA knowledge-base corpus CSV.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: list[str] = ["id", "url", "contents", "article_type", "title", "html_content"]


def load_wixqa_corpus(path: Path | str) -> pd.DataFrame:
    """Load the WixQA CSV, validate schema, and return a cleaned DataFrame.

    Args:
        path: Path to ``wix_kb_corpus.csv``.

    Returns:
        A cleaned DataFrame with columns: id, url, contents, article_type.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, dtype=str)

    # Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only the columns we need
    df = df[REQUIRED_COLUMNS].copy()

    # Drop rows where contents is empty or whitespace-only
    before = len(df)
    df = df[df["contents"].notna() & (df["contents"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        print(f"[load_wixqa] Dropped {dropped} row(s) with empty 'contents'.")

    df = df.reset_index(drop=True)
    return df
