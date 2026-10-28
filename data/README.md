# `data/`

This directory holds the WixQA evaluation corpora. **The CSVs are not tracked in
git** — they are third-party data (≈56 MB) that the repository regenerates on
demand rather than redistributing.

## Fetching the data

```bash
# KB corpus + the QA subset you want to evaluate on
python etl/download_wixqa.py --kb --subset expertwritten
python etl/download_wixqa.py --subset simulated
python etl/download_wixqa.py --subset synthetic
```

This writes, into this directory:

| File | Contents |
|---|---|
| `wix_kb_corpus.csv` | Knowledge-base articles (`id, url, contents, title, html_content, article_type`) |
| `wixqa_expertwritten.csv` | ExpertWritten QA split (`question, answer, article_ids`) |
| `wixqa_simulated.csv` | Simulated QA split |
| `wixqa_synthetic.csv` | Synthetic QA split |

`bm25_index.pkl` is a derived artefact built by the ingest pipeline and is also
untracked; it is rebuilt automatically when missing.

## Source & licence

WixQA is published by Wix on the Hugging Face Hub: <https://huggingface.co/datasets/Wix/WixQA>.
Consult that dataset card for its licence and citation requirements.
