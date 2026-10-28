# wixqa-dms-ingestion

> **Pipeline:** WixQA CSV → HTML → PDF → Paperless-ngx

Dieses Projekt erzeugt aus dem WixQA-Datensatz (`wix_kb_corpus.csv`) formatierte PDF-Dokumente und lädt diese in eine **Paperless-ngx**-Instanz hoch, wo sie als echte DMS-Dokumente für die spätere Weiterverarbeitung bereitstehen. Es setzt automatisch 12 Custom Fields für jeden Artikel.

> **Scope:** Nur Ingestion (CSV → PDF → DMS). Kein RAG, kein Embedding, kein Neo4j.

---

## Projektstruktur

```
wixqa-dms-ingestion/
├── data/
│   ├── raw/                        ← CSV hier ablegen
│   ├── generated/{html,pdf}/       ← automatisch erzeugt
│   └── mappings/                   ← JSONL-Mapping (Paperless-IDs)
├── docker/
│   └── paperless/                  ← docker-compose.yml für Paperless
├── src/
│   ├── config.py                   ← Pfade & Paperless-Konfiguration
│   ├── load_wixqa.py               ← CSV laden & bereinigen
│   ├── build_metadata.py           ← Metadaten & Titel extrahieren
│   ├── render_html.py              ← HTML erzeugen
│   ├── generate_pdf.py             ← WeasyPrint-Wrapper
│   └── paperless_client.py         ← Paperless REST-Client
└── scripts/
    ├── 01_generate_pdfs.py         ← Schritt 1: PDFs erzeugen
    ├── 02_setup_paperless.py       ← Schritt 2: Custom Fields anlegen
    └── 03_ingest_to_paperless.py   ← Schritt 3: Upload zu Paperless
```

---

## Setup

```bash
# 1. Virtuelle Umgebung erstellen und aktivieren
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Abhängigkeiten installieren
pip install -r requirements.txt
```

> **Hinweis (macOS):** WeasyPrint benötigt `cairo` und `pango`. Am einfachsten via Homebrew:
> ```bash
> brew install cairo pango libffi
> ```

---

## CSV ablegen

Der WixQA-Korpus ist nicht Teil des Repositories. Er wird vom Wurzelverzeichnis
des Projekts aus heruntergeladen:

```bash
python etl/download_wixqa.py --kb        # schreibt data/wix_kb_corpus.csv
```

Anschliessend die Datei hierher kopieren:

```
data/raw/wix_kb_corpus.csv
```

---

## Schritt 1 – PDFs erzeugen

```bash
python scripts/01_generate_pdfs.py
```

Erzeugt:
- `data/generated/html/<id>.html`
- `data/generated/pdf/<id>.pdf`

---

## Schritt 2 – Paperless-ngx starten & konfigurieren

Paperless-ngx wird lokal über Docker betrieben:

```bash
cd docker/paperless
docker compose up -d
cd ../..
```

**Erster Login & Token-Generierung:**
1. Gehe zu [http://localhost:8000](http://localhost:8000).
2. Logge dich mit `admin` / `admin` ein.
3. Klicke auf dein Profil oben rechts → **My Profile** → **API Auth Token**.
4. Generiere einen neuen Token und kopiere ihn.

**Konfiguration:**
Erstelle die `.env` Datei:
```bash
cp .env.example .env
```
Füge deinen Token ein:
```dotenv
PAPERLESS_BASE_URL=http://localhost:8000/api
PAPERLESS_TOKEN=dein_kopierter_token_hier
```

---

## Schritt 3 – Paperless-ngx einrichten

Führe das Setup-Skript aus, um die 12 benötigten Custom Fields in Paperless anzulegen. Dieses Skript ist idempotent und kann mehrfach ausgeführt werden:

```bash
python scripts/02_setup_paperless.py
```

---

## Schritt 4 – Upload zu Paperless

Das Hauptskript lädt die generierten PDFs hoch und weist die Custom Fields zu:

```bash
python scripts/03_ingest_to_paperless.py
```

- Lädt alle PDFs aus `data/generated/pdf/` in Paperless hoch.
- Wartet auf die asynchrone Verarbeitung (Consumption).
- Setzt die Metadaten.
- Schreibt eine JSONL-Mapping-Datei nach `data/mappings/paperless_document_mapping.jsonl`.
- Überspringt bereits hochgeladene Dokumente automatisch (`source_id` Idempotenz).

**Optionale Parameter:**
- `--limit N`: Lädt nur die ersten N Dokumente hoch (gut zum Testen).
- `--id SOURCE_ID`: Lädt gezielt ein einzelnes PDF hoch.
- `--dry-run`: Testet den Ablauf ohne echte Uploads.

### Mapping-Format (eine Zeile pro Dokument)

```json
{
  "source_id": "12345",
  "paperless_document_id": 42,
  "status": "success"
}
```

---

## Fehlerbehandlung

- Die Skripte fangen API-Fehler (5xx) und Throttling (429) ab und versuchen es per Exponential Backoff erneut.
- Bei nicht-behebbaren Fehlern wird der Upload des betroffenen PDFs abgebrochen, ein Fehler geloggt und in der Mapping-Datei als `status: "error"` vermerkt. Der Prozess läuft für die restlichen Dokumente weiter.
