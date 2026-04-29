# Shelfie: Automated Book Detection and Library Cataloging

A computer vision pipeline that converts photos of bookshelves into structured library catalogs. Detects book spines, reads titles with vision-language models, matches against Google Books, and ranks candidates using multimodal CLIP similarity.

## What It Does

1. **Spine Detection** — Faster R-CNN locates individual book spines in shelf photos
2. **OCR** — Qwen2-VL-2B extracts structured metadata (`title`, `author`, `publisher`) as JSON
3. **Retrieval** — Structured Google Books queries (`intitle:`, `inauthor:`) fetch candidate matches
4. **Re-ranking** — CLIP fuses text similarity (OCR vs candidate titles) with image similarity (spine crop vs cover thumbnail)
5. **Failure Router** — Labels each result as `ok` / `ambiguous` / `low_confidence` / `unreadable` instead of silently dropping uncertain matches
6. **Interactive Catalog** — Gradio app for reviewing, correcting, and exporting your library

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

# Set your Google Books API key 
export GOOGLE_BOOKS_API_KEY="your_key_here"

# Run the interactive catalog app
python -c "
from catalog_pipeline import launch_library_app
launch_library_app()
"
```

Then open the Gradio URL and upload shelf photos.

## Project Structure

```
├── catalog_pipeline.ipynb    # Main notebook: pipeline + evaluation + Gradio app
├── cnn_baseline.ipynb        # Faster R-CNN training and detection evaluation
├── ocr_testing.ipynb         # OCR backend benchmarking (10 models compared)
├── models/
│   └── fasterrcnn_book.pt    # Trained detector weights
├── data/
│   ├── LibVision-2/          # Shelf photos and YOLO detection annotations
│   └── my-books/             # Own shelf photos for testing
├── catalogs/                 # Generated per-shelf JSON/CSV catalogs
│   └── library.json          # Exported library 
├── cache/
│   ├── google_books.json     # Google Books API response cache
│   └── thumbnails/           # Downloaded cover thumbnails
└── requirements.txt
```




