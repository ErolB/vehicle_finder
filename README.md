# Vehicle Finder

  Semantic search over a car dealership's inventory. You describe the car you want
  in plain English — "reliable 4x4 for winter commuting, under 60k miles" — and it
  ranks the actual listings by meaning rather than keyword overlap.

  The repo ships with a scraped snapshot of 411 listings from Beck Chrysler Dodge
  Jeep, already parsed, described, and embedded, so the web app runs without
  re-running the pipeline.

  ## How it works

  ```
  scraper.py    inventory pages  ->  html/<VIN>.html        (Playwright, headless Chromium)
  parser.py     html/            ->  text/<VIN>.txt         (HTML text extraction + Claude cleanup)
  describe.py   text/            ->  descriptions/<VIN>.txt (Claude writes 5 short blurbs per car)
  finetune.py   text/ + descriptions/ -> model/             (domain-adapts a sentence-transformer)
  embedding.py  text/            ->  embeddings/<VIN>.npy   (one vector per listing)
  search.py     query + embeddings/  ->  ranked VINs
  app.py        the same search behind a Flask UI
  ```

  Every stage keys off the **VIN**, so a car's page, text, descriptions, and vector
  all share a filename. Each stage also skips files it has already produced, so an
  interrupted batch resumes where it stopped.

  The interesting part is `describe.py` → `finetune.py`. Listing pages are long,
  templated dealer prose; user queries are short and impressionistic. To close that
  gap, Claude extracts each vehicle's notable features, then writes five short
  descriptions per car, each highlighting a different random subset of those
  features. Those descriptions become the query-side of `(listing, description)`
  training pairs, and the embedding model is fine-tuned so a casual phrase lands
  near the listing it belongs to.

  ## Setup

  Python 3.10+.

  ```bash
  pip install flask numpy python-dotenv anthropic sentence-transformers torch playwright voyageai
  python -m playwright install chromium     # only needed to re-scrape
  ```

  Create a `.env` in the repo root:

  ```
  ANTHROPIC_API_KEY=sk-ant-...    # parser.py, describe.py
  VOYAGE_API_KEY=pa-...           # only for the optional Voyage backend
  ```

  The fine-tuned model (`model/`, ~400 MB) is tracked with **Git LFS** — run
  `git lfs install` before cloning, or `git lfs pull` afterwards.

  ## Running the app

  ```bash
  python app.py
  ```

  Then open http://localhost:5000. Type what you're after, and optionally narrow
  the candidate pool before ranking:

  - **Keywords** — a listing must contain *all* of them, matched as whole words
    (so `red` won't hit "covered"). Quote a phrase to keep it together:
    `"Apple CarPlay" 4x4`.
  - **New / used** — unchecking both means no restriction, not zero results.
  - **Price and mileage ranges** — the sliders span the real inventory's extremes.
    A listing whose figure can't be parsed ("Please Call", or a new car's blank
    odometer) is never dropped by these filters.

  Filters run *before* the vector search, so ranking always happens over the
  listings you actually allowed.

  ## Searching from the command line

  ```bash
  python search.py "family SUV with third row seating" -k 10
  python search.py "weekend off-roader" -w 4x4 "tow package"
  ```

  Prints `VIN`, cosine score, and vehicle name, best first.

  ## Rebuilding the pipeline

  Only needed if you want fresh inventory or a retrained model.

  ```bash
  python scraper.py                             # re-scrape (edit base_url for a different dealer)
  python parser.py                              # HTML -> clean text
  python describe.py                            # generate training descriptions ($, LLM calls)
  python finetune.py --paired --holdout 0.1     # fine-tune; reserves 10% of VINs for evaluation
  python embedding.py                           # rebuild embeddings/ with the new model
  python evaluate.py                            # recall@k / MRR on the held-out VINs
  ```

  Delete `embeddings/` before re-running `embedding.py` — existing `.npy` files are
  skipped, and a mix of vectors from two different models is meaningless.

  `finetune.py` also has an unsupervised SimCSE mode (drop `--paired`) that needs
  only `text/`, and refuses to start if `model/` is memory-mapped by a running
  `app.py` — stop the web app before retraining.

  Evaluation asks a direct question: given one of a car's generated descriptions as
  the query, how often does that car's own listing come back first? It reports
  recall@1/5/10, MRR, and median rank.

  ## Two embedding backends

  | | Local (default) | Voyage (`--api`) |
  |---|---|---|
  | Model | `model/`, fine-tuned from `all-mpnet-base-v2` (768-dim) | `voyage-4-large` |
  | Cost | free, runs on CPU | per-token API |
  | Long pages | chunked by token window and mean-pooled | 32k context, fits in one call |
  | Query/doc asymmetry | none | uses `input_type` query vs. document |

  Both sides must agree. Vectors built by `embedding.py --api` can only be searched
  with `search.py --api` (or the "use API" checkbox in the web UI); mixing them
  produces a dimension mismatch, which the app reports rather than crashing.

  ## Repository layout

  ```
  app.py            Flask UI: filters, sliders, backend toggle
  search.py         ranking, plus the parsers for name/price/mileage/condition/URL
  embedding.py      builds embeddings/, both backends, boilerplate stripping
  finetune.py       fine-tunes the embedding model (paired or SimCSE)
  evaluate.py       description -> own-listing retrieval metrics
  scraper.py        Playwright inventory crawler
  parser.py         HTML -> text
  describe.py       feature extraction + description generation
  templates/        the single-page search UI
  html/ text/ descriptions/ embeddings/    per-VIN data, checked in
  model/            fine-tuned weights (Git LFS)
  ```

  ## Implementation notes

  - **Boilerplate stripping.** Dealer nav, headers, and footers are byte-identical
    across all 411 pages and would dominate any embedding. Lines appearing in ≥90%
    of listings are dropped before embedding and before training, so both see the
    same text.
  - **Chunk-and-pool.** The local model truncates past its max sequence length, so
    a whole listing is split into token windows, embedded separately, and averaged.
    Training does the same, pairing every chunk with the listing's description.
  - **Price parsing** is deliberately narrow: it reads label/amount rows and skips
    discount and savings lines, plus the dollar figures floating in disclaimer
    prose. When an MSRP is higher than the sale price, the UI strikes it through.
  - **Reproducible augmentation.** The random feature combinations in `describe.py`
    are seeded per VIN, so re-running produces the same descriptions.
