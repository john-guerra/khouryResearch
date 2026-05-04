# Khoury Researcher Connections

An interactive visualization that surfaces collaborator-finding connections between Khoury College researchers, built from the *Khoury Research Day* "Give/Get" slide deck.

The graph shows three different *kinds* of connection, each as a toggleable layer:

- **Give → Get matches** (directed) — one researcher's "give" bullets semantically match another's "get" bullets. The marketplace view; surfaces who can directly help whom across topic clusters.
- **Topic similarity** (undirected) — cosine similarity over each researcher's combined bullets/areas/keywords. Surfaces intellectual neighbors.
- **Shared keywords** (undirected) — Jaccard overlap over self-reported keywords plus auto-extracted topic phrases.

Edges are top-3 per researcher per layer with score floors, so density stays manageable at any corpus size. Hover any edge to see *why* the connection exists. Click a researcher to see their full give/get bullets plus their top connections; turn on **Focus** to dim the rest of the graph and see only that person's neighborhood.

Live at: <https://johnguerra.co/viz/khouryResearch/> *(when deployed)*

## Project layout

```
giveAndGetSlides/
├── Khoury Research Day Give_Get_Slides.pptx     # source data (kept in repo)
├── pipeline/                                    # Python build step
│   ├── parse_pptx.py        # PPTX → researchers.json + photos/
│   ├── compute_features.py  # embeddings + keyword sets + similarity matrices
│   ├── build_graph.py       # → public/data/graph.json
│   ├── run_all.py           # one-command runner for the three steps above
│   └── photo_overrides/     # drop <slug>.jpg here to override extracted headshots
├── public/                                      # static site (rsync target)
│   ├── index.html
│   ├── about.html
│   ├── app.js               # D3 force layout + UI
│   ├── styles.css
│   └── data/
│       ├── graph.json       # nodes + 3 edge sets + metadata
│       └── photos/          # 512×512 JPEG, one per researcher
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

There is **no backend**. The pipeline emits one JSON file; the frontend consumes it. Re-running the pipeline is the only way data changes.

## Quick start

### 1. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` sentence-transformer (~80 MB).

### 2. Run the pipeline

```bash
python pipeline/run_all.py
```

This (re)parses the PPTX, computes embeddings, and writes `public/data/graph.json` plus per-researcher headshots into `public/data/photos/`.

You can also run individual steps:

```bash
python pipeline/parse_pptx.py        # → pipeline/researchers.json + public/data/photos/
python pipeline/compute_features.py  # → pipeline/features.npz + features.json
python pipeline/build_graph.py       # → public/data/graph.json
```

### 3. Serve locally

```bash
cd public && python -m http.server 8765
# open http://localhost:8765
```

### 4. Deploy

The `public/` directory is fully static. Two options:

**GitHub Pages** (default): pushes to `main` trigger `.github/workflows/pages.yml`, which uploads `public/` as a Pages artifact and deploys it. Enable Pages in repo settings → *Pages* → *Source: GitHub Actions*.

**Rsync** to any other static host:

```bash
rsync -auvz public/ user@host:/var/www/khoury-connections/
```

## Fixing a bad headshot

The parser picks the most-square image per slide as the headshot. When that's wrong (e.g. a slide's only image is a group photo), drop a real headshot in `pipeline/photo_overrides/<slug>.<ext>` and re-run the pipeline. The slug is the researcher's name lowercased and hyphen-separated (see the `id` field in `pipeline/researchers.json`).

## Tests

```bash
node --test tests/lib.test.mjs    # frontend pure-logic (focus + filters)
python3 -m unittest tests.test_graph    # pipeline output invariants
```

The frontend tests import the pure helpers from `public/lib.js` directly — no
DOM, no headless browser. The pipeline test runs against the existing
`public/data/graph.json`, so re-run the pipeline before testing if you've
edited `pipeline/`.

## Stack

- **Pipeline**: Python 3.12 · python-pptx · sentence-transformers (`all-MiniLM-L6-v2`) · scikit-learn · Pillow
- **Frontend**: Vanilla JS · D3 v7 (force layout, scales, zoom, transitions)
- **Hosting**: any static file host

## Roadmap (deferred from v1)

- **Auto-enrichment** of profiles from public sources (Khoury bios, DBLP, Google Scholar) so non-attendees of Research Day are included. Each enriched datum gets its own provenance badge.
- **2D embedding map** (UMAP) as a secondary view once the corpus grows past ~80 researchers — at N=24 a projection is too unstable to be useful.
- **Profile claiming** so researchers can edit their own give/get/keywords without waiting for the next Research Day.

## How "give → get" matching works

Each researcher's give-bullets and get-bullets are encoded individually with `multi-qa-MiniLM-L6-cos-v1` (a retrieval-trained bi-encoder). For every directed pair (A, B), the bullet-level similarity matrix `give_A @ get_B.T` gives a per-bullet-pair cosine; the per-edge score is the **mean of the top-3 cells** (or the max if either side has fewer than 3 bullets).

A `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker then orders the top-10 candidates per researcher: it scores each (best-bullet-give-A, best-bullet-get-B) pair and the top-3 by CE confidence become edges (filtered to bi-encoder score ≥ 0.30). The cross-encoder lifts ordering precision; the visible score stays in the bi-encoder cosine range so it maps cleanly to opacity.

The "why" string on each edge is the actual (give-bullet, get-bullet) pair the cross-encoder picked — verbatim text, copied at build time so it's trusted by the UI.

## How topic and keyword similarity work

**Topic layer** (undirected): bi-encoder cosine over each researcher's combined `primary_area + secondary_area + keywords + bullets`, using `all-MiniLM-L6-v2`.

**Keyword layer** (undirected): each researcher's keyword set (explicit + KeyBERT-extracted phrases) is mean-pooled to a centroid embedding; edges are top-3 centroid-cosine pairs above 0.55. The "why" string is the highest-cosine phrase pair across the two sets — e.g., "data viz ↔ information visualization", which lexical Jaccard would miss entirely.

## License

[MIT](LICENSE) © 2026 John A. Guerra Gómez. Code authored by [Claude Code](https://claude.com/claude-code) on John's behalf — see [`public/about.html`](public/about.html) for the full story.
