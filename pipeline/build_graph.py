"""Build the final graph.json consumed by the static frontend.

Reads researchers.json + features.npz + features.json. Emits one JSON file
with all nodes (full researcher data) and three edge sets — one per layer.

Edge selection:
- give→get: bi-encoder shortlist of top-K candidates per researcher, then
  cross-encoder rerank at the bullet-pair level. Top-3 above floor.
- topic + keyword: row-wise top-3 of the precomputed similarity matrix,
  with a per-layer score floor. Symmetric layers are deduped on sorted-pair.

The why-strings:
- give→get: the actual (give-bullet, get-bullet) pair selected by the
  cross-encoder, copied verbatim.
- keyword: highest-cosine semantic phrase pair between the two keyword sets,
  with lexical-shared fallback if no pair clears the floor.
- topic: empty (the layer's score is the explanation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESEARCHERS = ROOT / "pipeline" / "researchers.json"
FEATURES_NPZ = ROOT / "pipeline" / "features.npz"
FEATURES_JSON = ROOT / "pipeline" / "features.json"
OUT_JSON = ROOT / "public" / "data" / "graph.json"

TOP_K = 3            # edges kept per researcher per layer
CANDIDATES_K = 10    # bi-encoder shortlist size before CE rerank
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
KW_WHY_FLOOR = 0.55  # cosine threshold for "this phrase pair is the reason"
SEMANTIC_TO_LEXICAL_FALLBACK = True

FLOOR = {
    # The give_get score on the edge is the BI-ENCODER bullet-level score
    # (top-3-mean cosine). CE only chooses which top-3 candidates win the
    # rerank — selecting in highest-precision order — but its sigmoid is too
    # polarized to use as the visible score. Floor here matches the bi-encoder
    # cosine scale (0..1, real matches typically 0.3–0.8).
    "give_get": 0.30,
    "topic":    0.40,   # bi-encoder cosine, unchanged
    "keyword":  0.55,   # centroid cosine of keyword sets — replaces 0.10 Jaccard floor
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def slice_bullets(stack: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    return stack[offsets[i]:offsets[i + 1]]


def explain_keyword_lexical(kws_a: list[str], kws_b: list[str]) -> str:
    common = sorted(set(kws_a) & set(kws_b))
    if not common:
        return ""
    show = common[:4]
    suffix = f" (+{len(common)-len(show)} more)" if len(common) > len(show) else ""
    return ", ".join(show) + suffix


def best_phrase_pair(
    phrases_a: list[str],
    phrases_b: list[str],
    bi_sym,
    floor: float,
) -> tuple[str, str, float] | None:
    """Highest-cosine phrase pair across the two researcher's keyword sets.
    Returns (phrase_a, phrase_b, score) or None if no pair clears `floor`."""
    if not phrases_a or not phrases_b:
        return None
    a = bi_sym.encode(phrases_a, normalize_embeddings=True, convert_to_numpy=True)
    b = bi_sym.encode(phrases_b, normalize_embeddings=True, convert_to_numpy=True)
    M = a @ b.T
    ai, bi = np.unravel_index(int(np.argmax(M)), M.shape)
    score = float(M[ai, bi])
    if score < floor:
        return None
    return phrases_a[ai], phrases_b[bi], score


def main() -> int:
    researchers = json.loads(RESEARCHERS.read_text())
    npz = np.load(FEATURES_NPZ)
    features = json.loads(FEATURES_JSON.read_text())
    keyword_sets = features["keyword_sets"]
    token_sets = features.get("token_sets", keyword_sets)

    n = len(researchers)
    per_give = npz["per_give"]
    per_get = npz["per_get"]
    give_offsets = npz["give_offsets"]
    get_offsets = npz["get_offsets"]
    S_give_get = npz["S_give_get"]
    S_topic = npz["S_topic"]
    S_keyword = npz["S_keyword"]

    # ---- give→get: bi-encoder shortlist, then CE rerank --------------------
    print("Bi-encoder shortlist (top-{} per researcher)...".format(CANDIDATES_K))
    pairs_to_ce: list[tuple[str, str]] = []
    pair_meta: list[tuple[int, int, str, str]] = []  # (i, j, give_text, get_text)
    for i in range(n):
        # Top-CANDIDATES_K columns of S_give_get[i] (above 0; below-floor candidates
        # are still passed through — the CE may decide they're actually fine).
        order = np.argsort(-S_give_get[i])
        cands = [int(j) for j in order[:CANDIDATES_K] if S_give_get[i, j] > 0]
        gi = slice_bullets(per_give, give_offsets, i)
        if gi.shape[0] == 0:
            continue
        gi_texts = researchers[i].get("give") or []
        for j in cands:
            gj = slice_bullets(per_get, get_offsets, j)
            if gj.shape[0] == 0:
                continue
            gj_texts = researchers[j].get("get") or []
            M = gi @ gj.T
            a, b = np.unravel_index(int(np.argmax(M)), M.shape)
            give_text = gi_texts[a]
            get_text = gj_texts[b]
            pairs_to_ce.append((give_text, get_text))
            pair_meta.append((i, j, give_text, get_text))

    print(f"Loading cross-encoder {CE_MODEL!r} for {len(pairs_to_ce)} pair(s)...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_MODEL)
    raw_scores = np.asarray(ce.predict(pairs_to_ce), dtype=np.float32) if pairs_to_ce else np.zeros(0, dtype=np.float32)
    ce_scores = sigmoid(raw_scores) if raw_scores.size else raw_scores

    # CE-sigmoid scores are heavily polarized (~0.99 for unambiguous matches,
    # near 0 for the rest), which makes them a great SELECTION signal but a
    # poor display score. We use CE for ordering only; the edge's visible
    # score is the bi-encoder bullet-level cosine (S_give_get[i,j]). That
    # score stays in the 0..1 cosine range that maps cleanly to opacity.
    by_i: dict[int, list[tuple[int, float, float, str, str]]] = {}
    # tuple = (j, ce_score_for_sort, bi_score_for_display, give_text, get_text)
    for k, (i, j, give_text, get_text) in enumerate(pair_meta):
        bi_score = float(S_give_get[i, j])
        by_i.setdefault(i, []).append((j, float(ce_scores[k]), bi_score, give_text, get_text))
    edges_gg = []
    for i, candidates in by_i.items():
        candidates.sort(key=lambda t: -t[1])  # sort by CE score descending
        for j, _ce, bi_score, give_text, get_text in candidates[:TOP_K]:
            if bi_score < FLOOR["give_get"]:
                continue
            edges_gg.append({
                "source": researchers[i]["id"],
                "target": researchers[j]["id"],
                "score": round(bi_score, 4),
                "layer": "give_get",
                "directed": True,
                "why": f'"{give_text}"  →  "{get_text}"',
            })

    # ---- topic: unchanged top-k pickup, then symmetric dedup ---------------
    print("Selecting top-k topic edges...")
    seen_topic: set[tuple[str, str]] = set()
    edges_topic = []
    for i in range(n):
        order = np.argsort(-S_topic[i])
        for j in order[:TOP_K]:
            score = float(S_topic[i, j])
            if score < FLOOR["topic"]:
                break
            a, b = sorted((researchers[i]["id"], researchers[int(j)]["id"]))
            if (a, b) in seen_topic:
                continue
            seen_topic.add((a, b))
            edges_topic.append({
                "source": a, "target": b,
                "score": round(score, 4),
                "layer": "topic", "directed": False, "why": "",
            })

    # ---- keyword: centroid-cosine top-k, semantic phrase-pair why ----------
    print("Selecting top-k keyword edges + semantic why-strings...")
    # Re-load bi_sym for the why-string phrase encoding. Same model already
    # cached locally so this is cheap.
    from sentence_transformers import SentenceTransformer
    bi_sym = SentenceTransformer("all-MiniLM-L6-v2")

    seen_kw: set[tuple[str, str]] = set()
    edges_kw = []
    for i in range(n):
        order = np.argsort(-S_keyword[i])
        for j in order[:TOP_K]:
            score = float(S_keyword[i, j])
            if score < FLOOR["keyword"]:
                break
            a, b = sorted((researchers[i]["id"], researchers[int(j)]["id"]))
            if (a, b) in seen_kw:
                continue
            seen_kw.add((a, b))
            ai = next(k for k, r in enumerate(researchers) if r["id"] == a)
            bi_idx = next(k for k, r in enumerate(researchers) if r["id"] == b)

            # Why: highest-cosine phrase pair across the two keyword sets.
            pair = best_phrase_pair(
                keyword_sets[ai], keyword_sets[bi_idx], bi_sym, KW_WHY_FLOOR
            )
            if pair is not None:
                pa, pb, _ = pair
                why = pa if pa == pb else f"{pa} ↔ {pb}"
            elif SEMANTIC_TO_LEXICAL_FALLBACK:
                why = explain_keyword_lexical(token_sets[ai], token_sets[bi_idx])
            else:
                why = ""

            edges_kw.append({
                "source": a, "target": b,
                "score": round(score, 4),
                "layer": "keyword", "directed": False, "why": why,
            })

    # ---- Build node list ---------------------------------------------------
    nodes = []
    for i, r in enumerate(researchers):
        nodes.append({
            **r,
            "auto_keywords": keyword_sets[i],
            "provenance": "PPTX 2025",
        })

    primary_areas = sorted({r["primary_area"] for r in researchers if r["primary_area"]})
    locations = sorted({r["location"] for r in researchers if r["location"]})

    out = {
        "nodes": nodes,
        "edges": {
            "give_get": edges_gg,
            "topic": edges_topic,
            "keyword": edges_kw,
        },
        "filters": {
            "primary_areas": primary_areas,
            "locations": locations,
        },
        "meta": {
            "top_k": TOP_K,
            "score_floors": FLOOR,
            "models": {
                "give_get_bi": "multi-qa-MiniLM-L6-cos-v1",
                "give_get_ce": CE_MODEL,
                "topic_keyword": "all-MiniLM-L6-v2",
            },
            # 'model' kept for back-compat with the old meta-counts label.
            "model": "multi-qa + ms-marco-CE",
            "n_researchers": n,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"  nodes: {len(nodes)}")
    print(f"  give_get edges: {len(edges_gg)}")
    print(f"  topic edges: {len(edges_topic)}")
    print(f"  keyword edges: {len(edges_kw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
