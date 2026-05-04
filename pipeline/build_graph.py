"""Build the final graph.json consumed by the static frontend.

Reads researchers.json + features.npz + features.json. Emits one JSON file
with all nodes (full researcher data) and three edge sets — one per layer.

Edge selection: top-k per node per layer, with a per-layer score floor.
This bounds the visual density at any corpus size and keeps every node
non-isolated when at least one good match exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESEARCHERS = ROOT / "pipeline" / "researchers.json"
FEATURES_NPZ = ROOT / "pipeline" / "features.npz"
FEATURES_JSON = ROOT / "pipeline" / "features.json"
OUT_JSON = ROOT / "public" / "data" / "graph.json"

TOP_K = 3
FLOOR = {
    "give_get": 0.30,
    "topic": 0.40,
    "keyword": 0.10,
}


def top_k_edges(S: np.ndarray, k: int, floor: float) -> list[tuple[int, int, float]]:
    """For each row i, pick the top-k columns above floor. Returns list of (i, j, score)."""
    n = S.shape[0]
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        row = S[i].copy()
        # diagonal already set to -1 in compute_features
        # Pick indices of the top-k scores
        idx = np.argsort(-row)[:k]
        for j in idx:
            score = float(row[j])
            if score >= floor:
                out.append((i, int(j), score))
    return out


def explain_keyword(kws_a: list[str], kws_b: list[str]) -> str:
    common = sorted(set(kws_a) & set(kws_b))
    if not common:
        return ""
    show = common[:4]
    suffix = f" (+{len(common)-len(show)} more)" if len(common) > len(show) else ""
    return ", ".join(show) + suffix


def main() -> int:
    researchers = json.loads(RESEARCHERS.read_text())
    npz = np.load(FEATURES_NPZ)
    features = json.loads(FEATURES_JSON.read_text())
    keyword_sets = features["keyword_sets"]
    token_sets = features.get("token_sets", keyword_sets)

    n = len(researchers)
    S_give_get = npz["S_give_get"]
    S_topic = npz["S_topic"]
    S_keyword = npz["S_keyword"]

    # For give↔get explanation, re-encode each give and get bullet so we can pick the best pair.
    print("Encoding bullets to build give↔get explanations...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    per_give_emb: list[np.ndarray] = []
    per_get_emb: list[np.ndarray] = []
    for r in researchers:
        gi = r.get("give") or []
        ge = r.get("get") or []
        per_give_emb.append(
            model.encode(gi, normalize_embeddings=True, convert_to_numpy=True) if gi else np.zeros((0, 384))
        )
        per_get_emb.append(
            model.encode(ge, normalize_embeddings=True, convert_to_numpy=True) if ge else np.zeros((0, 384))
        )

    def give_get_why(i: int, j: int) -> str:
        gi_b = researchers[i].get("give") or []
        gj_b = researchers[j].get("get") or []
        if not gi_b or not gj_b:
            return ""
        sims = per_give_emb[i] @ per_get_emb[j].T
        best = np.unravel_index(np.argmax(sims), sims.shape)
        gi_bullet = gi_b[best[0]]
        gj_bullet = gj_b[best[1]]
        return f'"{gi_bullet}"  →  "{gj_bullet}"'

    # Build edges per layer
    print("Selecting top-k edges per layer...")
    raw_gg = top_k_edges(S_give_get, TOP_K, FLOOR["give_get"])
    raw_topic = top_k_edges(S_topic, TOP_K, FLOOR["topic"])
    raw_kw = top_k_edges(S_keyword, TOP_K, FLOOR["keyword"])

    edges_gg = [
        {
            "source": researchers[i]["id"],
            "target": researchers[j]["id"],
            "score": round(score, 4),
            "layer": "give_get",
            "directed": True,
            "why": give_get_why(i, j),
        }
        for i, j, score in raw_gg
    ]

    # Topic similarity is symmetric; collapse (i,j) and (j,i) into one undirected edge.
    seen_topic: set[tuple[str, str]] = set()
    edges_topic = []
    for i, j, score in raw_topic:
        a, b = sorted((researchers[i]["id"], researchers[j]["id"]))
        if (a, b) in seen_topic:
            continue
        seen_topic.add((a, b))
        edges_topic.append({"source": a, "target": b, "score": round(score, 4), "layer": "topic", "directed": False, "why": ""})

    seen_kw: set[tuple[str, str]] = set()
    edges_kw = []
    for i, j, score in raw_kw:
        a, b = sorted((researchers[i]["id"], researchers[j]["id"]))
        if (a, b) in seen_kw:
            continue
        seen_kw.add((a, b))
        # Look up indices for explanation
        ai = next(k for k, r in enumerate(researchers) if r["id"] == a)
        bi = next(k for k, r in enumerate(researchers) if r["id"] == b)
        edges_kw.append({
            "source": a,
            "target": b,
            "score": round(score, 4),
            "layer": "keyword",
            "directed": False,
            "why": explain_keyword(token_sets[ai], token_sets[bi]),
        })

    # Build node list — include every researcher field needed by the side panel.
    nodes = []
    for i, r in enumerate(researchers):
        nodes.append({
            **r,
            "auto_keywords": keyword_sets[i],
            "provenance": "PPTX 2025",
        })

    # Aggregate research areas for filter UI
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
            "model": "all-MiniLM-L6-v2",
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
