"""Compute embeddings, keyword sets, and similarity matrices.

Reads researchers.json (output of parse_pptx.py) and writes features.npz +
features.json with everything needed by build_graph.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
IN_JSON = ROOT / "pipeline" / "researchers.json"
OUT_NPZ = ROOT / "pipeline" / "features.npz"
OUT_JSON = ROOT / "pipeline" / "features.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_PHRASES = 6  # extra phrases per researcher pulled from bullets


def extract_keyphrases(model: SentenceTransformer, text: str, top_n: int = TOP_PHRASES, diversity: float = 0.5) -> list[str]:
    """KeyBERT-style: candidate n-grams (1-3) ranked by cosine similarity to the document, with MMR for diversity."""
    if not text.strip():
        return []
    try:
        vec = CountVectorizer(ngram_range=(1, 3), stop_words="english").fit([text])
        candidates = vec.get_feature_names_out().tolist()
    except ValueError:
        return []
    if not candidates:
        return []
    doc_emb = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    cand_embs = model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True)
    sims_to_doc = cosine_similarity(cand_embs, doc_emb).flatten()
    sims_among = cosine_similarity(cand_embs)

    # MMR: greedily pick the candidate that maximizes (1-d)*sim_to_doc - d*max_sim_to_selected.
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    if not remaining:
        return []
    first = int(np.argmax(sims_to_doc))
    selected.append(first)
    remaining.remove(first)
    while remaining and len(selected) < top_n:
        scores = []
        for c in remaining:
            redundancy = max(sims_among[c, s] for s in selected)
            mmr = (1 - diversity) * sims_to_doc[c] - diversity * redundancy
            scores.append((mmr, c))
        _, best = max(scores)
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def normalize_keyword(kw: str) -> str:
    s = kw.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def encode_mean(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Mean-pool embeddings; returns a normalized vector of dim D."""
    if not texts:
        # 384-dim model produces 384-d vectors; we don't know D until first call,
        # so fall back to zeros after we encode something else.
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    mean = embs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of a vs rows of b. Both assumed normalized; result is cosine similarity."""
    return a @ b.T


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


# Light stopword list for tokenizing keyword phrases.
_TOKEN_STOP = {
    "the", "and", "or", "of", "for", "in", "on", "at", "to", "with", "a", "an",
    "is", "are", "be", "by", "as", "it", "from", "that", "this", "we", "us",
    "their", "our", "your", "you", "i", "me",
    # Domain-generic words that don't carry topic signal:
    "research", "expertise", "experience", "support", "etc", "use", "using",
    "new", "old", "more", "general", "various",
}


def keyword_tokens(phrases: list[str]) -> set[str]:
    """Tokenize keyword phrases into a set of individual content words for matching.

    'Visual analytics' and 'information visualization' share the token 'visualization'
    once we split — multi-word phrases as atoms had zero overlap.
    """
    tokens: set[str] = set()
    for p in phrases:
        for w in re.split(r"[^a-zA-Z0-9]+", p.lower()):
            if len(w) >= 3 and w not in _TOKEN_STOP:
                tokens.add(w)
    return tokens


def main() -> int:
    researchers = json.loads(IN_JSON.read_text())
    n = len(researchers)
    print(f"Loaded {n} researchers; loading model {MODEL_NAME!r}...")
    model = SentenceTransformer(MODEL_NAME)

    give_emb = np.zeros((n, model.get_sentence_embedding_dimension()), dtype=np.float32)
    get_emb = np.zeros_like(give_emb)
    profile_emb = np.zeros_like(give_emb)
    keyword_sets: list[list[str]] = []

    for i, r in enumerate(researchers):
        give_emb[i] = encode_mean(model, r["give"])
        get_emb[i] = encode_mean(model, r["get"])

        profile_text_parts = [
            r.get("primary_area") or "",
            r.get("secondary_area") or "",
            ", ".join(r.get("keywords") or []),
            " ".join(r.get("give") or []),
            " ".join(r.get("get") or []),
        ]
        profile_text = ". ".join(p for p in profile_text_parts if p)
        if profile_text.strip():
            v = model.encode(profile_text, normalize_embeddings=True, convert_to_numpy=True)
            profile_emb[i] = v.astype(np.float32)

        # Keyword set: explicit keywords + KeyBERT phrases from bullets.
        kw_set: set[str] = set()
        for kw in r.get("keywords") or []:
            nk = normalize_keyword(kw)
            if nk:
                kw_set.add(nk)
        bullets_text = ". ".join((r.get("give") or []) + (r.get("get") or []))
        for phrase in extract_keyphrases(model, bullets_text):
            nk = normalize_keyword(phrase)
            if nk and len(nk) > 2:
                kw_set.add(nk)
        keyword_sets.append(sorted(kw_set))

        print(f"  [{i+1:2d}/{n}] {r['name']}: {len(kw_set)} kws, give={len(r['give'])}, get={len(r['get'])}")

    # Similarity matrices
    S_give_get = cosine_matrix(give_emb, get_emb)  # directed: i→j means i can help j
    S_topic = cosine_matrix(profile_emb, profile_emb)
    np.fill_diagonal(S_give_get, -1.0)  # never connect a person to themselves
    np.fill_diagonal(S_topic, -1.0)

    # Keyword similarity = Jaccard over tokenized phrases (so multi-word
    # phrases share words like "visualization" instead of being atomic).
    token_sets = [keyword_tokens(ks) for ks in keyword_sets]
    S_keyword = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                S_keyword[i, j] = -1.0
            else:
                S_keyword[i, j] = jaccard(token_sets[i], token_sets[j])

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        give_emb=give_emb,
        get_emb=get_emb,
        profile_emb=profile_emb,
        S_give_get=S_give_get,
        S_topic=S_topic,
        S_keyword=S_keyword,
    )
    OUT_JSON.write_text(json.dumps({
        "keyword_sets": keyword_sets,
        "token_sets": [sorted(s) for s in token_sets],
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_NPZ} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
