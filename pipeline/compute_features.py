"""Compute embeddings, keyword sets, and similarity matrices.

Reads researchers.json (output of parse_pptx.py) and writes features.npz +
features.json with everything needed by build_graph.py.

Two bi-encoders are loaded:
  - multi-qa-MiniLM-L6-cos-v1 — asymmetric (query/passage) for give↔get
  - all-MiniLM-L6-v2          — symmetric for topic + keyword

Per-bullet embeddings are saved (flattened with offsets) so build_graph.py can
rerank with a cross-encoder without re-encoding bullets.
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

# Asymmetric retrieval model — give-bullets are passages (offers), get-bullets
# are queries (needs). all-MiniLM is symmetric and trained on sentence-pair
# similarity; multi-qa-MiniLM is the same architecture trained on QA retrieval.
BI_QA_MODEL = "multi-qa-MiniLM-L6-cos-v1"
BI_SYM_MODEL = "all-MiniLM-L6-v2"

TOP_PHRASES = 6  # extra keyphrases per researcher pulled from bullets
TOP_PAIRS_FOR_GG = 3  # mean of top-N bullet-pair cosines becomes S_give_get


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


def encode_bullets(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Returns shape (len(texts), D) of normalized embeddings, or (0, D) if empty."""
    D = model.get_sentence_embedding_dimension()
    if not texts:
        return np.zeros((0, D), dtype=np.float32)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs.astype(np.float32)


def topk_mean_score(M: np.ndarray, k: int) -> float:
    """Mean of the top-k cells of M (a similarity matrix). Falls back to max
    if M has fewer than k cells. M is assumed to be give × get cosine."""
    if M.size == 0:
        return -1.0
    flat = M.ravel()
    if flat.size <= k:
        return float(flat.mean())
    top = np.partition(flat, -k)[-k:]
    return float(top.mean())


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
    """Tokenize keyword phrases into a set of individual content words for matching."""
    tokens: set[str] = set()
    for p in phrases:
        for w in re.split(r"[^a-zA-Z0-9]+", p.lower()):
            if len(w) >= 3 and w not in _TOKEN_STOP:
                tokens.add(w)
    return tokens


def main() -> int:
    researchers = json.loads(IN_JSON.read_text())
    n = len(researchers)
    print(f"Loaded {n} researchers")
    print(f"Loading {BI_QA_MODEL!r} (give↔get retrieval)...")
    bi_qa = SentenceTransformer(BI_QA_MODEL)
    print(f"Loading {BI_SYM_MODEL!r} (topic + keyword similarity)...")
    bi_sym = SentenceTransformer(BI_SYM_MODEL)
    D = bi_qa.get_sentence_embedding_dimension()
    assert bi_sym.get_sentence_embedding_dimension() == D, \
        "Both models must share embedding dim — saving as one stack."

    # Per-bullet embeddings stored flat with offsets so build_graph.py can
    # access bullets[give_offsets[i]:give_offsets[i+1]] without re-encoding.
    per_give: list[np.ndarray] = []
    per_get: list[np.ndarray] = []
    profile_emb = np.zeros((n, D), dtype=np.float32)
    kw_centroid = np.zeros((n, D), dtype=np.float32)
    keyword_sets: list[list[str]] = []

    for i, r in enumerate(researchers):
        per_give.append(encode_bullets(bi_qa, r.get("give") or []))
        per_get.append(encode_bullets(bi_qa, r.get("get") or []))

        profile_text_parts = [
            r.get("primary_area") or "",
            r.get("secondary_area") or "",
            ", ".join(r.get("keywords") or []),
            " ".join(r.get("give") or []),
            " ".join(r.get("get") or []),
        ]
        profile_text = ". ".join(p for p in profile_text_parts if p)
        if profile_text.strip():
            v = bi_sym.encode(profile_text, normalize_embeddings=True, convert_to_numpy=True)
            profile_emb[i] = v.astype(np.float32)

        # Keyword set: explicit keywords + KeyBERT phrases from bullets.
        kw_set: set[str] = set()
        for kw in r.get("keywords") or []:
            nk = normalize_keyword(kw)
            if nk:
                kw_set.add(nk)
        bullets_text = ". ".join((r.get("give") or []) + (r.get("get") or []))
        for phrase in extract_keyphrases(bi_sym, bullets_text):
            nk = normalize_keyword(phrase)
            if nk and len(nk) > 2:
                kw_set.add(nk)
        keyword_sets.append(sorted(kw_set))

        # Keyword centroid: mean of bi_sym embeddings of each phrase, then
        # L2-normalize so cosine = dot product. Used as the new keyword-layer
        # similarity (replaces token-Jaccard).
        if kw_set:
            kw_embs = bi_sym.encode(sorted(kw_set), normalize_embeddings=True, convert_to_numpy=True)
            mean = kw_embs.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                kw_centroid[i] = (mean / norm).astype(np.float32)

        print(f"  [{i+1:2d}/{n}] {r['name']}: {len(kw_set)} kws, give={len(r['give'])}, get={len(r['get'])}")

    # ---- Similarity matrices --------------------------------------------------
    # give→get: bullet-level pairwise, then mean of top-N cells.
    S_give_get = np.full((n, n), -1.0, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j or per_give[i].shape[0] == 0 or per_get[j].shape[0] == 0:
                continue
            M = per_give[i] @ per_get[j].T
            S_give_get[i, j] = topk_mean_score(M, TOP_PAIRS_FOR_GG)

    # topic: profile cosine.
    S_topic = profile_emb @ profile_emb.T
    np.fill_diagonal(S_topic, -1.0)

    # keyword: centroid cosine. -1 on diagonal and for researchers with no keywords.
    S_keyword = kw_centroid @ kw_centroid.T
    np.fill_diagonal(S_keyword, -1.0)
    # Researchers whose keyword centroid is zero (no keywords) shouldn't appear.
    has_kw = np.linalg.norm(kw_centroid, axis=1) > 0
    no_kw_mask = ~has_kw
    if no_kw_mask.any():
        S_keyword[no_kw_mask, :] = -1.0
        S_keyword[:, no_kw_mask] = -1.0

    # Persist per-bullet embeddings as a stacked matrix + offsets so we can
    # slice bullets[give_offsets[i]:give_offsets[i+1]] in build_graph.py.
    give_offsets = np.zeros(n + 1, dtype=np.int64)
    get_offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        give_offsets[i + 1] = give_offsets[i] + per_give[i].shape[0]
        get_offsets[i + 1] = get_offsets[i] + per_get[i].shape[0]
    per_give_stack = (
        np.concatenate(per_give, axis=0) if any(p.shape[0] for p in per_give) else np.zeros((0, D), dtype=np.float32)
    )
    per_get_stack = (
        np.concatenate(per_get, axis=0) if any(p.shape[0] for p in per_get) else np.zeros((0, D), dtype=np.float32)
    )

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        per_give=per_give_stack,
        per_get=per_get_stack,
        give_offsets=give_offsets,
        get_offsets=get_offsets,
        profile_emb=profile_emb,
        kw_centroid=kw_centroid,
        S_give_get=S_give_get,
        S_topic=S_topic,
        S_keyword=S_keyword,
    )

    # token_sets is no longer used for similarity, but we keep it because
    # build_graph's keyword why-string falls back to lexical-shared tokens
    # when the semantic phrase-pair score is too low to be meaningful.
    token_sets = [keyword_tokens(ks) for ks in keyword_sets]
    OUT_JSON.write_text(json.dumps({
        "keyword_sets": keyword_sets,
        "token_sets": [sorted(s) for s in token_sets],
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_NPZ} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
