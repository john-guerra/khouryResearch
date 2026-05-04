// Pure helpers shared between the D3 frontend (app.js) and the test suite.
// Nothing in here may touch the DOM, d3, or browser globals — keep it
// importable from plain Node so unit tests don't need a headless browser.

export const LAYERS = ["give_get", "topic", "keyword"];

export function normalizeTopic(s) {
  return String(s || "").toLowerCase().trim().replace(/\s+/g, " ");
}

// Build a per-node topic Set from the researcher's keyword + auto_keyword fields.
// Caches on the node so repeated calls during filter passes are O(1).
export function topicsForNode(n) {
  if (n._topicSet) return n._topicSet;
  const topics = new Set();
  for (const k of [...(n.keywords || []), ...(n.auto_keywords || [])]) {
    const t = normalizeTopic(k);
    if (t) topics.add(t);
  }
  n._topicSet = topics;
  return topics;
}

// Index every edge by both endpoints so neighbor lookups are O(deg).
// For directed (give_get) edges, the source's record marks `incoming: false`
// for itself and `incoming: true` on the target's record so the UI can render
// arrow direction correctly.
export function buildEdgesByNode(edgesByLayer) {
  const out = { give_get: new Map(), topic: new Map(), keyword: new Map() };
  for (const layer of LAYERS) {
    const map = out[layer];
    for (const e of edgesByLayer[layer] || []) {
      const list = (id) => map.get(id) || (map.set(id, []), map.get(id));
      list(e.source).push({ ...e, otherId: e.target });
      if (e.directed) {
        list(e.target).push({ ...e, otherId: e.source, incoming: true });
      } else {
        list(e.target).push({ ...e, otherId: e.source });
      }
    }
  }
  return out;
}

// 1-hop neighbors of `id` across the currently-active layers, plus id itself.
// CRITICAL: this MUST be re-evaluated every time the active-layer set changes.
// If you call it once and cache, focus-mode will dim the wrong nodes after a
// layer toggle.
export function neighborIdsOf(id, edgesByNode, activeLayers) {
  const set = new Set([id]);
  for (const layer of LAYERS) {
    if (!activeLayers.has(layer)) continue;
    const edges = edgesByNode[layer].get(id) || [];
    for (const e of edges) set.add(e.otherId);
  }
  return set;
}

// Whether any filter (search, scope, location chips, topic chips) is non-default.
// Used to decide whether 'Esc' should reset filters and whether the empty-state
// overlay is appropriate (we only show it when the user actually narrowed).
export function hasActiveFilters(opts) {
  const {
    searchTerm = "",
    searchScope = "all",
    selectedLocations = new Set(),
    selectedTopics = new Set(),
  } = opts || {};
  return (
    searchTerm !== "" ||
    searchScope !== "all" ||
    selectedLocations.size > 0 ||
    selectedTopics.size > 0
  );
}

// Decide whether a node matches the current filter state. Pure function so the
// logic is testable without a DOM. Returns boolean.
//
//   selectedLocations: Set<string>     AND-applied (must be in this location set if non-empty)
//   selectedTopics:    Set<string>     OR-applied  (any selected topic matching is enough)
//   searchTerm:        string          lowercased substring; empty matches everything
//   searchScope:       'all'|'give'|'get'  scopes searchTerm to bullets when not 'all'
export function nodeMatchesFilters(n, opts) {
  const {
    searchTerm = "",
    searchScope = "all",
    selectedLocations = new Set(),
    selectedTopics = new Set(),
  } = opts || {};

  if (selectedLocations.size > 0 && !selectedLocations.has(n.location)) return false;
  if (selectedTopics.size > 0) {
    const topics = topicsForNode(n);
    let any = false;
    for (const t of selectedTopics) if (topics.has(t)) { any = true; break; }
    if (!any) return false;
  }
  if (!searchTerm) return true;
  let haystack;
  if (searchScope === "give") {
    haystack = (n.give || []).join(" ");
  } else if (searchScope === "get") {
    haystack = (n.get || []).join(" ");
  } else {
    haystack = [
      n.name,
      n.primary_area,
      n.secondary_area,
      ...(n.keywords || []),
      ...(n.auto_keywords || []),
      ...(n.give || []),
      ...(n.get || []),
    ].join(" ");
  }
  return haystack.toLowerCase().includes(searchTerm);
}
