// Frontend pure-logic tests. Run with `node --test tests/lib.test.mjs`.
// These cover the focus-mode neighbor refresh bug (must change with active
// layer set) and filter combinatorics: locations AND, topics OR, scoped search.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  LAYERS,
  normalizeTopic,
  topicsForNode,
  buildEdgesByNode,
  neighborIdsOf,
  nodeMatchesFilters,
} from "../public/lib.js";

// ---------- Tiny fixture ----------------------------------------------------
// Hand-built mini graph: 4 researchers, edges crafted so each layer gives a
// different ego network for "alice". This is exactly the surface the
// focus-mode bug lived on, so the tests target it directly.
function fixture() {
  const nodes = [
    {
      id: "alice", name: "Alice", location: "Boston",
      primary_area: "Visualization", secondary_area: "HCI",
      keywords: ["VisAnalytics", "data viz"],
      auto_keywords: ["narrative visualization"],
      give: ["I help with d3 layouts"],
      get:  ["I want collaborators on user studies"],
    },
    {
      id: "bob", name: "Bob", location: "Seattle",
      primary_area: "ML", secondary_area: "NLP",
      keywords: ["embeddings", "data viz"],
      auto_keywords: [],
      give: ["I can advise on transformer pretraining"],
      get:  ["I want viz collaborators"],
    },
    {
      id: "carol", name: "Carol", location: "Boston",
      primary_area: "Education", secondary_area: "",
      keywords: ["pedagogy"],
      auto_keywords: [],
      give: ["I can help with curriculum design"],
      get:  ["I need help with d3 visualizations"],
    },
    {
      id: "dave", name: "Dave", location: "Oakland",
      primary_area: "Security", secondary_area: "",
      keywords: ["malware"],
      auto_keywords: [],
      give: ["I run a malware lab"],
      get:  ["I want intro to ML researchers"],
    },
  ];
  const edges = {
    // alice → carol (alice gives d3, carol wants d3)
    give_get: [
      { source: "alice", target: "carol", layer: "give_get", score: 0.62, directed: true, why: "d3 ↔ d3" },
      { source: "bob",   target: "alice", layer: "give_get", score: 0.55, directed: true, why: "viz ↔ user studies" },
    ],
    // alice ~ bob via topic; carol ~ dave via topic (no alice/bob in carol's neighborhood)
    topic: [
      { source: "alice", target: "bob",   layer: "topic", score: 0.71, directed: false, why: "" },
      { source: "carol", target: "dave",  layer: "topic", score: 0.42, directed: false, why: "" },
    ],
    // alice/bob share "data viz"
    keyword: [
      { source: "alice", target: "bob",   layer: "keyword", score: 0.25, directed: false, why: "data viz" },
    ],
  };
  return { nodes, edges };
}

// ---------- normalizeTopic + topicsForNode ---------------------------------
test("normalizeTopic lowercases + collapses whitespace", () => {
  assert.equal(normalizeTopic("  Data   Viz  "), "data viz");
  assert.equal(normalizeTopic("HCAI"), "hcai");
  assert.equal(normalizeTopic(""), "");
  assert.equal(normalizeTopic(null), "");
});

test("topicsForNode unions keywords + auto_keywords, normalized + cached", () => {
  const n = {
    keywords: ["Visualization", "  Data Viz "],
    auto_keywords: ["narrative visualization", "Visualization"],
  };
  const t = topicsForNode(n);
  assert.deepEqual([...t].sort(), ["data viz", "narrative visualization", "visualization"]);
  // Cache: same Set instance returned.
  assert.equal(topicsForNode(n), t);
});

// ---------- buildEdgesByNode -----------------------------------------------
test("buildEdgesByNode indexes both endpoints per layer", () => {
  const { edges } = fixture();
  const idx = buildEdgesByNode(edges);
  for (const layer of LAYERS) assert.ok(idx[layer] instanceof Map);

  // alice has give_get edges to carol (out) and from bob (in)
  const aliceGG = idx.give_get.get("alice");
  assert.equal(aliceGG.length, 2);
  const others = aliceGG.map((e) => e.otherId).sort();
  assert.deepEqual(others, ["bob", "carol"]);
  // The incoming edge from bob is marked
  const fromBob = aliceGG.find((e) => e.otherId === "bob");
  assert.equal(fromBob.incoming, true);
  const toCarol = aliceGG.find((e) => e.otherId === "carol");
  assert.notEqual(toCarol.incoming, true);
});

// ---------- neighborIdsOf: the focus-mode refresh bug ---------------------
test("neighborIdsOf returns ego network across active layers only", () => {
  const { edges } = fixture();
  const idx = buildEdgesByNode(edges);

  // Only give_get active: alice's neighbors are carol (out) + bob (in via reciprocal)
  const giveOnly = neighborIdsOf("alice", idx, new Set(["give_get"]));
  assert.deepEqual([...giveOnly].sort(), ["alice", "bob", "carol"]);

  // Only topic: alice ~ bob
  const topicOnly = neighborIdsOf("alice", idx, new Set(["topic"]));
  assert.deepEqual([...topicOnly].sort(), ["alice", "bob"]);

  // Only keyword: alice ~ bob
  const keywordOnly = neighborIdsOf("alice", idx, new Set(["keyword"]));
  assert.deepEqual([...keywordOnly].sort(), ["alice", "bob"]);

  // No layers active: just self
  const none = neighborIdsOf("alice", idx, new Set());
  assert.deepEqual([...none], ["alice"]);
});

test("neighborIdsOf MUST update when active-layer set changes (regression)", () => {
  const { edges } = fixture();
  const idx = buildEdgesByNode(edges);

  // Carol's ego network differs by layer — this is the bug surface:
  // before #28 the UI cached the result from the previous layer set.
  const ggCarol = neighborIdsOf("carol", idx, new Set(["give_get"]));
  assert.deepEqual([...ggCarol].sort(), ["alice", "carol"]);

  const topicCarol = neighborIdsOf("carol", idx, new Set(["topic"]));
  assert.deepEqual([...topicCarol].sort(), ["carol", "dave"]);

  // Switching layers MUST yield a different set, not stale data.
  assert.notDeepEqual([...ggCarol].sort(), [...topicCarol].sort());
});

// ---------- nodeMatchesFilters ---------------------------------------------
test("filters: empty filters match every node", () => {
  const { nodes } = fixture();
  for (const n of nodes) assert.equal(nodeMatchesFilters(n, {}), true);
});

test("filters: locations AND-restrict the result", () => {
  const { nodes } = fixture();
  const opts = { selectedLocations: new Set(["Boston"]) };
  const matched = nodes.filter((n) => nodeMatchesFilters(n, opts)).map((n) => n.id).sort();
  assert.deepEqual(matched, ["alice", "carol"]);
});

test("filters: topics use OR-semantics across selected topics", () => {
  const { nodes } = fixture();
  // 'data viz' matches alice + bob; 'pedagogy' matches carol.
  const opts = { selectedTopics: new Set(["data viz", "pedagogy"]) };
  const matched = nodes.filter((n) => nodeMatchesFilters(n, opts)).map((n) => n.id).sort();
  assert.deepEqual(matched, ["alice", "bob", "carol"]);
});

test("filters: scope='give' searches only the give bullets", () => {
  const { nodes } = fixture();
  // 'malware' appears in dave's GIVE bullets, no one's GET bullets
  let m = nodes.filter((n) =>
    nodeMatchesFilters(n, { searchTerm: "malware", searchScope: "give" })
  ).map((n) => n.id);
  assert.deepEqual(m, ["dave"]);

  // 'd3' appears in alice's GIVE and carol's GET — scoped to GIVE, only alice
  m = nodes.filter((n) =>
    nodeMatchesFilters(n, { searchTerm: "d3", searchScope: "give" })
  ).map((n) => n.id);
  assert.deepEqual(m, ["alice"]);

  // Same term, scoped to GET → only carol
  m = nodes.filter((n) =>
    nodeMatchesFilters(n, { searchTerm: "d3", searchScope: "get" })
  ).map((n) => n.id);
  assert.deepEqual(m, ["carol"]);

  // Same term, scope 'all' → both
  m = nodes.filter((n) =>
    nodeMatchesFilters(n, { searchTerm: "d3", searchScope: "all" })
  ).map((n) => n.id).sort();
  assert.deepEqual(m, ["alice", "carol"]);
});

test("filters: location + topic + search compose correctly", () => {
  const { nodes } = fixture();
  // Boston ∩ {data viz} ∩ search 'd3' in 'all' scope
  // - alice: Boston ✓, has 'data viz' ✓, give mentions d3 ✓
  // - carol: Boston ✓, no 'data viz' ✗
  const m = nodes.filter((n) =>
    nodeMatchesFilters(n, {
      selectedLocations: new Set(["Boston"]),
      selectedTopics: new Set(["data viz"]),
      searchTerm: "d3",
      searchScope: "all",
    })
  ).map((n) => n.id);
  assert.deepEqual(m, ["alice"]);
});

test("filters: empty 'give' bullets don't false-match on an empty haystack", () => {
  const node = {
    id: "x", name: "X", location: "Boston",
    keywords: [], auto_keywords: [],
    give: [], get: ["foo"],
  };
  assert.equal(
    nodeMatchesFilters(node, { searchTerm: "foo", searchScope: "give" }),
    false
  );
  assert.equal(
    nodeMatchesFilters(node, { searchTerm: "foo", searchScope: "get" }),
    true
  );
});
