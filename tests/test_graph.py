"""Pipeline output invariants.

These tests run against the real public/data/graph.json — they do NOT re-run
the pipeline (that's slow and downloads a 80MB model). Run with:

    python -m unittest tests.test_graph

The point is to catch regressions in build_graph: edge schema, top-k caps,
score floors, no self-loops, edge endpoints exist, etc.
"""
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO / "public" / "data" / "graph.json"

LAYERS = ("give_get", "topic", "keyword")
# Mirrors build_graph.py — keep these in sync if the floors change there.
SCORE_FLOORS = {"give_get": 0.30, "topic": 0.40, "keyword": 0.10}
TOP_K = 3


def load_graph():
    with GRAPH_PATH.open() as f:
        return json.load(f)


class GraphInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load_graph()
        cls.ids = {n["id"] for n in cls.g["nodes"]}

    def test_node_required_fields(self):
        for n in self.g["nodes"]:
            for field in ("id", "name", "location", "give", "get", "keywords", "photo"):
                self.assertIn(field, n, f"{n.get('id')} missing field {field}")
            self.assertIsInstance(n["give"], list)
            self.assertIsInstance(n["get"], list)
            # IDs must be unique slugs
            self.assertRegex(n["id"], r"^[a-z0-9\-]+$", n["id"])

    def test_node_ids_unique(self):
        ids = [n["id"] for n in self.g["nodes"]]
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        self.assertEqual(dupes, [], f"duplicate node ids: {dupes}")

    def test_edges_reference_existing_nodes(self):
        for layer in LAYERS:
            for e in self.g["edges"][layer]:
                self.assertIn(e["source"], self.ids, f"{layer}: bad source {e['source']}")
                self.assertIn(e["target"], self.ids, f"{layer}: bad target {e['target']}")
                self.assertNotEqual(e["source"], e["target"], f"{layer}: self-loop on {e['source']}")

    def test_edge_required_schema(self):
        for layer in LAYERS:
            for e in self.g["edges"][layer]:
                self.assertEqual(e["layer"], layer)
                self.assertIn("score", e)
                self.assertIn("directed", e)
                self.assertIn("why", e)
                self.assertIsInstance(e["score"], (int, float))
                self.assertGreaterEqual(e["score"], SCORE_FLOORS[layer],
                    f"{layer} edge below floor: {e}")

    def test_directed_only_give_get(self):
        for e in self.g["edges"]["give_get"]:
            self.assertTrue(e["directed"], f"give_get edge not directed: {e}")
        for layer in ("topic", "keyword"):
            for e in self.g["edges"][layer]:
                self.assertFalse(e["directed"], f"{layer} edge directed: {e}")

    def test_top_k_per_node_directed_layer(self):
        # Directed give_get layer: each node's out-degree must be ≤ TOP_K.
        # This is the row-wise cap in build_graph.top_k_edges.
        out_deg = Counter(e["source"] for e in self.g["edges"]["give_get"])
        over = [(k, v) for k, v in out_deg.items() if v > TOP_K]
        self.assertEqual(over, [], f"give_get out-degree > {TOP_K}: {over}")

    def test_top_k_total_bound_undirected(self):
        # Undirected layers (topic, keyword) start as N row-wise top-k picks
        # then get deduped on sorted-pair, so total edge count ≤ N * TOP_K.
        # (A single node can still appear in many others' top-3, so per-node
        # degree is unbounded — that's by design.)
        n = len(self.g["nodes"])
        for layer in ("topic", "keyword"):
            self.assertLessEqual(
                len(self.g["edges"][layer]), n * TOP_K,
                f"{layer} edge count exceeds N*TOP_K"
            )

    def test_no_duplicate_edges_per_layer(self):
        # Within a directed layer (s, t) is uniquely keyed; within an undirected
        # layer, the unordered pair must appear at most once.
        for layer in LAYERS:
            seen = set()
            for e in self.g["edges"][layer]:
                if e.get("directed"):
                    key = (e["source"], e["target"])
                else:
                    key = tuple(sorted((e["source"], e["target"])))
                self.assertNotIn(key, seen, f"{layer} duplicate edge: {key}")
                seen.add(key)

    def test_give_get_why_strings_nonempty(self):
        # The whole point of give→get edges is the human-readable "why" — if
        # we ever ship empty whys, the UI degrades to a meaningless arrow.
        for e in self.g["edges"]["give_get"]:
            self.assertTrue(e["why"], f"give_get edge missing 'why': {e}")

    def test_filters_match_node_universe(self):
        filters = self.g.get("filters", {})
        locations_set = {n["location"] for n in self.g["nodes"] if n.get("location")}
        self.assertEqual(set(filters.get("locations", [])), locations_set)

    def test_meta_count_matches_nodes(self):
        self.assertEqual(self.g["meta"]["n_researchers"], len(self.g["nodes"]))


if __name__ == "__main__":
    unittest.main()
