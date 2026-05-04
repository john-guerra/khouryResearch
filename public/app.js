// Khoury Researcher Connections — D3 force graph with three layer toggles,
// search/location filters, edge hover tooltips with "why this connection",
// and a side panel showing per-researcher details and top connections per layer.
//
// All dynamic UI is built with DOM methods (createElement + textContent) so
// we never touch innerHTML — escaping is guaranteed structurally.

import {
  LAYERS,
  normalizeTopic,
  topicsForNode,
  buildEdgesByNode,
  neighborIdsOf as neighborIdsOfPure,
  nodeMatchesFilters,
  hasActiveFilters as hasActiveFiltersPure,
} from "./lib.js";

const LAYER_LABELS = {
  give_get: "Give → Get",
  topic: "Topic similarity",
  keyword: "Shared keywords",
};

const state = {
  data: null,
  scoreRange: { give_get: [0.30, 0.65], topic: [0.40, 0.75], keyword: [0.60, 0.85] },
  active: new Set(["give_get"]),
  searchTerm: "",
  searchScope: "all", // 'all' | 'give' | 'get'
  selectedLocations: new Set(),
  selectedTopics: new Set(),
  selectedId: null,
  focusOnSelection: true,        // on by default — clicking a researcher dims the rest
  showLabels: false,
  nodeIndex: new Map(),
  edgesByNode: { give_get: new Map(), topic: new Map(), keyword: new Map() },
  // populated in buildGraph for the simulation/animation code to reference.
  allLinkData: [],
  nodes: [],
};

const svg = d3.select("#graph");
const linkGroup = svg.append("g").attr("class", "links");
const nodeGroup = svg.append("g").attr("class", "nodes");
const defs = svg.append("defs");

// Arrowhead for give→get edges.
// IMPORTANT: markerUnits="userSpaceOnUse" — without it, SVG defaults to
// strokeWidth-relative sizing, so markerWidth and refX scale with each line's
// stroke (which varies with edge score). That made arrows float off the node.
// Geometry with userSpaceOnUse: viewBox is 10×10, scaled to markerWidth user
// units. Tip at viewBox (10, 0) = (markerWidth, 0). refX places the marker so
// (refX_viewBox * markerWidth/10) aligns with line endpoint. Gap from line
// endpoint to arrow tip = markerWidth * (refX/10 - 1). With markerWidth=8 and
// refX=32: gap = 8 * 2.2 = 17.6 ≈ node radius (18).
defs.append("marker")
  .attr("id", "arrow-give-get")
  .attr("viewBox", "0 -5 10 10")
  .attr("refX", 32)
  .attr("refY", 0)
  .attr("markerUnits", "userSpaceOnUse")
  .attr("markerWidth", 8)
  .attr("markerHeight", 8)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-5L10,0L0,5")
  .attr("fill", "var(--give-get)");

let simulation = null;
const tooltip = makeTooltip();

// Semantic zoom: the d3.zoom transform is applied to POSITIONS only, not
// to a parent <g>. Photos, labels, and stroke widths keep their absolute
// pixel size — zooming reveals more detail by spreading the layout, not by
// magnifying pixels. tx/ty also do panning since transform.applyX bakes in
// translation.
let zoomTransform = d3.zoomIdentity;
function projectX(d) { return zoomTransform.applyX(xScale(d.x)); }
function projectY(d) { return zoomTransform.applyY(yScale(d.y)); }

// Virtual scales mapping simulation coordinates → screen coordinates.
// Domain is recomputed on each tick (lerped for smoothness) to follow the
// current bounding box of all nodes; range is the viewport minus a margin
// so node photos and labels always render fully inside the canvas.
const xScale = d3.scaleLinear();
const yScale = d3.scaleLinear();
let xDomain = [-200, 200];
let yDomain = [-200, 200];
const NODE_R = 26;
const VIEWPORT_MARGIN = NODE_R + 22;     // photo radius + halo + label baseline
const VIEWPORT_BOTTOM_MARGIN = NODE_R + 38; // extra room for the label below the node
const DOMAIN_PAD = 20;                    // simulation-space padding inside the domain
const DOMAIN_LERP = 0.12;                 // 0..1; higher = scale tracks faster but jitters more

function lerp(a, b, t) { return a + (b - a) * t; }

// Per-layer score normalization. Each layer has a different natural range:
//   give_get: cross-encoder sigmoid score ≈ 0.55–0.95 for real matches
//   topic:    bi-encoder cosine ≈ 0.4–0.75
//   keyword:  centroid cosine ≈ 0.6–0.9
// These defaults are user-tunable in the controls panel — see #score-range
// section. The live values live on state.scoreRange so changes re-style edges
// without a graph rebuild.
const DEFAULT_SCORE_RANGE = {
  give_get: [0.30, 0.65],
  topic:    [0.40, 0.75],
  keyword:  [0.60, 0.85],
};
const SCORE_RANGE_STORAGE_KEY = "khoury-score-range-v2";
function loadScoreRange() {
  try {
    const raw = localStorage.getItem(SCORE_RANGE_STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULT_SCORE_RANGE);
    const parsed = JSON.parse(raw);
    // Sanity-check shape; fall back to defaults if malformed.
    for (const layer of ["give_get", "topic", "keyword"]) {
      if (!Array.isArray(parsed[layer]) || parsed[layer].length !== 2) {
        return structuredClone(DEFAULT_SCORE_RANGE);
      }
    }
    return parsed;
  } catch {
    return structuredClone(DEFAULT_SCORE_RANGE);
  }
}
function saveScoreRange() {
  try { localStorage.setItem(SCORE_RANGE_STORAGE_KEY, JSON.stringify(state.scoreRange)); }
  catch { /* private mode / quota exceeded — ignore */ }
}
function normalizedScore(d) {
  const [lo, hi] = state.scoreRange[d.layer] || [0, 1];
  return Math.max(0, Math.min(1, (d.score - lo) / Math.max(0.0001, hi - lo)));
}
function baseStrokeOpacity(d) { return 0.5 + normalizedScore(d) * 0.4; }   // 0.5 → 0.9
function baseStrokeWidth(d)   { return 1.5 + normalizedScore(d) * 2.0; }   // 1.5 → 3.5

function updateScales() {
  if (!state.nodes.length) return;
  const xs = state.nodes.map((n) => n.x);
  const ys = state.nodes.map((n) => n.y);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  // Avoid degenerate domains when all nodes are on a single line.
  if (xMax - xMin < 1) { xMax += 50; xMin -= 50; }
  if (yMax - yMin < 1) { yMax += 50; yMin -= 50; }
  xMin -= DOMAIN_PAD; xMax += DOMAIN_PAD;
  yMin -= DOMAIN_PAD; yMax += DOMAIN_PAD;

  xDomain = [lerp(xDomain[0], xMin, DOMAIN_LERP), lerp(xDomain[1], xMax, DOMAIN_LERP)];
  yDomain = [lerp(yDomain[0], yMin, DOMAIN_LERP), lerp(yDomain[1], yMax, DOMAIN_LERP)];

  const w = svg.node().clientWidth;
  const h = svg.node().clientHeight;
  xScale.domain(xDomain).range([VIEWPORT_MARGIN, w - VIEWPORT_MARGIN]);
  yScale.domain(yDomain).range([VIEWPORT_MARGIN, h - VIEWPORT_BOTTOM_MARGIN]);
}

function initPanels() {
  // Default: on mobile/narrow viewports both panels start collapsed.
  if (window.innerWidth <= 768) {
    document.body.classList.add("controls-collapsed", "detail-collapsed");
  }
  document.getElementById("toggle-controls").addEventListener("click", () => {
    document.body.classList.toggle("controls-collapsed");
    // On mobile, only one overlay panel open at a time.
    if (window.innerWidth <= 768 && !document.body.classList.contains("controls-collapsed")) {
      document.body.classList.add("detail-collapsed");
    }
  });
  document.getElementById("toggle-detail").addEventListener("click", () => {
    document.body.classList.toggle("detail-collapsed");
    if (window.innerWidth <= 768 && !document.body.classList.contains("detail-collapsed")) {
      document.body.classList.add("controls-collapsed");
    }
  });
  // Tap on the backdrop closes the open panel
  document.body.addEventListener("click", (e) => {
    if (window.innerWidth > 768) return;
    if (e.target !== document.body) return;
    document.body.classList.add("controls-collapsed", "detail-collapsed");
  });
}

async function init() {
  state.scoreRange = loadScoreRange();
  initPanels();
  const res = await fetch("data/graph.json");
  state.data = await res.json();
  state.data.nodes.forEach((n) => state.nodeIndex.set(n.id, n));

  state.edgesByNode = buildEdgesByNode(state.data.edges);

  document.getElementById("meta-counts").textContent =
    `${state.data.meta.n_researchers} researchers • model: ${state.data.meta.model}`;

  buildLocationChips();
  buildTopicChips();
  buildScoreRangeControls();
  bindControls();
  buildGraph();
  updateLayerCounts();
  updateViewTitle();

  // Deep-link support: if the URL already carries #<slug>, select that
  // researcher after the graph is built. Then keep selection in sync with
  // the back/forward buttons or any manual hash edit.
  applyHashSelection();
  window.addEventListener("hashchange", applyHashSelection);
}

function buildTopicChips() {
  const container = document.getElementById("topic-chips");
  while (container.firstChild) container.removeChild(container.firstChild);

  // Tally how many researchers carry each normalized keyword. We only chip
  // topics shared by at least 2 researchers — singletons would balloon the
  // panel without enabling any cross-researcher filtering.
  const counts = new Map();
  for (const n of state.data.nodes) {
    for (const t of topicsForNode(n)) {
      counts.set(t, (counts.get(t) || 0) + 1);
    }
  }
  const shared = [...counts.entries()]
    .filter(([, c]) => c >= 2)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (!shared.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No keywords shared between researchers.";
    container.appendChild(empty);
    return;
  }
  const max = Math.max(1, ...shared.map(([, c]) => c));
  for (const [topic, count] of shared) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.style.setProperty("--scent", `${(count / max) * 100}%`);
    const label = document.createElement("span");
    label.textContent = topic;
    chip.appendChild(label);
    const cnt = document.createElement("span");
    cnt.className = "count-paren";
    cnt.textContent = `(${count})`;
    chip.appendChild(cnt);
    chip.addEventListener("click", () => {
      if (state.selectedTopics.has(topic)) state.selectedTopics.delete(topic);
      else state.selectedTopics.add(topic);
      chip.classList.toggle("active", state.selectedTopics.has(topic));
      applyFilters();
    });
    container.appendChild(chip);
  }
}

function buildLocationChips() {
  const container = document.getElementById("location-chips");
  while (container.firstChild) container.removeChild(container.firstChild);

  // Count researchers per location for the scented bars + parenthetical counts.
  const counts = new Map();
  for (const n of state.data.nodes) {
    counts.set(n.location, (counts.get(n.location) || 0) + 1);
  }
  const max = Math.max(1, ...counts.values());

  for (const loc of state.data.filters.locations) {
    const count = counts.get(loc) || 0;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.style.setProperty("--scent", `${(count / max) * 100}%`);
    const label = document.createElement("span");
    label.textContent = loc;
    chip.appendChild(label);
    const cnt = document.createElement("span");
    cnt.className = "count-paren";
    cnt.textContent = `(${count})`;
    chip.appendChild(cnt);
    chip.addEventListener("click", () => {
      if (state.selectedLocations.has(loc)) state.selectedLocations.delete(loc);
      else state.selectedLocations.add(loc);
      chip.classList.toggle("active", state.selectedLocations.has(loc));
      applyFilters();
    });
    container.appendChild(chip);
  }
}

// Per-layer score-range sliders (lo, hi). Edges below lo render at the floor
// opacity/width; edges at or above hi render at the cap. The bias toward
// "always somewhat visible" (floor opacity 0.5) is intentional — even a thin
// edge has signal. Changes live-update existing edges; values persist via
// localStorage so a researcher's tuning survives reload.
function buildScoreRangeControls() {
  const container = document.getElementById("score-range");
  if (!container) return;
  while (container.firstChild) container.removeChild(container.firstChild);

  for (const layer of LAYERS) {
    const block = document.createElement("div");
    block.className = "range-block";
    const heading = document.createElement("div");
    heading.className = "range-heading";
    heading.innerHTML = "";
    const swatch = document.createElement("span");
    swatch.className = `range-swatch layer-${layer.replace("_", "-")}`;
    const label = document.createElement("span");
    label.textContent = LAYER_LABELS[layer];
    heading.appendChild(swatch);
    heading.appendChild(label);
    block.appendChild(heading);

    const [lo0, hi0] = state.scoreRange[layer];
    const loInput = makeRangeInput("Faint at", lo0, (val) => {
      // Clamp lo to never exceed hi - 0.01 so the normalized denominator stays valid.
      const cur = state.scoreRange[layer];
      state.scoreRange[layer] = [Math.min(val, cur[1] - 0.01), cur[1]];
      onScoreRangeChange();
    });
    const hiInput = makeRangeInput("Bold at", hi0, (val) => {
      const cur = state.scoreRange[layer];
      state.scoreRange[layer] = [cur[0], Math.max(val, cur[0] + 0.01)];
      onScoreRangeChange();
    });
    block.appendChild(loInput.row);
    block.appendChild(hiInput.row);
    block.dataset.layer = layer;
    block._loInput = loInput;
    block._hiInput = hiInput;
    container.appendChild(block);
  }

  document.getElementById("reset-ranges").addEventListener("click", () => {
    state.scoreRange = structuredClone(DEFAULT_SCORE_RANGE);
    refreshScoreRangeUI();
    onScoreRangeChange();
  });
}

function makeRangeInput(label, value, onInput) {
  const row = document.createElement("label");
  row.className = "range-row";
  const lbl = document.createElement("span");
  lbl.className = "range-label";
  lbl.textContent = label;
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "1";
  slider.step = "0.01";
  slider.value = String(value);
  const num = document.createElement("span");
  num.className = "range-num";
  num.textContent = value.toFixed(2);
  slider.addEventListener("input", () => {
    const v = parseFloat(slider.value);
    num.textContent = v.toFixed(2);
    onInput(v);
  });
  row.appendChild(lbl);
  row.appendChild(slider);
  row.appendChild(num);
  return { row, slider, num };
}

function refreshScoreRangeUI() {
  for (const block of document.querySelectorAll("#score-range .range-block")) {
    const [lo, hi] = state.scoreRange[block.dataset.layer];
    block._loInput.slider.value = String(lo);
    block._loInput.num.textContent = lo.toFixed(2);
    block._hiInput.slider.value = String(hi);
    block._hiInput.num.textContent = hi.toFixed(2);
  }
}

function onScoreRangeChange() {
  saveScoreRange();
  restyleEdges();
}

// Re-apply stroke-opacity / stroke-width based on current state.scoreRange.
// Cheap: it just walks the existing edge selection — no force tick.
function restyleEdges() {
  linkGroup.selectAll("line.link")
    .attr("stroke-opacity", (d) => baseStrokeOpacity(d))
    .attr("stroke-width", (d) => baseStrokeWidth(d));
}

function bindControls() {
  // Layer pills: a click toggles active state. Same downstream effects as the
  // old checkbox listener — just driven by a button's class state.
  for (const pill of document.querySelectorAll(".layer-pill[data-layer]")) {
    pill.addEventListener("click", () => {
      const layer = pill.dataset.layer;
      const turningOn = !state.active.has(layer);
      if (turningOn) state.active.add(layer);
      else state.active.delete(layer);
      pill.classList.toggle("active", turningOn);
      pill.setAttribute("aria-pressed", String(turningOn));
      animateLayerChange();
      updateLayerCounts();
      updateViewTitle();
      applyFilters();
      if (state.selectedId) renderDetail(state.selectedId);
    });
  }

  document.getElementById("show-labels").addEventListener("change", (e) => {
    state.showLabels = e.target.checked;
    document.querySelector("svg#graph").classList.toggle("show-labels", state.showLabels);
  });

  document.getElementById("search").addEventListener("input", (e) => {
    state.searchTerm = e.target.value.trim().toLowerCase();
    applyFilters();
  });

  for (const btn of document.querySelectorAll(".scope-btn")) {
    btn.addEventListener("click", () => {
      state.searchScope = btn.dataset.scope;
      document.querySelectorAll(".scope-btn").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      applyFilters();
    });
  }

  document.getElementById("reset-filters").addEventListener("click", resetFilters);
  document.getElementById("empty-clear").addEventListener("click", resetFilters);

  // Keyboard shortcuts: '/' focuses the search box, Esc clears selection
  // first, then filters. Skip when the user is already typing in any input
  // (search box, etc.) so we don't intercept their normal typing.
  document.addEventListener("keydown", (event) => {
    const tag = (event.target.tagName || "").toLowerCase();
    const typingInInput = tag === "input" || tag === "textarea" || event.target.isContentEditable;

    if (event.key === "/" && !typingInInput) {
      event.preventDefault();
      const search = document.getElementById("search");
      search.focus();
      search.select();
      return;
    }
    if (event.key === "Escape") {
      // Esc inside the search box first blurs it; otherwise it does nothing
      // visible. Outside an input it cascades: selection → filters.
      if (tag === "input") {
        event.target.blur();
        return;
      }
      if (state.selectedId) {
        clearSelection();
      } else if (hasActiveFilters()) {
        resetFilters();
      }
    }
  });
}

function hasActiveFilters() {
  return hasActiveFiltersPure({
    searchTerm: state.searchTerm,
    searchScope: state.searchScope,
    selectedLocations: state.selectedLocations,
    selectedTopics: state.selectedTopics,
  });
}

function resetFilters() {
  state.searchTerm = "";
  state.searchScope = "all";
  state.selectedLocations.clear();
  state.selectedTopics.clear();
  document.getElementById("search").value = "";
  document.querySelectorAll("#location-chips .chip, #topic-chips .chip")
    .forEach((c) => c.classList.remove("active"));
  document.querySelectorAll(".scope-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.scope === "all")
  );
  applyFilters();
}

function updateLayerCounts() {
  for (const layer of LAYERS) {
    document.getElementById(`count-${layer}`).textContent = state.data.edges[layer].length;
  }
}

// Dynamic title above the chart. Surfaces what the user is currently looking
// at: the active layer set + (if applicable) the selected researcher's name.
// Kept short — the goal is "I see the title and know what these dots mean".
function updateViewTitle() {
  const el = document.getElementById("view-title");
  if (!el) return;
  const titles = {
    give_get: "Give → Get matches",
    topic: "Topic similarity",
    keyword: "Shared keywords",
  };
  const active = LAYERS.filter((l) => state.active.has(l));
  let layerPhrase;
  if (active.length === 0) layerPhrase = "no connection layer active";
  else if (active.length === 3) layerPhrase = "all connections";
  else layerPhrase = active.map((l) => titles[l]).join(" + ");

  if (state.selectedId) {
    const node = state.nodeIndex.get(state.selectedId);
    const name = node ? node.name : state.selectedId;
    el.textContent = state.focusOnSelection
      ? `${name}'s network — ${layerPhrase}`
      : `${name} selected — ${layerPhrase}`;
  } else {
    // Sentence-case the leading word; `layerPhrase` already starts with a
    // descriptor, but for the no-selection case we want a complete clause.
    el.textContent = active.length === 0
      ? "No connection layer active — turn one on above"
      : `Showing ${layerPhrase}`;
  }
}

function buildGraph() {
  const { nodes, edges } = state.data;
  const allEdges = [...edges.give_get, ...edges.topic, ...edges.keyword];

  const nodeById = state.nodeIndex;
  const linkData = allEdges.map((e) => ({
    ...e,
    source: nodeById.get(e.source),
    target: nodeById.get(e.target),
  }));
  state.allLinkData = linkData;
  state.nodes = nodes;

  const R = 18;

  // We clip headshots to a circle via CSS clip-path on .face-img — works
  // reliably with the per-node transforms. SVG <clipPath> needs explicit
  // clipPathUnits handling that interacts oddly with d3-force translates.

  // Wider invisible hit-area on each link so hovering is easy at thin strokes.
  const linkSel = linkGroup.selectAll("g.link-grp")
    .data(linkData, (d) => `${d.source.id}|${d.target.id}|${d.layer}`)
    .enter()
    .append("g")
    .attr("class", "link-grp")
    .style("pointer-events", "stroke");

  linkSel.append("line")
    .attr("class", "link-hit")
    .attr("stroke", "transparent")
    .attr("stroke-width", 12)
    .attr("fill", "none")
    .style("pointer-events", "stroke")
    .on("mousemove", (event, d) => showEdgeTooltip(event, d))
    .on("mouseleave", () => tooltip.hide());

  linkSel.append("line")
    .attr("class", (d) => `link link-${d.layer.replace("_", "-")}`)
    .attr("stroke-width", (d) => baseStrokeWidth(d))
    .attr("stroke-opacity", (d) => baseStrokeOpacity(d))
    .attr("marker-end", (d) => (d.layer === "give_get" ? "url(#arrow-give-get)" : null));

  // Edge labels — short hint per layer; full "why" still surfaces in the hover tooltip.
  linkSel.append("text")
    .attr("class", "link-label")
    .text((d) => edgeLabelFor(d));

  const node = nodeGroup.selectAll("g.node")
    .data(nodes, (d) => d.id)
    .enter()
    .append("g")
    .attr("class", "node")
    .on("click", (event, d) => selectNode(d.id))
    .on("mousemove", (event, d) => showNodeTooltip(event, d))
    .on("mouseleave", () => tooltip.hide())
    .call(d3.drag()
      // Drag is opt-in: hold Shift to grab a node. Plain clicks/taps still
      // select the researcher without nudging the layout, which is what most
      // people expect from a graph viewer.
      .filter((event) => event.shiftKey)
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        // Drag events arrive in screen-space. Invert the zoom AND the scales
        // to get back to simulation-space coords (where the forces live).
        d.fx = xScale.invert(zoomTransform.invertX(event.x));
        d.fy = yScale.invert(zoomTransform.invertY(event.y));
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }));

  node.append("circle")
    .attr("class", "halo")
    .attr("r", R + 6);
  node.append("image")
    .attr("class", "face-img")
    .attr("href", (d) => d.photo)
    .attr("x", -R)
    .attr("y", -R)
    .attr("width", R * 2)
    .attr("height", R * 2)
    .attr("preserveAspectRatio", "xMidYMid slice");
  node.append("circle")
    .attr("class", "frame")
    .attr("r", R)
    .attr("fill", "none");
  node.append("text")
    .attr("class", "label")
    .attr("y", R + 14)
    .text((d) => d.name);

  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  svg.attr("viewBox", [0, 0, width, height]);

  // Simulation runs on ACTIVE edges only — toggling a layer changes the layout,
  // not just visibility. animateLayerChange() updates the link force.
  const activeLinks = linkData.filter((d) => state.active.has(d.layer));
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(activeLinks).id((d) => d.id).distance((d) => 110 + (1 - d.score) * 90).strength(0.35))
    .force("charge", d3.forceManyBody().strength(-700))
    .force("collide", d3.forceCollide().radius(R + 22))
    // .force("center", d3.forceCenter(width / 2, height / 2)) // when nodes get loose center doesn't work well
    .force("x", d3.forceX(width / 2).strength(0.05))
    .force("y", d3.forceY(height / 2).strength(0.05))
    .on("tick", reflow);

  function reflow() {
    updateScales();
    linkGroup.selectAll("line").each(function (d) {
      d3.select(this)
        .attr("x1", projectX(d.source))
        .attr("y1", projectY(d.source))
        .attr("x2", projectX(d.target))
        .attr("y2", projectY(d.target));
    });
    linkGroup.selectAll("text.link-label")
      .attr("x", (d) => (projectX(d.source) + projectX(d.target)) / 2)
      .attr("y", (d) => (projectY(d.source) + projectY(d.target)) / 2 - 3);
    node.attr("transform", (d) => `translate(${projectX(d)},${projectY(d)})`);
  }

  svg.call(d3.zoom()
    .scaleExtent([0.4, 4])
    .on("zoom", (event) => {
      // Semantic zoom: stash the transform and re-run the reflow. We do NOT
      // apply a transform to the parent groups — that would scale photo and
      // text size with the zoom (geometric zoom). Reflowing per-element keeps
      // every visual asset at its native pixel size.
      zoomTransform = event.transform;
      reflow();
    }));

  // Double-click on empty canvas clears the current selection. Node clicks
  // already handle selection themselves; we only fire when the dblclick target
  // is the SVG root (not a node/edge inside it). d3.zoom owns dblclick for
  // zoom-in by default, so we disable that and use it for clear-focus instead.
  svg.on("dblclick.zoom", null);
  svg.on("dblclick", (event) => {
    if (event.target === svg.node()) clearSelection();
  });

  renderEdges();
  updateLayerCounts();
}

function renderEdges() {
  linkGroup.selectAll("g.link-grp")
    .style("display", (d) => (state.active.has(d.layer) ? null : "none"));
}

function edgeLabelFor(d) {
  // Keep labels short — full text is in the hover tooltip and side panel.
  if (d.layer === "give_get") {
    if (!d.why) return d.score.toFixed(2);
    // "give-bullet" → "get-bullet"; show first ~24 chars of each side
    const m = d.why.match(/^"([^"]+)"\s*[→-]+\s*"([^"]+)"$/);
    if (m) {
      const left = m[1].length > 22 ? m[1].slice(0, 21) + "…" : m[1];
      const right = m[2].length > 22 ? m[2].slice(0, 21) + "…" : m[2];
      return `${left} → ${right}`;
    }
    return d.why.slice(0, 50);
  }
  if (d.layer === "keyword") {
    if (!d.why) return d.score.toFixed(2);
    return d.why.slice(0, 36);
  }
  // topic — just the score
  return d.score.toFixed(2);
}

function animateLayerChange() {
  if (!simulation) return;

  // Step 1 — fade edges in/out via class-toggled display + CSS transition on stroke-opacity.
  linkGroup.selectAll("g.link-grp").each(function (d) {
    const grp = d3.select(this);
    const active = state.active.has(d.layer);
    const line = grp.select("line.link");
    if (active) {
      grp.style("display", null);
      // re-trigger transition from 0 to target opacity
      line.attr("stroke-opacity", 0)
        .transition().duration(450).ease(d3.easeCubicOut)
        .attr("stroke-opacity", baseStrokeOpacity(d));
    } else {
      line.transition().duration(300).ease(d3.easeCubicIn)
        .attr("stroke-opacity", 0)
        .on("end", function () { grp.style("display", "none"); });
    }
  });

  // Step 2 — swap the active link set into the simulation's force.
  const activeLinks = state.allLinkData.filter((d) => state.active.has(d.layer));
  simulation.force("link").links(activeLinks);

  // Step 3 — lock every node at its current position, then release them one at
  // a time so the layout cascades instead of jumping. Order is by current x
  // coord so the wave moves left → right (visually pleasing).
  const nodes = state.nodes;
  nodes.forEach((n) => { n.fx = n.x; n.fy = n.y; });
  const order = [...nodes].sort((a, b) => a.x - b.x);
  simulation.alpha(0.55).restart();

  // Wait for edges to start fading before we begin the cascade.
  const startDelay = 220;
  const stagger = Math.max(35, Math.min(70, 1400 / nodes.length));
  order.forEach((n, i) => {
    setTimeout(() => {
      n.fx = null;
      n.fy = null;
      simulation.alpha(Math.max(simulation.alpha(), 0.3)).restart();
    }, startDelay + i * stagger);
  });
}

function neighborIdsOf(id) {
  return neighborIdsOfPure(id, state.edgesByNode, state.active);
}

function applyFilters() {
  const baseMatches = (n) => nodeMatchesFilters(n, {
    searchTerm: state.searchTerm,
    searchScope: state.searchScope,
    selectedLocations: state.selectedLocations,
    selectedTopics: state.selectedTopics,
  });

  // Focus mode: when enabled, only the selected researcher and their direct
  // neighbors are considered "matching" — the rest fade out.
  const focusSet = state.focusOnSelection && state.selectedId ? neighborIdsOf(state.selectedId) : null;
  const matches = (n) => baseMatches(n) && (!focusSet || focusSet.has(n.id));

  nodeGroup.selectAll("g.node").classed("dimmed", (d) => !matches(d));
  linkGroup.selectAll("g.link-grp").classed("dimmed", (d) => {
    if (!state.active.has(d.layer)) return true;
    if (focusSet && d.source.id !== state.selectedId && d.target.id !== state.selectedId) return true;
    return !matches(d.source) || !matches(d.target);
  });
  linkGroup.selectAll("line.link").classed("dimmed", function () {
    return d3.select(this.parentNode).classed("dimmed");
  });

  // Empty state: only triggered by filter/search/topic — focus mode always
  // keeps the selected node visible, so it can't on its own produce zero
  // visible nodes. Counting baseMatches keeps the message scoped to filters.
  const baseVisible = state.nodes.reduce((acc, n) => acc + (baseMatches(n) ? 1 : 0), 0);
  const showEmpty = baseVisible === 0 && hasActiveFilters();
  const empty = document.getElementById("empty-state");
  if (empty) empty.hidden = !showEmpty;
}

function updateFocusModeClass() {
  // The `.focus-mode` class on body deepens the dim of unrelated nodes/edges.
  // Only meaningful when focus is requested AND a researcher is selected.
  document.body.classList.toggle(
    "focus-mode",
    state.focusOnSelection && !!state.selectedId
  );
}

function selectNode(id) {
  state.selectedId = id;
  nodeGroup.selectAll("g.node").classed("selected", (d) => d.id === id);
  linkGroup.selectAll("g.link-grp").classed("label-on", (d) => {
    return d.source.id === id || d.target.id === id;
  });
  updateFocusModeClass();
  applyFilters();           // re-evaluate focus mode if it's on
  renderDetail(id);
  updateViewTitle();
  syncHash(id);
}

function clearSelection() {
  state.selectedId = null;
  // Don't reset focusOnSelection — it's a user preference, the next selection
  // should respect whatever they had on. Default state is true (focus on).
  nodeGroup.selectAll("g.node").classed("selected", false);
  linkGroup.selectAll("g.link-grp").classed("label-on", false);
  updateFocusModeClass();
  applyFilters();
  // Restore the empty placeholder in the detail panel.
  const root = document.getElementById("detail");
  while (root.firstChild) root.removeChild(root.firstChild);
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = "Click any researcher to see their give/get bullets and top connections.";
  root.appendChild(empty);
  updateViewTitle();
  syncHash(null);
}

// URL hash <-> selection sync. We use replaceState so clicking around doesn't
// pollute the back stack with one entry per node, but still updates the
// address bar so the URL is shareable. hashchange fires on back/forward and
// on manual edits — those should drive the selection.
function syncHash(id) {
  const desired = id ? `#${id}` : "";
  if (location.hash === desired) return;
  // Guard against the ensuing hashchange driving us back into selectNode.
  syncHash._suppress = true;
  try {
    if (id) {
      history.replaceState(null, "", `${location.pathname}${location.search}#${id}`);
    } else {
      history.replaceState(null, "", `${location.pathname}${location.search}`);
    }
  } finally {
    // Clear the flag on the next tick — replaceState doesn't fire hashchange,
    // but a quick toggle protects us if a future browser ever does.
    setTimeout(() => { syncHash._suppress = false; }, 0);
  }
}

function applyHashSelection() {
  if (syncHash._suppress) return;
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw) {
    if (state.selectedId) clearSelection();
    return;
  }
  if (!state.nodeIndex.has(raw)) return;     // unknown slug — leave state alone
  if (raw === state.selectedId) return;
  selectNode(raw);
}

function setFocusOnSelection(on) {
  state.focusOnSelection = !!on;
  updateFocusModeClass();
  applyFilters();
  updateViewTitle();
}

function showEdgeTooltip(event, d) {
  const a = d.source, b = d.target;
  const arrow = d.layer === "give_get" ? " → " : " ↔ ";
  const lines = [
    `${a.name}${arrow}${b.name}`,
    `${LAYER_LABELS[d.layer]} • score ${d.score.toFixed(2)}`,
  ];
  if (d.why) lines.push(d.why);
  tooltip.showLines(event, lines, `tooltip-${d.layer.replace("_", "-")}`);
}

function showNodeTooltip(event, d) {
  const lines = [d.name];
  if (d.primary_area) lines.push(d.primary_area);
  if (d.location) lines.push(d.location);
  tooltip.showLines(event, lines);
}

function makeTooltip() {
  const el = document.createElement("div");
  el.id = "tooltip";
  el.style.cssText = [
    "position: fixed",
    "pointer-events: none",
    "background: var(--panel)",
    "color: var(--text)",
    "border: 1px solid var(--border)",
    "border-radius: 6px",
    "padding: 8px 10px",
    "font-size: 12px",
    "max-width: 360px",
    "box-shadow: 0 6px 20px rgba(0,0,0,0.45)",
    "z-index: 1000",
    "opacity: 0",
    "transition: opacity 0.1s ease",
  ].join("; ");
  document.body.appendChild(el);

  return {
    showLines(event, lines, extraClass) {
      // Clear contents safely
      while (el.firstChild) el.removeChild(el.firstChild);
      el.className = "";
      if (extraClass) el.classList.add(extraClass);
      lines.forEach((line, i) => {
        const div = document.createElement("div");
        div.textContent = line;
        if (i === 0) div.style.fontWeight = "600";
        if (i === 1) { div.style.fontSize = "11px"; div.style.color = "var(--muted)"; div.style.marginTop = "1px"; }
        if (i >= 2) { div.style.marginTop = "4px"; div.style.fontStyle = "italic"; div.style.color = "var(--muted)"; }
        el.appendChild(div);
      });
      el.style.left = `${event.clientX + 14}px`;
      el.style.top = `${event.clientY + 12}px`;
      el.style.opacity = "1";
    },
    hide() { el.style.opacity = "0"; },
  };
}

function renderDetail(id) {
  const node = state.nodeIndex.get(id);
  if (!node) return;
  const root = document.getElementById("detail");
  while (root.firstChild) root.removeChild(root.firstChild);

  // Selection action bar — clear button + focus toggle.
  const actions = el("div", "selection-actions");
  const tag = el("span", "selection-tag", "Selected");
  actions.appendChild(tag);

  const focusLabel = document.createElement("label");
  focusLabel.className = "focus-toggle";
  const focusInput = document.createElement("input");
  focusInput.type = "checkbox";
  focusInput.checked = state.focusOnSelection;
  focusInput.addEventListener("change", (e) => setFocusOnSelection(e.target.checked));
  focusLabel.appendChild(focusInput);
  focusLabel.appendChild(document.createTextNode(" Focus"));
  focusLabel.title = "Show only this researcher and their direct connections";
  actions.appendChild(focusLabel);

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "selection-clear";
  clearBtn.textContent = "Clear ✕";
  clearBtn.title = "Clear selection";
  clearBtn.addEventListener("click", clearSelection);
  actions.appendChild(clearBtn);

  root.appendChild(actions);

  // Header
  const head = el("div", "head");
  if (node.photo) {
    const img = document.createElement("img");
    img.src = node.photo;
    img.alt = node.name;
    head.appendChild(img);
  }
  const headText = el("div");
  headText.appendChild(el("h3", null, node.name));
  if (node.location) headText.appendChild(el("div", "sub", node.location));
  const badges = el("div", "badges");
  if (node.primary_area) badges.appendChild(el("span", "badge area", node.primary_area));
  if (node.secondary_area) badges.appendChild(el("span", "badge", node.secondary_area));
  if (node.provenance) badges.appendChild(el("span", "badge", node.provenance));
  headText.appendChild(badges);
  head.appendChild(headText);
  root.appendChild(head);

  // Email + links
  if (node.email) {
    const emailRow = el("div", "links");
    emailRow.style.marginTop = "6px";
    const a = document.createElement("a");
    a.href = `mailto:${node.email}`;
    a.textContent = node.email;
    emailRow.appendChild(a);
    root.appendChild(emailRow);
  }
  if (node.links && node.links.length) {
    const linksRow = el("div", "links");
    linksRow.style.marginTop = "4px";
    node.links.forEach((href, i) => {
      const a = document.createElement("a");
      a.href = href;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = href;
      if (i > 0) linksRow.appendChild(document.createElement("br"));
      linksRow.appendChild(a);
    });
    root.appendChild(linksRow);
  }

  // Give / Get bullets
  appendListSection(root, "What they can give", node.give);
  appendListSection(root, "What they want to get", node.get);

  // Keywords
  if (node.keywords && node.keywords.length) {
    root.appendChild(el("h4", null, "Keywords (self-reported)"));
    const kw = el("div", "keywords");
    node.keywords.forEach((k) => kw.appendChild(el("span", "kw", k)));
    root.appendChild(kw);
  }
  if (node.auto_keywords && node.auto_keywords.length) {
    root.appendChild(el("h4", null, "Topic phrases (auto-extracted)"));
    const kw = el("div", "keywords");
    node.auto_keywords.forEach((k) => kw.appendChild(el("span", "kw", k)));
    root.appendChild(kw);
  }

  // Connection blocks
  appendConnections(root, "give_get", id, node);
  appendConnections(root, "topic", id, node);
  appendConnections(root, "keyword", id, node);
}

function appendListSection(root, title, items) {
  root.appendChild(el("h4", null, title));
  const ul = document.createElement("ul");
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.appendChild(el("em", null, "(none)"));
    ul.appendChild(li);
  } else {
    items.forEach((b) => ul.appendChild(el("li", null, b)));
  }
  root.appendChild(ul);
}

function appendConnections(root, layer, id, node) {
  const cls = `conn-block layer-${layer.replace("_", "-")}`;
  const block = el("div", cls);
  block.appendChild(el("h4", null, `${LAYER_LABELS[layer]} connections`));

  const list = (state.edgesByNode[layer].get(id) || [])
    .slice()
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  if (list.length === 0) {
    const empty = el("div", null, "No connections in this layer.");
    empty.style.color = "var(--muted)";
    empty.style.fontSize = "12px";
    block.appendChild(empty);
  } else {
    list.forEach((e) => {
      const other = state.nodeIndex.get(e.otherId);
      if (!other) return;
      const conn = el("div", "conn");
      conn.dataset.id = other.id;
      const direction = layer === "give_get"
        ? (e.incoming ? `${other.name} → ${node.name}` : `${node.name} → ${other.name}`)
        : other.name;
      conn.appendChild(el("span", "conn-name", direction));
      conn.appendChild(el("span", "conn-score", e.score.toFixed(2)));
      if (e.why) conn.appendChild(el("div", "conn-why", e.why));
      conn.addEventListener("click", () => selectNode(other.id));
      block.appendChild(conn);
    });
  }
  root.appendChild(block);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function handleCanvasResize() {
  if (!simulation) return;
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  if (!width || !height) return;
  svg.attr("viewBox", [0, 0, width, height]);
  simulation.force("center", d3.forceCenter(width / 2, height / 2));
  xScale.range([VIEWPORT_MARGIN, width - VIEWPORT_MARGIN]);
  yScale.range([VIEWPORT_MARGIN, height - VIEWPORT_BOTTOM_MARGIN]);
  simulation.alpha(0.25).restart();
}
window.addEventListener("resize", handleCanvasResize);
// ResizeObserver picks up canvas size changes when side panels collapse
// (grid columns change but window size doesn't).
if (typeof ResizeObserver !== "undefined") {
  const ro = new ResizeObserver(() => handleCanvasResize());
  // Wire up after init() runs and the SVG has dimensions.
  setTimeout(() => ro.observe(svg.node()), 0);
}

init().catch((err) => {
  console.error(err);
  document.getElementById("meta-counts").textContent = "Failed to load graph.json";
});
