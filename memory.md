# Session Memory — 2026-05-12

Summary of changes made to the Khoury Researcher Connections project in this session.

## 1. Bug fix: Michelle Borkin's GET section was silently dropped

**Symptom:** After running the pipeline, Michelle Borkin showed 0 "things she can get" in the graph, even though her slide clearly had a GET section.

**Investigation:**
- Read `pipeline/researchers.json` → confirmed `get: []` for her record.
- Inspected slide 26 of the PPTX directly via `python-pptx`. Her GET frame's text was:
  ```
  WHAT YOU CAN GET
  New collaborations: I want to help you make the best visualizations!
  Research expertise in education
  ```
- Traced through `parse_pptx.classify_frame` → it was classifying the frame as `'other'` instead of `'get'`.

**Root cause:** Asymmetry in the header constants in `pipeline/parse_pptx.py`:
```python
GIVE_HEADERS = ("WHAT YOU CAN GIVE", "GIVE:", "GIVE")
GET_HEADERS  = ("WHAT YOU WANT TO GET", "GET:", "GET")
#                ^ no "WHAT YOU CAN GET" variant
```
The GIVE side handled the "CAN" phrasing; the GET side only handled the "WANT TO" phrasing. The PPTX template appears to have evolved over time, and Michelle's slide used the "CAN" phrasing for GET.

**Fix:**
```python
GET_HEADERS = ("WHAT YOU WANT TO GET", "WHAT YOU CAN GET", "GET:", "GET")
```
After re-running `python pipeline/run_all.py`, Michelle correctly parsed with `give=3, get=2`, and she now appears as a target in multiple give→get edges (Brianna Dym, Hazra Imran, Felix Muzny, Mohit Singhal, Jessica Staddon).

## 2. UX change: Layer pills made exclusive (radio-button behavior)

**Before:** Layer pills (`give_get`, `topic`, `keyword`) were toggle buttons backed by a `Set` — multiple could be active simultaneously, producing a combined overlay.

**After:** Clicking a pill exclusively activates that layer and deactivates the others.

**Location:** `public/app.js` in `bindControls()` (~line 391).

**Change:**
```js
// Before — toggle add/delete on the Set
state.active.add(layer) / state.active.delete(layer);

// After — replace the Set with a single-element Set
state.active = new Set([layer]);
for (const p of allPills) {
  const on = p.dataset.layer === layer;
  p.classList.toggle("active", on);
  p.setAttribute("aria-pressed", String(on));
}
```

**Rationale:** Cleaner exploration UX — you focus on one relationship type at a time. Also avoids the empty-state problem (all layers off = blank graph).

## 3. Privacy: PPTX removed from git history

**Concern:** The source PPTX (`Khoury Research Day Give_Get_Slides.pptx`) likely contains private researcher contact info, headshots, and notes that shouldn't live in a public repo.

**Action taken:**
1. Merged the bug fix + UX change PR into `main` locally (without the PPTX).
2. Ran `git filter-repo --path "Khoury Research Day Give_Get_Slides.pptx" --invert-paths --force` to purge the file from all commits in history.
3. Re-added the `origin` remote (filter-repo removes it as a safety measure).
4. Added the PPTX to `.gitignore`.
5. Force-pushed `main` to GitHub.

**Consequence:** All commit SHAs changed. Anyone with a local clone of the repo must delete it and re-clone.

## Final commits on `main`

```
fe489d0 Remove PPTX from tracking (contains private researcher data)
c902751 Merge fix/michelle-borkin-get-exclusive-layers
912da8f Fix Michelle Borkin GET parsing; make layer pills exclusive
```

## Deployment

GitHub Actions (`.github/workflows/pages.yml`) auto-deploys `public/` to GitHub Pages on every push to `main`. The force-push triggered a fresh deploy.

Live URL: <https://johnguerra.co/viz/khouryResearch/>

## Lessons / things to watch

- **Parser symmetry:** When extracting structured sections from semi-templated documents, header constants should be defined symmetrically across all section types. Asymmetries cause silent data loss.
- **Source-data sensitivity:** Treat raw source files (PPTX, spreadsheets with PII) as sensitive by default. Add to `.gitignore` from day one; only commit the *derived, sanitized* outputs (`researchers.json`, `graph.json`).
- **History rewrites:** `git filter-repo` works cleanly but always: (a) confirms the user understands the force-push consequence, (b) removes the `origin` remote — you must re-add it before pushing.
