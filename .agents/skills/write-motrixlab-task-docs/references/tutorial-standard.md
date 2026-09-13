# MotrixLab Framework Tutorial Documentation Standard

Applies to framework-level tutorial pages under `docs/source/{zh_CN,en}/user_guide/tutorial/` — concepts,
building environments, training, and advanced topics. Task-environment pages follow `writing-standard.md`
instead; the language, validation, and bilingual rules of that standard apply to tutorials too.

## Contents

1. Layered information architecture
2. Page-internal order
3. Diagrams (SVG)
4. Navigation and toctree hygiene
5. Build hygiene specific to tutorials

## 1. Layered information architecture

Write from the macro picture down to details, and let the directory tree express that order:

- Each layer (for example `building_envs/`, `training/`, `advanced/`) has an `index.md` that answers only
  three questions: what problem this layer solves, which steps it contains, and where to go next. Details
  sink into child pages.
- Prefer a small number of layers keyed to the user's journey (concepts → build → train → advanced) over a
  flat list of peer pages.
- A thin topic is a section in its parent index, never a standalone page. Before creating a page, state the
  question it answers; if a short section answers it, do not create the page.
- The macro element of a topic (for example the shared lifecycle) comes before the details that depend on
  it. In a page, the big-picture section precedes config tables and contract details.

One figure, one question: an overview figure draws only the shared skeleton (for example the
`ArrayEnv` step pipeline with framework-owned stages). Workflow-specific annotations — which stages a
`DirectEnv` hook fills, which stages manager kernels drive — belong in that workflow's own page, not in
the shared overview.

## 2. Page-internal order

Use this order by default:

1. a complete minimal working example (or, for concept pages, the macro diagram);
2. concept expansion, section per element;
3. contract details (semantics, invariants, common pitfalls);
4. reference tables and field lists, as lookup material at the end.

Do not open a page with comparison tables, field tables, or contract enumerations before the reader has a
mental model. Keep a short bridge sentence at each transition ("the next section builds the big picture
before the details of …").

Reference material the reader copies or looks up — directory trees, code, per-field tables — stays text.

## 3. Diagrams (SVG)

Draw as SVG: architecture layers, step pipelines, boundary/topology diagrams, workflow comparisons. Keep
as text: repository/runs directory trees, code, field and hook reference tables.

Layout rules:

- Canvas width matches the content column: at most ~800 px, so an embedded figure renders near 1:1 and
  labels stay readable. A six-stage pipeline uses a 2×3 snake layout (three columns per row, second row
  flowing right-to-left) instead of one long row.
- Stage boxes hold only the stage title. Sub-items (term callbacks, hook names) are individual rounded
  chips stacked under their stage, connected by short vertical lines, color-coded by role with a legend.
  Do not pack many small text lines inside one box.
- Route cross-references (for example "a mid-transition request joins the next reset") without crossing
  chips; a dashed annotation chip placed under the target chip is usually clearer than a long connector.
- Figure labels use English technical identifiers; captions and the surrounding prose are bilingual.

Theme and rendering rules:

- Provide a light/dark pair and embed with the `only-light` / `only-dark` classes:

  ````markdown
  ```{image} /_static/images/tutorial/<figure>-light.svg
  :alt: <meaningful alt text>
  :class: only-light
  ```

  ```{image} /_static/images/tutorial/<figure>-dark.svg
  :alt: <meaningful alt text>
  :class: only-dark
  ```
  ````

- Use solid per-theme fills, not `rgba()` translucency — translucent fills composite unreliably across
  SVG renderers. Draw explicit arrowhead triangles instead of `marker-end`. Avoid clipped edge labels.
- Store sources under `docs/source/_static/images/tutorial/`.
- Before embedding, rasterize both variants and actually look at them (dark on a dark background): check
  arrows render, no text is clipped or overlapped, and contrast holds. Never claim an unviewed figure is
  correct.

## 4. Navigation and toctree hygiene

- A toctree `:caption:` and the title of the index page below it must not be identical; the sidebar would
  render the same name twice (caption → page → children).
- An index page must not repeat the sidebar as an in-page text tree. A short layered summary (one line per
  child, what problem it solves) plus path guidance is the replacement.
- Keep toctree entries, page titles, and cross-reference link texts consistent with the actual `#` heading
  of the target page.
- When moving or renaming pages: use `git mv`, then grep the whole `docs/source` tree for old filenames
  and stale relative depths (`../../../configs`-style links change when a page moves a level). Update
  `zh_CN` and `en` toctrees and links in the same change.

## 5. Build hygiene specific to tutorials

`docs/source/conf.py` copies the selected language directory over the build source root with
`dirs_exist_ok=True` and never deletes removed files. Stale copies of deleted or moved pages surface as
"document isn't included in any toctree" warnings that do not come from your change. Clean before building:

```bash
git clean -fdX docs/source
```

Then run the standard strict builds from `writing-standard.md` section 10, and inspect the rendered
sidebar: nesting depth, entry titles, and that the new pages appear where the reading path claims.
