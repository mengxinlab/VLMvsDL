# Manuscript directory

Organized by target journal. Shared assets (figures, references, classes) live at
the root; each submission compiles from its own subfolder.

## Active submission
- `radiology_ai/` — **current target: Radiology: Artificial Intelligence** (Original Research).
  - `main_radiology_ai.tex` — blinded manuscript (double-anonymized; no author identifiers).
  - `title_page_radiology_ai.tex` — separate non-blinded full title page (authors, degrees, funding, repo link).
  - `supplementary_information.tex` — Supplementary Notes 1–9, Tables 1–13, Figs 1–6.
  - `cover_letter_radiology_ai.md`.
  - Compile from this folder, e.g. `latexmk -cd -pdf radiology_ai/main_radiology_ai.tex`.
    Figures and refs are referenced as `../figures/` and `../refs`.

## Archived (superseded) submissions
- `archive/cmig/` — original CMIG submission (rejected). `main.tex` + generic cover letter.
- `archive/npj/` — npj Digital Medicine submission (rejected, "single model family too narrow").
  `main_npj.tex` (Springer Nature `sn-jnl`/`sn-nature` class) + `manuscript_latex.pdf` export.
- `archive/zh_check/` — Chinese-language proofreading build.

  Archived `.tex` files reference shared assets by their old root-relative paths and may
  need `../` path fixes (and, for npj, `sn-jnl.cls`) to recompile. The tracked `.pdf`
  in each folder preserves the rendered submission as-is.

## Shared assets (root)
- `figures/` — all manuscript and supplementary figures (PDF).
- `refs.bib` — shared bibliography.
- `sn-jnl.cls`, `sn-nature.bst`, `bst/` — Springer Nature class/styles (used by `archive/npj/`).
- `REVISION_NOTES.md`, `highlights.txt` — drafting notes.
- `submission_upload/` — regenerated export bundles (git-ignored).
