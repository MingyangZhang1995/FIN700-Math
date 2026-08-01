# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

FIN700 is a **math camp for incoming PhD students in Finance**. This repository holds the lecture notes, organized into three sections:

1. **Linear algebra**
2. **Probability theory**
3. **Basic machine learning**

The audience is PhD-level but the material is foundational (a bootcamp/camp), so notes should build up rigorously from first principles rather than assume prior graduate coursework in these areas. The three sections progress in the order above (linear algebra → probability → ML), since later sections lean on earlier ones.

## Claude's role

Claude is responsible for **writing the lecture notes**, following the user's guidance. The user directs *what* to cover and *how* to frame it; Claude drafts the prose, math, examples, and figures. Follow the user's instructions on scope and emphasis for each topic rather than expanding coverage independently.

## Format and tooling

Notes are written in **Quarto (`.qmd`)** as a **Quarto book** that renders to both **HTML and PDF**. Author content in Quarto markdown: prose in Markdown, math in LaTeX (`$...$` inline, `$$...$$` display), and executable Python cells where numerical examples or figures help.

```bash
uv run quarto preview                # live-reload preview while drafting
uv run quarto render                 # build both HTML and PDF into _output/
uv run quarto render --to pdf        # build a single format
uv run quarto render notes/sections/probability/introduction.qmd  # render one file
```

Use `uv run quarto ...` so Quarto's Python cells execute in the project `.venv`. A Python environment is scaffolded (managed with **`uv`**, Python `>=3.14`) with Jupyter + the scientific stack (`numpy`, `matplotlib`, `scipy`, `ipykernel`) as dev dependencies to back those cells:

```bash
uv sync              # create/update .venv and install dependencies
uv add <package>     # add a dependency (e.g. numpy, matplotlib, scipy)
```

PDF output uses the system TeX Live 2026 with the **LuaLaTeX** engine; HTML math renders via MathJax. A full `uv run quarto render` builds both HTML and the PDF into `_output/` and is confirmed working.

### Dropbox filesystem gotchas (this repo lives in a synced Dropbox folder)

Two issues stem from the working tree being under Dropbox sync; both are already handled, but keep them in mind:

- **`uv` hardlink failures** — set `UV_LINK_MODE=copy` (env var) for any `uv add`/`uv sync`, otherwise uv errors trying to hardlink across the cloud filesystem.
- **Quarto vs. Dropbox file locks** — Dropbox grabs handles on Quarto's intermediate files and makes render cleanup crash. Fixes in place: `execute.freeze` is disabled in `_quarto.yml`, and the regenerable build dirs `.quarto/` and `_output/` are marked Dropbox-ignored via the `com.dropbox.ignored` NTFS stream (`Set-Content -Path .quarto -Stream com.dropbox.ignored -Value 1`). **If either dir is deleted and recreated, re-apply that mark**, or renders will fail with `os error 32` ("file used by another process"). An occasional non-fatal `WARN` about removing a stale `.quarto/_freeze` file is harmless.

## Repository layout

- `_quarto.yml` — book config: chapter/part list, shared cross-referencing and numbering, and the `html`/`pdf` format definitions (including shared LaTeX preamble and Python-execution defaults). New chapters must be registered here to appear in the book.
- `index.qmd` — book preface (unnumbered front matter).
- `notes/sections/{linear-algebra,probability,machine-learning}/` — one directory per part; each holds the part's chapter `.qmd` files, currently a placeholder `introduction.qmd`.
- `notes/notation.qmd` — running notation reference, rendered as an appendix.
- `references.bib` — bibliography; cite with `[@key]`.

### Conventions

- **Cross-references**: label sections/theorems/equations and reference them with `@sec-...`, `@thm-...`, `@eq-...` so numbering stays automatic across HTML and PDF. Numbering is by chapter (e.g. Theorem 2.3).
- **Math macros / LaTeX packages** shared across the notes go in the `pdf.include-in-header` block of `_quarto.yml` (already includes `amsmath`, `amssymb`, `amsthm`, `bm`) — keep them there rather than redefining per file.
- **Notation**: use `\boldsymbol{x}` for vectors and `\boldsymbol{A}` for matrices, matching `notes/notation.qmd`. Prefer `\boldsymbol` over `\bm` (MathJax/HTML doesn't know `\bm`) and over `\mathbf` (which can't bold Greek letters); `\boldsymbol` renders in both HTML and PDF and bolds Greek (e.g. `\boldsymbol{\beta}`).

## Notes

- Quarto 1.9+ and TeX Live 2026 are installed. Quarto may not be on a fresh shell's `PATH`; it lives at `C:\Program Files\Quarto\bin`.
- The default branch is `main`; active work here has been on `master`.
