# BRIEF 3 — omr-synth: a free, honest browser demo (synthetic-pairs gallery)

You (Codex/Sol) built the omr-synth scaffold + Jianpu renderer + predict.py. Read `DESIGN.md`,
`README.md`, `src/`, and `predict.py` first. Continue the build.

**HARD CONSTRAINTS (unchanged): spend $0 (cap $10). NO torch, NO GPU, NO paid API, NO training run,
NO large downloads, NO model execution.** Use the existing `.venv` (music21, verovio, cairosvg [needs
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`], Pillow, numpy) for the FREE synth path only.

## The task: a self-contained, static browser demo that PROVES the thesis without a trained model
Because inference is torch-gated (no model yet), the demo must NOT claim to run a model. Instead it is
an **honest synthetic-pairs gallery**: it shows real (rendered score image → OMRDSL token target →
both notations) triples straight from the synthetic engine — the exact same-music-two-notations proof.

Build a Python generator `src/demo/build_demo.py` (invoked via a new `make demo` target) that:
1. Generates a small curated set of sample scores (e.g. 8–12), each rendered BOTH ways
   (Western via verovio, Jianpu programmatically) from the SAME symbolic record — reuse existing code.
2. Emits a SINGLE self-contained `demo/index.html` (no external requests — works from `file://`):
   - Every image embedded as a base64 `data:` URI (PNG). No network, no CDN, no external fonts/JS.
   - For each sample: the Western render, the Jianpu render, and the shared OMRDSL token string,
     laid out so the "one engine, two notations, identical token target" point is obvious at a glance.
   - A short honest header: what this is (synthetic ground-truth pairs from the render-forward engine),
     what it is NOT (not a trained-model prediction yet — training is the funded next step).
   - Include the curriculum level and notation tags per sample. Show at least one hard (level-3,
     manuscript-degraded) sample so the augmentation is visible.
   - Light theme, clean, legible, mobile-friendly. Inline CSS only. Tasteful — no emoji section markers.
3. `demo/` output (index.html + any assets) may be committed (it's a small curated artifact, not the
   full regenerable dataset) — keep it lean; base64 inline means index.html is the only real file.
   Regenerable via `make demo`.

## Also
- Keep all existing tests passing; add a test that `build_demo.py` produces a valid self-contained
  html (contains data: URIs, contains OMRDSL tokens, no http(s):// external refs).
- Update `README.md` (a "Demo" section: `make demo` then open `demo/index.html`) and `DESIGN.md`.
- Do NOT run training or install torch. When done, print: number of demo samples, notations shown,
  path to the html, tests status, and confirm $0 (no paid resources).
