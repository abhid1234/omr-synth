# BRIEF — build "omr-synth": synthetic-data-first OMR for underserved music notation

You are Codex (Sol). **Structure the project architecture and build the scaffold.** Make the design
decisions yourself; this brief gives goal, constraints, and the proven tooling. Work autonomously.

## The goal (chosen from a deep research pass)
Build the foundation for a **specialized model that reads music notation general VLMs are bad at** —
specifically **handwritten composer manuscripts and/or underserved (non-Western) notations.** Printed
sheet-music OMR is already solved (oemer/homr); handwritten manuscripts (e.g. the Ricordi/Verdi
archive) and most non-Western notations are NOT. The winning strategy (from the research) is
**synthetic-data-first**: render notation *forward* (symbolic score → image) so you get unlimited,
perfectly-labeled (image → ground-truth) training pairs for ~$0, then train a model *backward*
(image → symbolic). Data is the moat; the synthetic engine is the core IP. Full research context:
`/Users/abhijitdas/Developer/Workspace/Claude/_research/custom-model/REPORT.md`.

## 🚨 HARD CONSTRAINTS FOR TONIGHT (do not violate)
- **Spend $0. Absolute cap $10.** NO GPU rental, NO paid API calls, NO cloud training, NO large model
  downloads. Everything tonight must run **locally, on CPU, for free.**
- **Do NOT run any training.** WRITE the model + training code (PyTorch), but do not install torch or
  execute training — that's deferred for the user to fund later. Only the **synthetic-data + data-
  pipeline + eval** code should actually RUN tonight (it only needs the already-installed stack).
- Make **as much real progress as possible** within those limits: a working synthetic data engine that
  produces sample pairs, a clean architecture, and a runnable non-GPU proof.

## Proven tooling (already installed in ./.venv — use it; don't reinstall heavy deps)
- **music21** 10.5.0 — procedural score generation + MusicXML export.
- **verovio** — MusicXML/MEI → SVG (works: `tk=verovio.toolkit(); tk.loadData(xml); tk.renderToSVG(1)`).
- **cairosvg** — SVG → PNG. **Requires env var `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`** at
  runtime (set it in the render module via os.environ or document it). Proven working.
- **Pillow / numpy** — augmentation/degradation.
- Run python as `./.venv/bin/python`. Proven end-to-end: music21 → MusicXML → verovio SVG → cairosvg PNG.

## Tonight's deliverables (all free/local)
1. **`DESIGN.md`** — FIRST. Your architecture: the exact notation target + curriculum (start tractable,
   e.g. procedurally-generated monophonic/simple-polyphonic Western scores rendered then *degraded* to
   simulate manuscript/real conditions, with a **swappable renderer** so a non-Western notation can be
   added later); the **ground-truth serialization** the model predicts (e.g. a linearized token
   sequence — **kern / a compact DSL — justify the choice); the **model architecture** (CNN/ViT encoder
   + autoregressive Transformer decoder, OR a small-VLM LoRA fine-tune — pick and justify) and why;
   honest limits.
2. **Synthetic data engine (`src/synth/`)** — procedurally generate valid symbolic scores → render
   (verovio) → rasterize (cairosvg) → **augment/degrade** (rotation, warp, noise, ink bleed, staff-line
   variation, handwriting-like jitter) → emit `(image.png, ground_truth.<fmt>)` pairs + a manifest.
   **This must actually RUN** and produce, say, 20–50 sample pairs in a `samples/` dir as proof.
3. **Ground-truth format + tokenizer (`src/vocab/`)** — the target serialization + encode/decode +
   a vocab; round-trip tested.
4. **Data pipeline (`src/data/`)** — dataset/manifest loader, train/val/test split, curriculum hooks.
5. **Model + training (`src/model/`, `train.py`)** — WRITTEN not run: encoder-decoder def + training
   loop + config (batch, lr, LoRA if used) + a clear "how to run on a rented GPU later" note. Do not
   install torch or execute.
6. **Eval (`src/eval/`)** — the metric (symbol error rate / TEDn-style tree edit distance / token
   accuracy), computable on synthetic pairs now; note the real-manuscript eval set (MUSCIMA++,
   MusiCorpus) for later.
7. **`README.md`** (vision, the gap, the synthetic-first bet, how to generate data, how to train later,
   the $0-tonight boundary), **`requirements.txt`** (pin what's installed; mark torch/training deps as
   "later"), a `.gitignore`, MIT `LICENSE` (Abhi Das), and a `Makefile`/scripts (`make synth`, etc.).
8. **Tests** for the parts that run (synth engine, tokenizer round-trip, eval metric) via pytest or
   stdlib — and they should pass.

## How to work
- Write `DESIGN.md` first, then build. Use `./.venv/bin/python`. Set the DYLD env var for cairosvg.
- Keep the model/training code complete and reviewable but DON'T run it (no torch install, no training).
- Prove the synthetic engine works by generating real sample pairs into `samples/`.
- When done, print a summary: the name you chose, what runs vs what's deferred, sample count generated,
  and the exact command the user runs later to start training (on their own GPU budget).
