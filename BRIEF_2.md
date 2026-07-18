# BRIEF 2 — omr-synth: add the differentiator (underserved notation) + scale the data

You (Codex/Sol) already built the omr-synth scaffold (synthetic Western OMR, OMRDSL-v1, swappable
renderer interface, 24 samples, 6 passing tests). Read `DESIGN.md`, `README.md`, and `src/` first.
Continue the build. **Same hard constraints as before: spend $0 (cap $10). NO torch, NO GPU, NO paid
API, NO training run, NO large downloads. Only the free local synth/data/eval code RUNS** (uses the
existing `.venv`: music21, verovio, cairosvg [needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`],
Pillow, numpy). Model/train code stays written-but-unrun.

## Task 1 — the DIFFERENTIATOR: a genuinely underserved notation renderer
Western staff OMR is a stepping stone; the *point* of this project is notation general models can't
read. Implement a SECOND renderer behind the existing renderer interface for a genuinely underserved
system. **Recommended: Jianpu (numbered musical notation, 简谱)** — huge real user base across Asia,
poorly served by OMR, and **drawable programmatically without verovio** (digits 1–7 for scale degrees,
dots above/below for octave, underlines for subdivisions, dashes for duration, key/time header). Draw
it directly with Pillow or hand-built SVG→cairosvg from the SAME symbolic event record the Western
path uses. (If you judge another underserved notation better — cipher/sargam/numbered — justify and
pick; but Jianpu is the strong default.)
- Reuse the symbolic score generator; add a Jianpu renderer + a matching ground-truth serialization
  (extend OMRDSL or a small variant — keep it mechanically convertible).
- Generate a batch of Jianpu (image, target) sample pairs into `samples/` alongside the Western ones,
  tagged by notation in the manifest. This concretely proves the "swappable renderer → underserved
  notations" thesis — the project's whole reason to exist.

## Task 2 — scale + harden the synthetic data
- Broaden the score generator: more keys/time-sigs/rhythms/voices, longer excerpts, more variety.
- Add a harder **curriculum tier** toward manuscript/real-world conditions: stronger geometric warp,
  paper texture, ink-bleed/variable stroke, staff-line waviness/breaks, handwriting-style jitter —
  parameterized by curriculum level, seeded/deterministic.
- Bump the default generated set to a few hundred pairs (still fast, CPU, free). Keep it regenerable
  via `make synth` with a size arg.

## Also
- Keep all tests passing; add tests for the new renderer + curriculum (round-trip, manifest tagging).
- Update `DESIGN.md` (the notation module, the Jianpu target) and `README.md`.
- Do NOT run training or install torch. When done, print: notations supported, sample counts per
  notation, tests status, and confirm $0 (no paid resources).
