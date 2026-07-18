# omr-synth

Synthetic-data-first optical music recognition for notation that general vision-language models do
not reliably read: handwritten composer manuscripts first, with interfaces for underserved notation
systems later.

Printed Western OMR has strong tools. Handwritten archives and many non-Western traditions do not
have abundant, consistently labeled image/symbol pairs. `omr-synth` reverses that bottleneck: make a
valid symbolic score, render it forward, degrade it under recorded randomness, and retain perfect
ground truth. The synthetic engine is the product's first data moat.

This scaffold is named **omr-synth**. Its first curriculum generates short mono- and simple
two-voice Western excerpts, renders them through MusicXML and Verovio, rasterizes with CairoSVG, and
adds seeded scan/manuscript-like defects. Targets use compact `OMRDSL-v1`; see [DESIGN.md](DESIGN.md)
for the architecture, format rationale, model choice, and limits.

## What runs locally now

Use the existing virtual environment. CairoSVG on the tested macOS/Homebrew setup needs the fallback
library path; the render module sets it and the Makefile also supplies it explicitly.

```bash
make synth
make test
make check
```

The equivalent generator command is:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ./.venv/bin/python -m src.synth.generate \
  --output samples --count 24 --seed 1729
```

Output is `samples/images/*.png`, matching `samples/targets/*.omrdsl`, and
`samples/manifest.jsonl`. Generation is deterministic by example seed. The framework-neutral loader
supports stable train/validation/test splits and `max_curriculum` filtering. Evaluation provides
token edit distance, symbol error rate, token accuracy, and exact match.

## Deferred GPU training

Tonight's boundary is $0: no Torch install, model download, GPU, paid API, or training run. Model and
training code are reviewable in `src/model/omr.py` and `train.py`, but importing them requires the
optional PyTorch stack. In a separate GPU environment later:

```bash
python -m pip install 'torch>=2.5,<3' 'torchvision>=0.20,<1'
python train.py --manifest samples/manifest.jsonl --output checkpoints --epochs 30 --batch-size 8
```

For meaningful training, generate a much larger manifest and keep real manuscripts held out by
source/composer/page. MUSCIMA++ and appropriately licensed MusiCorpus material are future real-domain
evaluation candidates, not bundled data.

## Layout

```text
src/synth/   score generation, renderer protocol, Verovio renderer, augmentation, CLI
src/vocab/   canonical OMRDSL serializer/validation and fixed tokenizer
src/data/    validated manifest loader, stable splits, curriculum hooks
src/eval/    dependency-free sequence metrics
src/model/   deferred PyTorch encoder-decoder
tests/       standard-library runnable proof tests
```

## Scope warning

Synthetic engraving degradation is not a substitute for real pen strokes or cultural/notation
expertise. The current DSL omits layout, ornaments, beams, lyrics, and many semantics. Results on the
included proof set establish plumbing and reproducibility only, not manuscript generalization.

## License

MIT, copyright Abhi Das. See [LICENSE](LICENSE).
