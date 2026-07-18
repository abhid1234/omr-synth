# omr-synth

Synthetic-data-first optical music recognition for notation that general vision-language models do
not reliably read: Western staff and Jianpu (numbered musical notation, 简谱), with a curriculum from
clean engraving to manuscript-like images.

Printed Western OMR has strong tools. Handwritten archives and many non-Western traditions do not
have abundant, consistently labeled image/symbol pairs. `omr-synth` reverses that bottleneck: make a
valid symbolic score, render it forward, degrade it under recorded randomness, and retain perfect
ground truth. The synthetic engine is the product's first data moat.

The same deterministic symbolic event record drives two swappable renderers. Western excerpts go
through MusicXML, Verovio, and CairoSVG. Jianpu is drawn directly with Pillow as scale-degree digits,
octave dots, subdivision underlines, duration marks, rests, bars, and key/time headers. Both produce
compact `OMRDSL-v1` targets with an explicit notation token; see [DESIGN.md](DESIGN.md).

## What runs locally now

Use the existing virtual environment. CairoSVG on the tested macOS/Homebrew setup needs the fallback
library path; the render module sets it and the Makefile also supplies it explicitly.

```bash
make synth
make test
make check
```

`make synth` creates 256 pairs by default, balanced between both notations. Override the total with,
for example, `make synth SIZE=400`.

The equivalent generator command is:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ./.venv/bin/python -m src.synth.generate \
  --output samples --count 256 --seed 1729 --notations western jianpu
```

Output is `samples/images/*.png`, matching `samples/targets/*.omrdsl`, and
`samples/manifest.jsonl`. Generation is deterministic by example seed. Every manifest row includes
notation, renderer, and curriculum metadata. The framework-neutral loader supports stable
train/validation/test splits and `max_curriculum` filtering. Evaluation provides
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
src/synth/   score generation, renderer protocol, Western/Jianpu renderers, augmentation, CLI
src/vocab/   canonical OMRDSL serializer/validation and fixed tokenizer
src/data/    validated manifest loader, stable splits, curriculum hooks
src/eval/    dependency-free sequence metrics
src/model/   deferred PyTorch encoder-decoder
tests/       standard-library runnable proof tests
```

## Scope warning

Synthetic degradation is not a substitute for real pen strokes or cultural/notation expertise.
Jianpu conventions vary by region and repertoire; this is a mechanically consistent core subset,
not a complete representation. Results establish plumbing and reproducibility, not generalization.

## License

MIT, copyright Abhi Das. See [LICENSE](LICENSE).
