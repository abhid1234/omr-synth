# omr-synth

Synthetic-data-first optical music recognition for notation that general vision-language models do
not reliably read: Western staff, Jianpu (numbered musical notation, 简谱), and Indian Sargam, with a curriculum from
clean engraving to manuscript-like images.

Printed Western OMR has strong tools. Handwritten archives and many non-Western traditions do not
have abundant, consistently labeled image/symbol pairs. `omr-synth` reverses that bottleneck: make a
valid symbolic score, render it forward, degrade it under recorded randomness, and retain perfect
ground truth. The synthetic engine is the product's first data moat.

The same deterministic symbolic event record drives three swappable renderers. Western excerpts go
through MusicXML, Verovio, and CairoSVG. Jianpu is drawn directly with Pillow as scale-degree digits,
octave dots, subdivision underlines, duration marks, rests, bars, and key/time headers. Sargam is
drawn directly with Pillow as tonic-relative S R G M P D N with komal/tivra and saptak marks. All three produce
compact `OMRDSL-v1` targets with an explicit notation token; see [DESIGN.md](DESIGN.md).

## What runs locally now

Use the existing virtual environment. CairoSVG on the tested macOS/Homebrew setup needs the fallback
library path; the render module sets it and the Makefile also supplies it explicitly.

```bash
make synth
make test
make check
```

`make synth` creates 256 pairs by default, balanced across all three notations (within one). Override the total with,
for example, `make synth SIZE=400`.

The equivalent generator command is:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ./.venv/bin/python -m src.synth.generate \
  --output samples --count 256 --seed 1729 --notations western jianpu sargam
```

Output is `samples/images/*.png`, matching `samples/targets/*.omrdsl`, and
`samples/manifest.jsonl`. Generation is deterministic by example seed. Every manifest row includes
notation, renderer, and curriculum metadata. The framework-neutral loader supports stable
train/validation/test splits and `max_curriculum` filtering. Evaluation provides
token edit distance, symbol error rate, token accuracy, and exact match.

## Demo

Build the honest synthetic-pairs gallery and open the resulting local file in any browser:

```bash
make demo
open demo/index.html
```

The single self-contained HTML file embeds ten curated Western/Jianpu/Sargam image trios, their shared
semantic OMRDSL targets, and curriculum metadata. It makes no network requests and does not run or
claim to run a trained model; it is a render-forward ground-truth proof, including visibly degraded
level-3 samples.

## Deferred training and inference

Tonight's boundary is $0: no Torch install, model download, GPU, paid API, training run, or inference
run. Model and training code are reviewable in `src/model/omr.py` and `train.py`. `predict.py` adds
checkpoint image inference with greedy OMRDSL decoding and an optional MusicXML/Verovio visual
round-trip. Torch is imported only when real inference starts, so help, dry-run, preprocessing, and
symbolic conversion remain usable in the existing environment:

```bash
make predict-help
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ./.venv/bin/python predict.py --dry-run
```

In a separate Torch environment later:

```bash
python -m pip install 'torch>=2.5,<3' 'torchvision>=0.20,<1'
python train.py --manifest samples/manifest.jsonl --output checkpoints --epochs 30 --batch-size 8
python predict.py --checkpoint checkpoints/epoch-030.pt --image page.png
python predict.py --checkpoint checkpoints/epoch-030.pt --image page.png \
  --musicxml prediction.musicxml --render prediction.png
```

The checkpoint vocabulary is used for decoding, and inference applies the same grayscale,
aspect-preserving white padding and ink-positive scaling as training. `--musicxml` reconstructs the
semantic OMRDSL subset; `--render` creates a Western staff PNG with Verovio even when the recognized
input notation was Jianpu or Sargam. This is a semantic/visual check, not recovery of original engraving layout.

For meaningful training, generate a much larger manifest and keep real manuscripts held out by
source/composer/page. MUSCIMA++ and appropriately licensed MusiCorpus material are future real-domain
evaluation candidates, not bundled data.

## Layout

```text
src/synth/   score generation, renderer protocol, Western/Jianpu/Sargam renderers, augmentation, CLI
src/vocab/   canonical OMRDSL serializer/validation and fixed tokenizer
src/data/    validated manifest loader, stable splits, curriculum hooks
src/eval/    dependency-free sequence metrics
src/model/   deferred PyTorch encoder-decoder
tests/       standard-library runnable proof tests
```

## Scope warning

Synthetic degradation is not a substitute for real pen strokes or cultural/notation expertise.
Jianpu and Sargam conventions vary by region and repertoire; these are mechanically consistent core
subsets, not complete representations. Results establish plumbing and reproducibility, not generalization.

## License

MIT, copyright Abhi Das. See [LICENSE](LICENSE).
