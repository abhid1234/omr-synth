# omr-synth design

## Product boundary

`omr-synth` is a synthetic-data-first foundation for optical music recognition (OMR) where labeled
real images are scarce. The first target is deliberately tractable: one-page, one- or two-voice
Western staff excerpts containing notes, rests, barlines, ties, key signatures, time signatures, and
simple dynamics. Symbolic scores are generated first, then rendered and degraded toward imperfect
scans and handwritten-composer conditions. This is a curriculum, not a claim that synthetic Western
engraving is equivalent to handwriting.

The renderer is an interface. Verovio/MusicXML is the first implementation; a future notation module
can provide its own generator, renderer, serializer, and augmentation profile for numbered notation,
sargam, cipher notation, or another underserved system without changing manifest, split, or training
interfaces.

## Pipeline and package boundaries

1. `src/synth/scores.py` samples a deterministic `music21` score and its canonical event record.
2. `src/vocab/dsl.py` serializes that record into the model target.
3. `src/synth/render.py` exports MusicXML, renders SVG through Verovio, and rasterizes through
   CairoSVG. It sets Cairo's macOS fallback library path without overwriting a caller's setting.
4. `src/synth/augment.py` applies seeded paper, geometric, scan, staff-line, ink-bleed, and local
   jitter effects.
5. `src/synth/generate.py` writes atomic image/target pairs and a JSONL manifest containing seed,
   curriculum level, paths, image dimensions, and rendering metadata.
6. `src/data/` validates and loads manifests, makes stable leakage-resistant splits by example ID,
   and filters curriculum levels.
7. `src/model/` and `train.py` define the deferred trainable recognizer. They are not imported by the
   runnable synthesis path.
8. `src/eval/` computes token edit distance, symbol error rate (SER), and exact-match/token accuracy.

## Ground truth: OMRDSL v1

The prediction target is a whitespace-tokenized, linear event DSL rather than raw MusicXML, **kern,
or MEI. XML is verbose and makes sequence learning spend capacity on document structure; **kern is
powerful but brings implicit spine semantics unnecessary for this first curriculum. OMRDSL keeps the
musical content explicit and canonical while remaining mechanically translatable back to richer
formats later.

Every token is vocabulary-atomic. A typical target is:

```text
<bos> VERSION_1 PART_BEGIN CLEF_G2 KEY_0 TIME_4_4 VOICE_1 BAR_1 NOTE_C4 DUR_1 ... BAR_END PART_END <eos>
```

Durations are integer multiples of a sixteenth note (`DUR_1`, `DUR_2`, `DUR_4`, ...). Pitches use
letter, optional accidental (`s`/`b`), and octave. Chords are bracketed by `CHORD_BEGIN` and
`CHORD_END`; voices and measures are explicit. The serializer sorts simultaneous chord pitches and
normalizes metadata, so one score has one target. The initial fixed vocabulary covers the generated
pitch/duration range and reserves `<pad>`, `<bos>`, `<eos>`, and `<unk>`. Decode rejects malformed ID
streams; parser validation catches structural errors.

This format intentionally omits engraving layout, beams, ornaments, lyrics, expressive timing, and
full semantic round-trip to MusicXML. Those become versioned tokens only after the image curriculum
contains them.

## Synthetic curriculum

- Level 0: clean monophonic, common time, quarter/eighth/half durations, no augmentation.
- Level 1: monophonic with rests, key/time variation, mild rotation, paper/noise, and blur.
- Level 2: simple two-voice texture and stronger scan defects, ink spread, staff fading, perspective
  warp, and row-wise handwriting-like displacement.

All randomness comes from a recorded per-example seed. Augmentation never changes the semantic
target. The default proof set mixes levels 0–2, while training can select a maximum level and increase
it over epochs. Synthetic validation/test examples use distinct ID-hash partitions; real manuscript
collections must be held out by source/composer/page, never split by crop.

## Model

The deferred model is a compact vision encoder plus autoregressive Transformer decoder, trained from
scratch on paired crops/pages. The encoder is a convolutional patch stem with 2-D positional
embeddings; the decoder uses causal self-attention and cross-attention over image tokens. This is
preferable to a small-VLM LoRA baseline here because it has no tokenizer mismatch, can be kept small,
is inspectable, and does not require downloading or licensing a pretrained model. The code exposes
dimensions so later experiments can scale from roughly tens of millions of parameters.

Training uses teacher forcing, padding-masked cross entropy, AdamW, gradient clipping, optional AMP,
checkpointing, and deterministic splits; the standalone eval module supplies SER. Beam search is left as an inference
extension; greedy decoding is included for a transparent baseline. Training requires a separate
PyTorch install and a user-provided GPU environment and is never exercised by synthesis/tests.

## Evaluation

Primary metric is token symbol error rate: Levenshtein insertions + deletions + substitutions divided
by reference token count. Exact sequence match and aligned-position token accuracy are secondary.
Report metrics by curriculum level and musical feature, not only globally. Later evaluation must use
licensed, source-held-out real handwriting such as MUSCIMA++ and relevant MusiCorpus subsets, with an
adapter that maps their annotations into OMRDSL or a richer successor. Synthetic scores measure
pipeline correctness and controlled robustness, not real-world generalization.

## Honest limits and risks

Rendered engraving plus degradation does not reproduce a composer's stroke formation, corrections,
overwriting, unusual spacing, or notation conventions. The initial generator has narrow musical and
typographic diversity and image-level distortions can create unrealistic artifacts. OMRDSL v1 loses
layout and some musical semantics. A model trained only on it will have a domain gap. The next data
investments should be renderer/font diversity, learned stroke-level synthesis, real unlabeled image
pretraining, a small carefully licensed real validation set, and source-aware evaluation. Human review
is required before generated samples are treated as representative of any non-Western tradition.
