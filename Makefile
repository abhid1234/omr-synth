PYTHON := ./.venv/bin/python
CAIRO_ENV := DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
SIZE ?= 256

.PHONY: synth test check train-help clean-samples

synth:
	$(CAIRO_ENV) $(PYTHON) -m src.synth.generate --output samples --count $(SIZE)

test:
	$(CAIRO_ENV) $(PYTHON) -m unittest discover -s tests -v

check: test
	$(PYTHON) -m compileall -q src tests

train-help:
	@echo 'Deferred GPU command: python train.py --manifest samples/manifest.jsonl --output checkpoints'

clean-samples:
	@echo 'Remove samples manually if you intend to replace the proof set.'
