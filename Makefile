PYTHON := ./.venv/bin/python
CAIRO_ENV := DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
SIZE ?= 256

.PHONY: synth demo demo-predictions test check eval train-help predict-help clean-samples

synth:
	$(CAIRO_ENV) $(PYTHON) -m src.synth.generate --output samples --count $(SIZE)

demo:
	$(CAIRO_ENV) $(PYTHON) -m src.demo.build_demo

demo-predictions:
	$(CAIRO_ENV) $(PYTHON) demo_predictions.py --checkpoint "$(CHECKPOINT)"

test:
	$(CAIRO_ENV) $(PYTHON) -m unittest discover -s tests -v

check: test
	$(PYTHON) -m compileall -q src tests

eval:
	$(CAIRO_ENV) $(PYTHON) evaluate.py --checkpoint "$(CHECKPOINT)"

train-help:
	@echo 'Deferred GPU command: python train.py --manifest samples/manifest.jsonl --output checkpoints'

predict-help:
	$(CAIRO_ENV) $(PYTHON) predict.py --help
	$(CAIRO_ENV) $(PYTHON) predict.py --dry-run

clean-samples:
	@echo 'Remove samples manually if you intend to replace the proof set.'
