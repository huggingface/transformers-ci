.PHONY: format test

PYTHON ?= python
RUFF ?= ruff

format:
	$(RUFF) format src tests deploy/scripts

test:
	$(PYTHON) -m pytest tests/
