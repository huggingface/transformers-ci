.PHONY: format

RUFF ?= ruff

format:
	$(RUFF) format src tests
