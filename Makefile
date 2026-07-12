PYTHON = python3
UV = $(PYTHON) -m uv
MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
			 --ignore-missing-imports --disallow-untyped-defs \
			 --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src

debug:
	$(UV) run $(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .venv

lint:
	$(UV) run $(PYTHON) -m flake8 src/*.py
	$(UV) run $(PYTHON) -m mypy src/*.py $(MYPY_FLAGS)

lint-strict:
	$(UV) run $(PYTHON) -m flake8 .
	$(UV) run $(PYTHON) -m mypy . --strict