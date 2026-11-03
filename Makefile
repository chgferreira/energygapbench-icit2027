PYTHON ?= .venv/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := /tmp/energygap-mpl

.PHONY: test smoke pilot plan inspect doctor recovery release
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) -m energygap run --profile smoke --progress log --budget-total-hours .1
pilot:
	$(PYTHON) -m energygap run --profile pilot --progress log --budget-total-hours .1
plan:
	$(PYTHON) -m energygap plan --profile conference --dry-run
inspect:
	$(PYTHON) -m energygap inspect-data --root raw_data
doctor:
	$(PYTHON) -m energygap doctor
recovery:
	$(PYTHON) -m energygap recovery-demo
release:
	$(PYTHON) scripts/package_release.py
