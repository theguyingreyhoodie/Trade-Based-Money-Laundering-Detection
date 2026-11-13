# Privacy-preserving TBML detection: one target per pipeline step.
# `make all` reproduces every result; `make quick` does a small smoke run.

PY      ?= .venv/bin/python
PROFILE ?=
ARGS    := $(if $(PROFILE),--profile $(PROFILE),)

.PHONY: help setup chain-install chain-compile chain-up chain-down data features baselines fl \
        fl-experiments ledger ledger-check alerts stretch evaluate test all quick dashboard clean

help:
	@echo "setup           create .venv, install Python deps and the Hardhat project"
	@echo "chain-up/down   start / stop the local Hardhat node (also started automatically)"
	@echo "data            Phase 1  generate synthetic trade data for all seeds"
	@echo "features        Phase 2  link records and build risk indicators"
	@echo "baselines       Phase 3  rules engine + centralised ML"
	@echo "fl              Phase 4  primary federated run"
	@echo "fl-experiments  Phase 4  IID / non-IID / client-count sweep, local-only, FedProx"
	@echo "ledger          Phase 5  register trade documents on the evidence ledger"
	@echo "ledger-check    Phase 5  tamper demo and ledger benchmark"
	@echo "alerts          Phase 6  risk scores and explainable alerts"
	@echo "stretch         secure aggregation, DP-SGD, vertical-FL demo, poisoning defence"
	@echo "evaluate        Phase 7  tables, plots, RESULTS.md"
	@echo "all             everything above, in order (PROFILE=quick for a smoke run)"
	@echo "test            pytest"
	@echo "dashboard       Streamlit investigator dashboard"

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,privacy,dashboard]"
	$(MAKE) chain-install

chain-install:
	cd chain && npm install --no-audit --no-fund && npx hardhat compile

chain-compile:
	cd chain && npx hardhat compile

chain-up:
	$(PY) scripts/chain.py up

chain-down:
	$(PY) scripts/chain.py down

data:
	$(PY) scripts/generate_data.py --all-seeds $(ARGS)

features:
	$(PY) scripts/build_features.py --all-seeds $(ARGS)

baselines:
	$(PY) scripts/run_baselines.py --all-seeds $(ARGS)

fl:
	$(PY) scripts/run_fl.py $(ARGS)

fl-experiments:
	$(PY) scripts/run_fl_experiments.py $(ARGS)

ledger:
	$(PY) scripts/ledger_register.py $(ARGS)

ledger-check:
	$(PY) scripts/ledger_tamper_demo.py $(ARGS)
	$(PY) scripts/ledger_benchmark.py $(ARGS)

alerts:
	$(PY) scripts/generate_alerts.py $(ARGS)

stretch:
	$(PY) scripts/run_stretch.py $(ARGS)

evaluate:
	$(PY) scripts/evaluate.py $(ARGS)

all:
	PROFILE=$(PROFILE) bash scripts/run_all.sh

quick:
	PROFILE=quick bash scripts/run_all.sh

test:
	$(PY) -m pytest

dashboard:
	$(PY) -m streamlit run dashboard/app.py

clean:
	rm -rf data artifacts .pytest_cache .chain.pid .chain.log
