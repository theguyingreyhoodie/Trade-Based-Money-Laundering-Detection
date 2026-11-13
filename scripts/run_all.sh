#!/usr/bin/env bash
# Reproduce every result in one go.
#
#   bash scripts/run_all.sh                 full run (all seeds, all experiments)
#   PROFILE=quick bash scripts/run_all.sh   small smoke run (a few minutes)
#
# The local Hardhat node is started automatically by the ledger steps when
# Node.js is available; otherwise the pure-Python ledger backend is used.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
ARGS=()
if [[ -n "${PROFILE:-}" ]]; then ARGS+=(--profile "$PROFILE"); fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run() { "$PY" "$@" ${ARGS[@]+"${ARGS[@]}"}; }

step "Phase 1  synthetic trade data";              run scripts/generate_data.py --all-seeds
step "Phase 2  record linkage + risk indicators";  run scripts/build_features.py --all-seeds
step "Phase 3  rules engine + centralised ML";     run scripts/run_baselines.py --all-seeds
step "Phase 5  register trade documents";          run scripts/ledger_register.py
step "Phase 4  primary federated run (hashes on the ledger)"; run scripts/run_fl.py
step "Phase 4  federated experiment sweep";        run scripts/run_fl_experiments.py
step "Phase 6  risk scores + explainable alerts";  run scripts/generate_alerts.py
step "Phase 5  tamper demo + ledger benchmark";    run scripts/ledger_tamper_demo.py
                                                   run scripts/ledger_benchmark.py
step "Stretch  secure agg, DP-SGD, VFL, defence";  run scripts/run_stretch.py
step "Phase 7  evaluation report";                 run scripts/evaluate.py

step "Done. See results/RESULTS.md"
echo "The local chain is left running so the dashboard can verify evidence live; stop it with: make chain-down"
