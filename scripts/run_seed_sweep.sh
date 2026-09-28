#!/usr/bin/env bash
#
# Repeat the two arms whose difference is too small to trust from one run.
#
# The Phase 3 comparison left one claim unresolved: crps (0.0268 twCRPS) and
# twcrps_p90 (0.0264) differ by under 2%, which a single run cannot separate
# from run-to-run variation. Every other conclusion in that table rests on
# effects ten to thirty times larger and needs no repeats.
#
# Seed 0 is already trained -- the untagged runs/finetune_crps and
# runs/finetune_twcrps_p90 -- so this adds seeds 1 and 2 for each.
#
# What the seed changes: head initialisation and batch order. It does NOT
# change the augmentation draw, which is fixed in configs/data/augment.yaml.
# So this measures optimisation variance, not variance over data pipelines --
# the standard thing to report, but say which one you measured.
#
# Usage:
#   caffeinate -i ./scripts/run_seed_sweep.sh
#   caffeinate -i ./scripts/run_seed_sweep.sh --epochs 1 --max-steps 20
#
# Roughly 70 minutes per run, so about 4.5 hours for the default sweep.

set -euo pipefail

SEEDS=(1 2)
P90_THRESHOLDS="data/stats/thresholds_train_p90.json"
LOG_DIR="runs"
SUMMARY="${LOG_DIR}/seed_sweep_summary.txt"

mkdir -p "${LOG_DIR}"

for required in runs/mae_small/best.pt "${P90_THRESHOLDS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing ${required}" >&2
        exit 1
    fi
done

started_all=$(date +%s)
: > "${SUMMARY}"

run_arm() {
    local objective="$1" tag="$2" seed="$3"
    shift 3
    local name="${objective}${tag}"

    echo
    echo "=============================================================="
    echo " ${name}   seed ${seed}   ($(date '+%H:%M:%S'))"
    echo "=============================================================="

    local started
    started=$(date +%s)
    cirrus finetune --objective "${objective}" --tag "${tag}" --seed "${seed}" "$@" \
        2>&1 | tee "${LOG_DIR}/finetune_${name}_console.log"

    printf '%-22s %5d min   %s\n' "${name}" \
        "$(( ($(date +%s) - started) / 60 ))" \
        "$(tail -n 1 "${LOG_DIR}/finetune_${name}/log.csv")" >> "${SUMMARY}"
}

for seed in "${SEEDS[@]}"; do
    # Plain CRPS: the default thresholds file is unused by this objective.
    run_arm crps "_s${seed}" "${seed}" "$@"

    # Tail-weighted at the milder p90 threshold.
    run_arm twcrps "_p90_s${seed}" "${seed}" --thresholds "${P90_THRESHOLDS}" "$@"
done

echo
echo "=============================================================="
echo " sweep done in $(( ($(date +%s) - started_all) / 60 )) min"
echo "=============================================================="
echo "columns: epoch, train_objective, val_objective, val_mae_mm, seconds"
cat "${SUMMARY}"
echo
echo "now run: cirrus compare"
echo "seed 0 is the existing untagged crps and twcrps_p90 runs"
