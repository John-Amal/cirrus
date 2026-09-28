#!/usr/bin/env bash
#
# Train all four objective arms of the Phase 3 comparison, one after another.
#
# Sequential on purpose: run in parallel and the arms contend for the GPU and
# for dataloader workers, which makes the timings meaningless and slows the
# whole set down.
#
# Usage:
#   caffeinate -i ./scripts/run_finetune_arms.sh
#   caffeinate -i ./scripts/run_finetune_arms.sh --epochs 1 --max-steps 20
#
# caffeinate -i stops macOS sleeping while the script runs. Keep the machine
# plugged in: on battery, macOS throttles and the runs take far longer.
#
# Any extra arguments are passed through to every arm, which is how the
# timing run above works.

set -euo pipefail

ARMS=(mse l1 crps twcrps)
LOG_DIR="runs"
SUMMARY="${LOG_DIR}/finetune_summary.txt"

mkdir -p "${LOG_DIR}"

# Fail early rather than three arms in: every arm needs these.
for required in runs/mae_small/best.pt data/stats/thresholds_train.json; do
    if [[ ! -e "${required}" ]]; then
        echo "missing ${required}" >&2
        echo "run 'cirrus pretrain' and 'cirrus thresholds' first" >&2
        exit 1
    fi
done

started_all=$(date +%s)
: > "${SUMMARY}"

for arm in "${ARMS[@]}"; do
    echo
    echo "=============================================================="
    echo " ${arm}   ($(date '+%H:%M:%S'))"
    echo "=============================================================="

    started=$(date +%s)
    # tee keeps the per-step lines, which is where a divergence shows up
    # first; the CSV only gets one row per epoch.
    cirrus finetune --objective "${arm}" "$@" 2>&1 \
        | tee "${LOG_DIR}/finetune_${arm}_console.log"
    elapsed=$(( $(date +%s) - started ))

    # The last CSV row holds the final epoch's validation numbers.
    final=$(tail -n 1 "${LOG_DIR}/finetune_${arm}/log.csv")
    printf '%-8s %5d min   %s\n' "${arm}" "$(( elapsed / 60 ))" "${final}" \
        >> "${SUMMARY}"
done

echo
echo "=============================================================="
echo " all arms done in $(( ($(date +%s) - started_all) / 60 )) min"
echo "=============================================================="
echo "columns: epoch, train_objective, val_objective, val_mae_mm, seconds"
cat "${SUMMARY}"
echo
echo "logs and checkpoints: ${LOG_DIR}/finetune_<arm>/"
