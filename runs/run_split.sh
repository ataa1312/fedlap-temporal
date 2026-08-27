#!/bin/bash
# In-model repeat/new split matrix: feature vs f+es(spec) vs f+es(persist) vs cn.
# usage: run_split.sh <dataset> <C> [arms...]
#
# The whole body lives in main() so bash PARSES THE FILE COMPLETELY before running
# any of it. Without that, editing this script while a job is executing makes bash
# resume from a stale byte offset in the new file: on 2026-08-27 that re-ran a
# completed loop iteration, logged a spurious `rc=127 n=0` over four finished cells
# and destroyed bitcoin_otc_C1_persist's results.
set -u

main() {
  # repo-relative, so this works from any checkout (cluster home or laptop)
  cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
  local PY=../../.venv/bin/python
  local DS=$1 C=$2
  shift 2
  local ARMS=${@:-"feature spec persist"}
  local OUT=runs/split_matrix
  local H
  H=$(hostname -s)
  mkdir -p "$OUT"

  for arm in $ARMS; do
    local LOG=$OUT/${DS}_C${C}_${arm}.log
    if complete "$LOG"; then
      echo "$(date +%H:%M:%S) SKIP $DS C$C $arm complete" >> $OUT/progress.txt
      continue
    fi
    # Cell lock: the three hosts share one NAS home, so without this two hosts can
    # write the same log. mkdir is the atomic primitive; the owner file is for triage.
    if ! mkdir "$LOG.lock" 2>/dev/null; then
      echo "$(date +%H:%M:%S) SKIP $DS C$C $arm locked-by $(cat "$LOG.lock/owner" 2>/dev/null)" >> $OUT/progress.txt
      continue
    fi
    echo "$H pid=$$" > "$LOG.lock/owner"
    # Re-check under the lock: another host may have finished this cell between the
    # check above and acquiring the lock.
    if complete "$LOG"; then
      rm -rf "$LOG.lock"
      echo "$(date +%H:%M:%S) SKIP $DS C$C $arm completed-by-peer" >> $OUT/progress.txt
      continue
    fi

    local EXTRA
    EXTRA=$(arm_overrides "$arm") || { rm -rf "$LOG.lock"; continue; }

    # Host is recorded per cell: host heterogeneity is a real confound (an arm and its
    # baseline that ran on different hosts produced an uninterpretable cell).
    echo "$H" > "$OUT/${DS}_C${C}_${arm}.host"
    echo "$(date +%H:%M:%S) START $DS C$C $arm on $H" >> $OUT/progress.txt
    nice -n 5 $PY main.py -c config/${DS}_gru.yaml --repeat 3 \
      --set subgraph.num_subgraphs=$C metric.repeat_new_split=true wandb.mode=disabled $EXTRA \
      > "$LOG" 2>&1
    local rc=$?
    rm -rf "$LOG.lock"
    echo "$(date +%H:%M:%S) DONE  $DS C$C $arm rc=$rc n=$(count_results "$LOG") on $H" >> $OUT/progress.txt
  done
  echo "$(date +%H:%M:%S) ALLDONE $DS C$C on $H" >> $OUT/progress.txt
}

count_results() { grep -c '^.*RESULT ' "$1" 2>/dev/null || echo 0; }

complete() { [ -s "$1" ] && [ "$(count_results "$1")" -ge 3 ]; }

arm_overrides() {
  local spectral="spectral.solver=chebyshev spectral.update_mode=update"
  case $1 in
    feature) echo "model.data_type=feature" ;;
    spec)    echo "model.data_type=f+es $spectral spectral.es_features=spec" ;;
    # persist is NOT an informative control on the new subset: it is the split
    # label handed back as a feature, so its new-pair penalty is forced by rank
    # arithmetic (results.md 20.4a). Kept for the repeat subset and aggregate.
    persist) echo "model.data_type=f+es $spectral spectral.es_features=persist" ;;
    # cn: the OTHER baseline 10.11 pre-registered and that was never run. An
    # offline probe puts it above the spectral affinity on both reddit graphs,
    # so `spec` is not attributable to the spectrum until it clears this.
    cn)      echo "model.data_type=f+es $spectral spectral.es_features=cn" ;;
    # The structure placebo. shuffled_fixed permutes the node->row assignment of
    # the REAL basis with one fixed permutation: matched value distribution,
    # matched orthonormality, matched temporal drift, structure severed. Without
    # it nothing separates "the spectrum" from any graded proximity feature of
    # the same shape.
    placebo) echo "model.data_type=f+es $spectral spectral.es_features=spec spectral.basis_source=shuffled_fixed" ;;
    *) return 1 ;;
  esac
}

# `exit` on the same line as the call: bash never reads past this, so even a
# mid-run edit cannot make it resume into replacement text.
main "$@"; exit $?
