#!/bin/bash
# Audit items 12-14 (results.md §20.6 open list), run unattended.
#   12  uci measured INSIDE this matrix — same code, split flag and seeds as the
#       other five datasets. uci is the only dataset with a genuine D_agg/D_new
#       dissociation and it was measured separately, so the whole "reversal"
#       currently compares five fresh datasets against a sixth.
#   13  reddit_title C1 re-run HOST-CLEAN. Its spec arm ran on sim09 while its
#       baselines ran on sim07, aliasing host with treatment.
#   14  reddit_body at 5 seeds. It is one of two surviving datasets and its
#       +0.017 sits near its own floor at n=3.
#
# usage: run_audit_1214.sh <12|13|14>
# Waits until this host has no run of ours in flight, so it never shares a GPU.
# Function-wrapped + exit-guarded: a mid-run edit cannot make bash resume into
# replacement text (see run_split.sh).
set -u

main() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
  local ITEM=$1
  local OUT=runs/split_matrix
  local R=runs/run_split.sh
  local H
  H=$(hostname -s)
  mkdir -p "$OUT"
  local ALL="feature spec persist cn placebo"

  # Gate: never start while another of our jobs holds this GPU.
  local self=$$
  echo "$(date +%H:%M:%S) AUDIT$ITEM gate: waiting for idle on $H" >> $OUT/progress.txt
  while [ "$(ps -u "$(whoami)" -o pid=,args= | grep '[m]ain\.py' | grep -vc "^ *$self ")" -gt 0 ]; do
    sleep 120
  done
  echo "$(date +%H:%M:%S) AUDIT$ITEM start on $H" >> $OUT/progress.txt

  case $ITEM in
    12)
      bash $R uci 1 $ALL
      bash $R uci 9 $ALL
      ;;
    13)
      # Archive the host-split cells rather than delete them: the numbers are
      # real, only their host attribution is broken.
      for a in feature spec persist; do
        [ -f "$OUT/reddit_title_C1_${a}.log" ] && \
          mv "$OUT/reddit_title_C1_${a}.log" "$OUT/reddit_title_C1_${a}.hostsplit.log"
      done
      bash $R reddit_title 1 $ALL
      ;;
    14)
      REPEAT=5 bash $R reddit_body 1 $ALL
      REPEAT=5 bash $R reddit_body 9 $ALL
      ;;
    *)
      echo "$(date +%H:%M:%S) AUDIT$ITEM unknown item on $H" >> $OUT/progress.txt
      exit 1
      ;;
  esac
  echo "$(date +%H:%M:%S) AUDIT$ITEM FINISHED on $H" >> $OUT/progress.txt
}

main "$@"; exit $?
