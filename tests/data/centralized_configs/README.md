# Centralized config snapshots (test fixture)

Point-in-time copies of the centralized ROLAND configs (`codes/configs/<name>.yaml`),
snapshotted 2026-07-05. Used by the config-parity test to check that each fedlap base
config `config/<dataset>_{gru,ma}.yaml` matches its centralized counterpart on the
shared ROLAND hyperparameters.

The two codebases are DELIBERATELY separate — these tests do NOT read the live
centralized repo; they compare against this snapshot. Re-sync when the centralized
configs change (run from `fedlap/`):

    for f in uci_gru uci_ma bitcoin_alpha_gru bitcoin_alpha_ma bitcoin_otc_gru \
             bitcoin_otc_ma as733_gru as733_ma reddit_body_gru reddit_body_ma \
             reddit_title_gru reddit_title_ma; do
      cp ../configs/$f.yaml tests/data/centralized_configs/$f.yaml
    done
