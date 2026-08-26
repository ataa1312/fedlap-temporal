# MEMORY — FedLap handoff for the next agent

## !! 2026-07-26 — THE RECURRENCE FINDING + the distributed probe matrix

**The science (supersedes the gate-1 "spectral fusion works" reading; results.md §10.10).**
Decision-level fusion re-ranks pairs that ALREADY have an edge in the cumulative graph and
slightly damages pairs that do not. A **trivial 1-bit `exists` feature beats spectral affinity
~3x** on uci (+0.16 vs +0.055 at C1); on the REPEAT subset every dataset gains hugely
(spec +0.10..+0.18, exists +0.30..+0.45) and on the NEW subset every dataset loses (−0.02..−0.04).
The overall gain is therefore just the MIX, set by the dataset's repeat fraction:
uci 0.49 → clear win; bitcoin_{alpha,otc} 0.08 → null; as733 **0.997**; reddit_{body,title}
0.52/0.58. Sparsity is causally sufficient too (thinning uci's basis to bitcoin density kills the
spectral gain: +0.052 → +0.006). KEY METHOD LESSON: the `shuffled_fixed` placebo removes structure
AND history together, so it could never separate them — every spectral claim needs the positive
`exists` baseline and the repeat/new split. Also: snapshots are DISJOINT windows
(`_temporal.py::split_by_calendar`), so the basis (cumulative union) injects history the backbone
only holds in compressed GRU form — that, not "structure", is what the fused arms add.

**The tooling.** `analysis/probes/proto_fusion.py` (args `[dataset] [config] [Cs] [seeds] [K]
[thin]`) now carries the `exists`/`cn` baselines, the repeat/new split, basis thinning, and a
SHARED BASIS CACHE via `$PROTO_BASIS_CACHE` (per-snapshot `.npy`, PID-unique temp + `os.replace`
— a shared temp name races between sibling C-jobs, commit `127fc43` fixes exactly that).
`analysis/probes/precompute_bases.py <ds> <cfg> <K> <offset> <stride>` fills that cache; snapshots
are independent so N hosts take N offsets — STRIDED, not contiguous (solve cost grows with t).
reddit_body's 176 bases: ~3 h single-host → **~16 min across 6 hosts**.

**Cluster (2026-07-26).** Runners `runs/run_matrix.sh <ds> <cfg> <K> <maxjobs> <C...>` and
`runs/run_precompute.sh <offset> <stride>`; output `runs/matrix_v2/` (+ `progress.txt` with
per-job rc + host), cache `runs/basis_cache/`. **sim15 = RTX PRO 5000 Blackwell 48 GB**, verified
with a real sm_120 matmul under the pinned torch 2.9.1+cu128 — the best host by far.
**sim16 is NOT usable yet: its host key CHANGED** (present in known_hosts, different ED25519 key —
looks re-imaged like sim15); needs the user's OK + `ssh-keygen -R` before use. as733 needs **one
job per 16 GB GPU** (peak grows with the cumulative graph; 4-way and even 2-way OOMed sim13) and
runs ~40 min per seed. Launch with `ssh -f` + `setsid nohup` (plain `ssh ... &` hangs the client).
NEVER `pkill -f <pattern>` over ssh — it matches your own session's command line and kills the
connection (cost me one session; use `ps -u ge27yuv -o args= | grep` then kill by PID).

## !! LATEST (2026-07-25, late) — gates 1+2 RUN: score fusion works IN-PROTOCOL on uci, null on bitcoin

Everything below this block still holds; this is what changed today. Full detail: results.md §10.10
(blocks GATE 1 / GATE 2 / BREADTH), tooling `analysis/probes/proto_fusion.py` (committed `388faf2`,
K arg `7b16c27`; takes `[dataset] [config] [Cs] [seeds] [K]`, repo-root resolved from `__file__`).

- **GATE 1 (uci, local, PASSED).** The probe swaps `compute_mrr_from_z` for an RNG-/return-identical
  copy (verified: its model mean == the run's reported `mean_mrr` to 4 dp), so the readout IS the
  reported protocol (1000 negs/source, test split, `mrr_method=max`). Decode-time fusion of
  [model score, spectral affinity] adds **+0.026–0.041 MRR at every C{1,3,7,9}**, placebo
  (`shuffled_fixed`-style row permutation) −0.007..+0.000. Two leakage-free λ recipes agree:
  prequential (past snapshots) and val-edge-fitted (current snapshot's val split) — the latter is
  what an in-model λ should copy. λ GROWS with sharding (+0.76 → +2.55).
  **RETRACTED:** "spectral affinity alone beats the model at C≥7" — false in-protocol (0.069–0.076
  vs model 0.075–0.129); it was an artifact of the 200-negative probe.
- **GATE 2 (bitcoin_otc, sim13).** Conditional AUC ceiling REPLICATES (+0.004/+0.020/+0.048/+0.087
  at C1/3/7/9, cut fractions ≈1−1/C) — mechanism now twice-confirmed. But the in-protocol fusion is
  **null: real ≈ placebo at every C**. Same dataset split as §10.7's input PE.
- **BREADTH + K.** bitcoin_alpha also null; the bitcoin nulls survive K ∈ {50,100,200,300} (spectral
  affinity alone is FLAT in K there, 0.037–0.045), while uci STRENGTHENS with K (affinity alone
  0.070 → 0.091 → 0.104; net gain +0.038 → +0.046; at K=200/C9 the MRR doubles). So K=50 was
  undersized for uci but was never the bitcoin constraint. "Fusion only helps weak backbones" is
  also refuted (bitcoin_alpha's backbone is as weak as uci's at C7/9 and gains nothing).
  Reading: communication graphs yes, rating graphs no. as733 cell RUNNING on sim13
  (`runs/run_gate2d.sh`, 4 jobs by C x 3 seeds, ~1 h/run — harvest
  `runs/gate2b_breadth/as733_K50_C*.log`); reddit not attempted (35k nodes → expensive basis).
- **Gate 3 (edge-score smodel) NOT started** — reframed as a conditional method plus its go/no-go
  diagnostic (measure spectral-affinity-alone MRR first). Design: `decode`-time additive term in a
  `DynamicClassifier` subclass, learnable λ in the smodel, exact basis via the existing
  `_spectral_step`/`set_QD` path, placebo free via `spectral.basis_source`, `pe_dim` re-tuned.
- Uncommitted at handoff: results.md (user reviewing). Cluster runs on sim13 shared with another
  user's MATLAB (leave it alone). Probe runs are ~50 s on bitcoin, ~15 s on uci, ~1 h on as733.

## !! READ FIRST (2026-07-25) — spectral REOPENED: federated ceiling confirmed, conversion gap real (§10.10)

`results.md` **§10** is now the load-bearing result. `spectral.basis_source` (commit `1690824`)
replaces the eigenbasis with a null basis of the SAME shape (`random` = Haar orthonormal,
`shuffled` = real eigenvectors with node rows permuted), server-side in `_substitute_basis`.
Crossed with `update_mode` this gives TWO separable controls:

    STRUCTURE control = keep  (basis frozen -> sources differ ONLY in content)
    STABILITY effect  = recompute (real re-solved = stable if low-churn; random redrawn each step)

Results (5 datasets, gru, C3/7):
- STRUCTURE control (keep) is NULL on EVERY dataset incl as733 — a frozen random basis scores like
  the frozen real one (as733 keep: lap 0.305 ~ random 0.318 ~ shuffled 0.299). Graph structure is
  NOT used.
- The ONE non-null is as733 recompute (real 0.342 >> random 0.300 ~ shuffled 0.297, +0.04, all
  seeds). But (a) proves it's NOT structure -> it's TEMPORAL STABILITY: low-churn as733's re-solved
  real basis barely moves, random/shuffled re-randomize each step. High-churn recompute is null
  again (reddit_body lap 0.324 ~ random 0.326 ~ shuffled 0.335 -- real basis churns too).
- fusion concat vs add: null (+0.001). SignNet reddit_body keep control: null (+0.002/+0.003).

Mechanism: the smodel collapses Q's ~95 effective dims to ~2 before fusion (LEARNED, 4.9->2.3).
Magnitude is NOT the issue (||S||/||z|| = 0.29-0.43, 99% node-varying).

Consequences: keep>update>recompute is a STABILITY ordering, not spectral; the four null prototypes
varied HOW Q is built = structure = necessarily null; SignNet parity = capacity not structure;
**BasisNet predicted null — do NOT build it** (structure unused; stability already maxed by keep /
deterministic solve, which BasisNet doesn't touch). §4.2's gauge mechanism superseded (churn-
dependence right, cause wrong). Any future spectral variant must beat its own `basis_source=random`
control UNDER keep. Bottleneck = smodel capacity (512->128->64 MLP, dropout 0.5, LayerNorm on
512-wide input), not the basis.

CLOSURE (2026-07-22/23, §10.6 + §10.7): the ORACLE PROBES showed the consumed Arnoldi basis was
near-chance even at reconstructing its own graph (AUC 0.53 vs exact eigh 0.965; UCI), so the §10
null was overdetermined — while the EXACT low-50 basis has real, partly AA-complementary signal
(FUT 0.710; 0.674 on AA-blind pairs). The repair experiment `model.data_type=f+pe` (`e525928` +
`bcc43af`: exact sym-Laplacian low-k at the INPUT, sqrt(N)-scaled, active-subgraph solver, judged
vs stability-matched `shuffled_fixed`/`random_fixed` controls) STILL yields no consistent gain:
uci mean +0.008 (5/6 cells, matches the oracle prediction), bitcoin_otc mean −0.011 (real basis
WORSE than its null control federated; null codes beat feature at C7 = capacity effect), as733 ~0.
Spectral structure is NOT a reliable lever for this task/backbone. Do not reopen without new
evidence; any new variant must beat shuffled_fixed under keep. §9 resilience = the paper's
positive result.

RESCOPING + THE PIVOTAL RESULT (2026-07-25; results.md §5 = agenda of record): decoder CLOSED
(concat best); fusion CLOSED; depth CLOSED; Laplace smodel SIDELINED (§10.9). The CONDITIONAL-
INFORMATION PROBE (model's own scores as baseline, decode hook, prequential logistic, uci, 3 seeds)
first measured the exact basis's marginal value at C1 = +0.0083±0.0005 (near-redundant: MP computes
smoothing) — but the FEDERATED EXTENSION (user's objection, confirmed) shows the ceiling GROWS with
sharding: +0.022 / +0.044 / +0.067 AUC at C3/7/9, tracking the measured severed-MP-edge fraction
(0.67/0.86/0.90 ~ 1-1/C, counted per snapshot from the real partitions). Mechanism CONFIRMED: the
global basis carries what the forbidden cross-client message passing would have computed. Realized
embedding-level gains (f+pe §10.7) captured almost none of this -> the federated CONVERSION GAP is
real. The probe's own 2-feature score-level fusion realizes the ceiling (recovers ~40% of sharding's
AUC loss at C9). LIVE DIRECTION: decode-time spectral score fusion (edge-score smodel, FedLap
idiom). NEXT: (1) MRR-style probe readout, (2) bitcoin_otc probe, (3) implementation + placebo-
controlled sweeps. Probe scripts: scratchpad cond_probe{,2,3}.py.
Queued next: §9 x ma replication (headline is gru-only); API compliance; paper. Aborted partial
sweeps (do NOT read as complete): runs/abl_btcalpha, abl2_btcalpha_signnet, pe_reddit. Tooling:
analysis/compile_results.py; runs/master_results.csv. Probe scripts: scratchpad cond_probe{,2}.py.

ALSO: `spectral.deterministic_start` now defaults to **True** (`3291fce`, user instruction) —
every number in results.md §2-§9 was produced with False. All §10 sweeps are COMPLETE (nothing
running). Runners: `runs/run_ablation.sh` (fusion x basis), `runs/run_ablation2.sh` (smodel x basis),
`runs/run_pe.sh` (input-PE).


## !! CLUSTER SSH DISCIPLINE (2026-07-25 — we got BANNED once)

The TUM login node runs **ban2fail**; my automated ssh retry loops after an auth failure got the
user's IP banned (the ban looks like "Connection refused", NOT an outage). Before ANY remote work
read the auto-memory `feedback-ssh-ban2fail` and follow it: no scripted retries on auth failure
(hard stop after 2 consecutive failures); no tight `timeout` around the ssh connect phase (killed
handshakes log preauth strikes); minutes-scale connection spacing, no multi-host fan-out bursts;
server-side waiters over client polling; ControlMaster/ControlPersist preferred; after an unban
the first contact is ONE manual ssh.

Read this whole file before touching code. Then read `../PROCEDURE.md`
(mandatory pre-development procedure added 2026-07-03 — every agent must read
and follow it before development). Deeper history lives in the auto-memory
(`fedlap-roland-migration` et al.) and the plan file
`~/.claude/plans/fedlap-phase3-federated-orchestrator.md`.

## What this repo is

Fork of JavadAliakbari/FedLap being EXTENDED (never "replaced" — the user
co-wrote parts and the fmodel/smodel composition idiom is non-negotiable) from
static federated node classification to **federated temporal spectral link
prediction with a ROLAND backbone**. This is THE paper contribution. Branch:
`roland-dev` (`dev` = pre-ROLAND DySAT baseline, `main` = upstream).

## How to run

- Interpreter: `../../.venv/bin/python` from this directory (no python on PATH).
- Tests: `../../.venv/bin/python -m pytest tests/ -q` — currently 100 pass / 1 skip.
- CONFIGS (restructured `0af4621`): **12 base configs `config/<dataset>_{gru,ma}.yaml`**
  (uci, bitcoin_{alpha,otc}, as733, reddit_{body,title} × gru|ma) — the old
  `<name>_<temporal>_<update>` matrix was COLLAPSED (36 files → 12). Each base has
  ALL per-(dataset,temporal) hyperparameters (dims/α/edge_dim/snapshot_freq, spectral
  smodel params surfaced) EXCEPT `model.data_type: null` + `spectral.update_mode: null`
  — the RUN must choose the mode via `--set` (`assert_cfg` enforces it: data_type
  required always; update_mode required only for f+s/structure). feature needs only
  data_type; f+s/structure need both. `config/config_*.yml` = upstream FedLap (kept);
  DySAT `config_Custom*.yml` deleted. Base-config generator in session scratchpad
  (`gen_base_configs.py`), NOT committed.
- Reference run (feature-only parity ~0.113; CPU multi-seed, MPS/threads shift it):
  `../../.venv/bin/python main.py -c config/uci_gru.yaml --repeat 3 --set model.data_type=feature subgraph.num_subgraphs=1 wandb.mode=disabled`
- f+s spectral run: `... -c config/uci_gru.yaml --set model.data_type=f+s spectral.update_mode=update ...` (~0.089).
- `local_finetune` internal-validation early-stops (`2a4b9e5`): `model.local_epochs`
  is a MAX. Faithful = `iterations=1 local_epochs=100` (baked into the base configs).
- Judge multi-seed distributions, never one seed (MPS ±0.01; and a 3-week f+s run
  scored 0.095 on one seed but 0.060–0.095 across four — single-seed lies).

## State at handoff (2026-07-07)

- **Committed through `3725175`.** LATEST SESSION (see work queue #4–#8): Phase 5
  parity (`00dfa4b`); directed/is_directed removal (`bf23045` + centralized
  `0adcb9d6`); config folder merge configs/→config/ (`13c7387`); config_Custom
  deletion (`6f30d2a`); the `<name>_<temporal>_<update>` variant matrix then
  COLLAPSED to 12 base `config/<dataset>_{gru,ma}.yaml` with `model.data_type` +
  `spectral.update_mode` null → forced at runtime via `--set` (`0af4621`,
  assert_cfg enforces); resample-val (`65913aa`); reddit+as733 loaders+configs
  (`271fa16`/`bb9026f`); **per-dataset `base_lr` fix `f33862d`** (bitcoin/reddit
  0.003, as733 0.03 — THIS was the real bitcoin "gap" I'd mis-called noise; bitcoin
  now 0.158 ≈ centralized 0.156); config-parity test + centralized-config snapshot
  fixture (`da310cf`/`3725175`, `tests/data/centralized_configs/`). Suite **112/1**.
  DONE (verified 2026-07-07 at HEAD `3725175`): the 17 shared ROLAND keys are already
  promoted from `EXCLUDED_ROLAND_KEYS` → `APPLICABLE_PARAMS` in `test_config_parity.py`
  (54 applicable = old 37 + 17; none left excluded; `pytest tests/test_config_parity.py`
  12/12 green). No antigravity relay needed. Older history since W10 (`8dc8d82`):
  - `2a4b9e5` — `local_finetune` internal-validation early stop + frozen val
    batch (one frozen `_val_batch` per snapshot, shared by the local AND the
    round-level early stops).
  - `fec65bc` — W7 spectral path: learnable f+s smodel
    (`src/GNN/fed_dynamic_classifier.py`) + server eigen-provider/`_spectral_step`
    + f+s dispatch + zero-init `output_bn` knob.
  - `8e784d6`/`d3263bb`/`4807797` — DySAT/EdgeClassifier cleanup (−2140 lines; see
    work queue #2).
  - `2595858` — Phase 4: `mcc` + `best_threshold` in `metrics/classification.py`
    (on-device torch) + secondary metrics (auc/ap/f1/mcc) wired through the
    federated eval → return payload → LOGGER → wandb (work queue #3).
  - `ddd3169`/`bbd5725` — Gemini's Phase-4 metrics + Phase-5 parity tests
    (`tests/test_phase4_metrics.py`, `tests/test_phase5_changes.py`; suite 100/1).
  - `00dfa4b` — **Phase 5 parity (work queue #4): DONE.** Wired meta-learning
    (W_init moving-avg via `sum_lod`) + LR scheduler into the live-update loop,
    aligned `config/uci.yaml` to centralized `uci_gru.yaml` (8 MP layers,
    `dims_pre_mp=[64]`, `dims_post_mp=[128]`, `scheduler: cos`, `is_meta: true`,
    `iterations: 1`/`local_epochs: 100`), hardened `procrustes_project`
    (`torch.svd`→`torch.linalg.svd` + catch-and-skip on `LinAlgError`).
  - `bf23045` (fedlap) + centralized `0adcb9d6` — dropped the unused
    `dataset.directed`/`is_directed` config key from BOTH repos (ROLAND has NO
    directed config param — it hardcodes `directed=True` as a deepsnap Graph ctor
    arg; the reimpls always build directed edges, nothing branches on the flag).
  Phases 0–2, W1–W10, W7, Phase 4, and Phase 5 are DONE. Working tree clean (only
  this untracked MEMORY.md). NOTE: the parent `codes` repo still shows `M fedlap`
  (gitlink not bumped) + centralized directed-removal is committed there too.
  **Feature parity REACHED:** C=1 3-seed aligned config = 0.113 (0.1055/0.1081/
  0.1262) ≈ centralized `uci_gru` target 0.112±0.008 (was ~0.104 pre-alignment).
  f+s aligned = 0.089 (< feature, gate-3 holds; procrustes fix ran clean).
- **W7 gate-3 is CLOSED as ACCEPTED, not passed** (below). The spectral path
  ships and works; on UCI it does not beat feature-only, and that is documented,
  not a blocker.

## W7 — what was built

Locked design (user decisions, do not relitigate): learnable smodel FedLap-style
(subclasses own an smodel; NO spectral plumbing in DynamicClassifier); smodel
MLP output sized to `gnn.dims[-1]` so `add` and `concat` fusion both work
(default `add`, FedLap-native); server-side global eigendecomp sliced to
clients via `set_QD` (privacy out of scope for now); Laplacian on the
CUMULATIVE edge union up to t; all 3 `spectral.update_mode`s kept +
`spectral.recompute_prob` Bernoulli basis refresh inside `update`; Lanczos vs
exact via `model.smodel_type` prefix; ONE continual W; `federated.sfv_share`
default `local` (per-client W — note: basic FedLap's `joint_train_w` actually
SYNCS SFVs via the just_SFV gradient path, `avg` reproduces that; user chose
local default knowingly).

Implementation map:
- `src/GNN/fed_dynamic_classifier.py`: `DynamicSLaplace` (SpectralLaplaceMixin
  + SClassifier; S = MLP(relu(Q@W)); sfv_share=avg puts W in state_dict),
  `FedDynamicClassifier` (fusion in encode, head widened 2d for concat,
  FedMixin-style protocol + set_QD/intrinsic_regularizer delegation),
  `FedDynamic{Spectral,Lanczos}LaplaceClassifier`, `make_sgraph` (fresh-leaf
  SFV re-wrap per owner), `make_fed_dynamic_classifier`.
- `src/dynamic_server.py`: f+s `initialize` (server builds hop2vec SFV once →
  `share["SFV"]`); spectral provider `get_previous_UD`/`get_spectral_features`
  ported from GNNServer (explicit graph param) + `_spectral_step` (cumulative
  undirected union on CPU, Bernoulli recompute, procrustes to U_0, set_QD
  slices) called at the top of the per-t loop in `joint_train_w`.
- `_step_train_pair` adds `spectral.regularizer_coef × intrinsic_regularizer()`
  (coef default 0).

## W7 — gate-3 ACCEPTED (2026-07-05), not passed

Decision (c): accept that f+s does not beat feature-only on UCI and document it;
the spectral path is correct and committed (`fec65bc`). Do NOT reopen without
new evidence (another dataset, or a SignNet/BasisNet encoder).

**Numbers (C=1, faithful config iterations=1 local_epochs=100, CPU multi-seed):**
feature ~0.104 vs f+s ~0.079 (BN on) — clean multi-seed separation, gap ~0.025.

**Why — three independent fixes ALL plateau ~0.078, below feature:**
1. Amplification: training grows S (‖S‖/‖z‖ 1→~10 over steps) so the spectral
   branch dominates the l2-normed z. Addressed by a zero-init output BatchNorm on
   the smodel (`spectral.output_bn`, gamma=0 so S starts at 0). No help:
   BN-on 0.079 ≈ BN-off 0.077.
2. Basis degeneracy (isolated-node null space, the old root-cause story): the
   scratchpad active-subgraph prototype fixes it → 0.073–0.091, still < feature.
3. Basis density: coarsening to multi-week snapshots (`snapshot_freq=1814400s`)
   densifies the early basis. No help multi-seed (f+s 0.060–0.095, mean ~0.075 =
   1-week) and it CRASHES on some seeds (known bug below).

Converging read: the gap is intrinsic on UCI — the recurrent fmodel already
captures the useful signal; the static spectral stream is largely redundant and
its cross-snapshot basis is unstable (sign/rotation, near-degenerate spectrum).
NOT a wiring bug: spectral params ARE trained (grad flows; W moves ~34% in 3
steps) and ARE optimized+federated (smodel MLP + BN in state_dict → averaged;
SFV W federated only when `federated.sfv_share=avg`, default `local`).

**KNOWN BUG — FIXED (Phase 5, `00dfa4b`):** `Graph.procrustes_project` used
`torch.svd`, which raised `_LinAlgError` on ill-conditioned alignment matrices
(coarse `snapshot_freq`, seed-dependent). Now `torch.linalg.svd(M,
full_matrices=False)` (R = U@Vh, identical to the old R when it succeeds) wrapped
in `try/except torch.linalg.LinAlgError → return from_U` (skip the rotation for
that snapshot rather than crash). Verified: f+s 3-seed aligned config ran clean.

## Work queue

1. Gemini test prompts for W7 (user relays; do NOT author tests yourself):
   FedDynamicClassifier construction/fusion/protocol (incl. `sfv_share` modes +
   the `output_bn`/DynamicSLaplace path), `_spectral_step` modes + `recompute_prob`,
   f+s dispatch, and `local_finetune` early-stop + frozen `_val_batch`.
2. DySAT/EdgeClassifier cleanup — DONE (2026-07-05, `8e784d6`/`d3263bb`/`4807797`,
   −2140 lines). The authorship check changed the approach: GNNServer/GNNClient/
   GNN_classifier/fGNN/DGCN are the user's heavily-edited UPSTREAM classes, so they
   were RESET to upstream `c800401` (kept, not deleted). Surgically stripped the
   user's dead DySAT + EdgeClassifier code from the LIVE files (graph.py rw/context-
   pair cluster + `random_walk` abar branch, classifier.py EdgeClassifier, sGNN.py
   SEdgeClassifier, laplace.py *EdgeLaplace, GNN_models.py dead config_parser import).
   config_parser RESET to upstream (kept + cleaned); eval_utils.py DELETED. Pure-
   upstream periphery (simulations/FedGCN/fedsage/FedPub/src/test/FedLap) KEPT. LIVE
   spared: sGNN/laplace/GNN_models/custom_gat/Lanczos/MLP_model/config_parser. Suite 77/1.
3. Phase 4 — DONE (2026-07-05, `2595858`). `metrics/classification.py` gained
   `mcc` (from the confusion counts, at the fixed threshold) + `best_threshold`
   (F1-argmax over distinct logits, on-device). `best_threshold` is INFORMATIONAL
   — it does NOT re-threshold the other metrics (user chose "fixed threshold" to
   keep accuracy/f1/mcc ROLAND-comparable; NOT FedLap's F1-tuned get_metrics).
   Wiring: `_eval_mrr` (FL path) decodes the global-stitched z in EVAL mode
   (clean BN/dropout, saved/restored so training + MRR RNG are untouched — proven
   bit-identical to the MRR-only baseline under single-thread); `_eval_mrr_local`
   node-count-weight-averages the per-client dict (`_weighted_mean_metrics`,
   nan-aware); `joint_train_w` collects `metrics_history`, logs auc/ap/f1/mcc per
   snapshot, returns `mean_metrics` + `metrics_history`; `main.py` logs them +
   pushes `mean_<k>` to wandb. Verified: suite 77/1; feature C=1 bit-identical to
   baseline, f+s C=1 3-seed 0.0785 ≈ 0.079, feature C=3 0.0615 ≈ 0.062. Gemini
   OWES tests (mcc/best_threshold correctness, FL vs local aggregation, payload
   shape). NOTE: the FL-path MRR itself still decodes in TRAIN mode (pre-existing
   quirk vs the local path's eval-mode) — left untouched to preserve the baseline;
   candidate for a later eval-mode cleanup.
4. Phase 5 — feature parity + cleanup: **DONE (2026-07-05, `00dfa4b` + `bf23045`).**
   - Aligned `config/uci.yaml` to `uci_gru.yaml` (8 MP layers, `dims_pre_mp=[64]`,
     `dims_post_mp=[128]`, `scheduler: cos`, `is_meta: true`, `iterations: 1`/
     `local_epochs: 100`). `dataset.directed` was NOT set true — it was REMOVED
     everywhere instead (unused; see the state block).
   - Meta-learning wired into `joint_train_w` (FL path): per snapshot, warm-start
     `self.load_state_dict(w_init)` before the rounds, then blend
     `w_init = sum_lod([w_init, self.state_dict()], [1-α, α])` after (α=`meta.alpha`
     for moving_average, `1/(t+1)` for online_mean). FL=False path unchanged (meta
     is FL-scoped). LR scheduler wired into `local_finetune` via `_make_scheduler`
     (stepped per local epoch; `none`→no-op).
   - Procrustes SVD hardened (`torch.linalg.svd` + catch-and-skip). KNOWN BUG above
     is now FIXED.
   - config_parser/config.dynamic were already clean (no-op).
   - **RESULT: feature parity reached** (C=1 3-seed = 0.113 ≈ centralized 0.112).
5. **Config folder merge — DONE (2026-07-05, `13c7387`; user's DySAT
   `config_Custom*.yml` deleted + dangling `.env CONFIGPATH` dropped `6f30d2a`).**
   Merged our Registry `configs/` INTO FedLap's original `config/` (present at fork
   `652fba0`) → ONE `config/` package. `git mv` of registry/config/assertions/
   __init__ .py; rewrote all `from configs.` → `from config.` (main/parser/
   registries/datasets/models/utils/tests); dropped `configs/`. FedLap kept intact
   (its upstream `config_*.yml` untouched; only added our python). Suite 100/1.
6. **Spectral-config surfacing + UCI variant matrix — DONE for UCI (2026-07-05,
   UNCOMMITTED).** `config/uci.yaml` split into 6 self-contained
   `uci_{gru,ma}_{keep,update,recompute}.yaml` (gru α=0.9, ma=moving_average α=0.7;
   `<update>`=`spectral.update_mode`); the active spectral smodel params surfaced at
   defaults; `uci_gru_update.yaml` proven config-IDENTICAL to the old `uci.yaml`
   (behavior-neutral); all 6 load+assert+run; suite 100/1. Generator in the session
   scratchpad (NOT committed — files are the deliverable). Decisions (locked): (a)
   surfaced the *active* spectral smodel dims/params
   explicitly in each dataset config (were code defaults: `structure_model.
   num_structural_features=512`, `DGCN_structure_layers_sizes=[128]` smodel MLP,
   `spectral.{regularizer_coef,recompute_prob,output_bn}`, `model.{fusion,
   smodel_type}`, `federated.sfv_share`). (b) Generate `<name>_<temporal>_<update>.
   yaml` variants: `<temporal>`=`gnn.embed_update_method` (gru|ma),
   `<update>`=`spectral.update_mode` (keep|update|recompute) → 6 files/dataset
   (`uci_{gru,ma}_{keep,update,recompute}.yaml`). LOCKED: keep FedLap intact
   (repeat dormant `structure_model` keys, don't refactor); every variant file
   SELF-CONTAINED (no yaml-to-yaml inheritance in this config system); a tiny
   generator is OK to author them. FUTURE (deferred): consolidate the active
   spectral+smodel params into ONE clean `spectral` section (drops the dormant
   static-FedLap keys `gnn_epochs`/`rw_len`/`num_mp_vectors`/`DGCN_layers`…).
7. **Bitcoin variant matrix — DONE (2026-07-05, `aa8e1bb`).** 12 self-contained
   `config/bitcoin_{alpha,otc}_{gru,ma}_{keep,update,recompute}.yaml`, each aligned
   to its centralized counterpart (per-base dims/bn/α differ: alpha_gru 2×64 bn=F
   α=0.8; alpha_ma 6×64 bn=T α=0.6; otc_gru 4×64 bn=F α=0.9; otc_ma 8×128 bn=T
   α=0.9; all `edge_dim: 2`). Loaders `bitcoin_{alpha,otc}` registered + raw data
   already in `fedlap/data/`; loader byte-identical to centralized; 226 alpha /
   262 otc snapshots match. Generator in session scratchpad (not committed).
   **PARITY: at parity (within multi-seed noise).** Bitcoin-Alpha GRU feature C=1
   fedlap 0.146 vs centralized fresh 0.156 (−7%), BUT UCI fedlap 0.113 vs
   centralized fresh 0.097 (+16%) — OPPOSITE directions ⇒ NOT a systematic gap,
   just per-run noise (std ~0.07 UCI / ~0.10 Bitcoin over 27/226 snapshots). LESSON
   (again): judge multi-seed, a single-seed "gap" lies — I over-read a 2-seed
   Bitcoin diff twice before running fresh centralized both datasets.
   Encoders VERIFIED proper en route (user asked): EdgeEncoder/NodeEncoder modules
   byte-identical to centralized; `ModelBinder.encode` threading matches centralized
   `recurrent.py::encode` (edge_encoder once → node_encoder → pre_mp → recurrent);
   config faithful (edge_encoder/dim/bn, node_encoder off).
   Also `65913aa` (train): `local_finetune` now RESAMPLES val negatives each epoch
   (ROLAND parity; the W7 frozen `_val_batch` is now used ONLY by the round-level
   `val_loss`). Neutral for parity (UCI 0.113→0.113, Bitcoin 0.144→0.146); kept
   because it matches centralized's val handling.
8. **Reddit + AS-733 — DONE (2026-07-05, `271fa16` loaders + `bb9026f` configs).**
   Ported `src/datasets/{as733,reddit}.py` from centralized (verbatim + import
   tweaks: `configs.registry`→`config.registry`, `datasets._temporal`→
   `src.datasets._temporal`), registered in `src/datasets/__init__.py`. All 7
   datasets now register (uci/college_msg, bitcoin_{alpha,otc}, as733,
   reddit_{body,title}). Raw data copied to `fedlap/data/` (gitignored): as733-raw
   (114M, 733 daily `asYYYYMMDD.txt`), reddit-raw (794M: body/title tsv + 300-D
   subreddit embeddings). Loaders verified: as733 = 733 snaps / 7716 nodes /
   edge_dim 1 / all-ones x; reddit_body = 177 snaps / 35776 nodes / edge_dim 88 /
   x=300-D embeddings (reddit is the FIRST dataset with real node features +
   node_encoder still false — raw 300-D fed into pre-MP). 18 self-contained configs
   (as733×6, reddit_body×6, reddit_title×6; per-base dims/α/edge_dim/snapshot_freq
   from centralized — as733 freq=D, reddit freq=W, reddit edge_dim=88). Verified:
   suite 100/1; reddit_body feature C=1 (local_epochs=5, short) mean_mrr=0.312
   (auc 0.966; centralized full ~0.38 — right ballpark); as733 processes snapshots
   fine (per-t mrr ~0.34), just slow over 733 snaps. Generator in scratchpad.
   NOTE: reddit f+s (spectral) will be SLOW/heavy — eigendecomp on 35k/54k-node
   graphs × 177 snaps; feature-only is fine, full f+s parity sweeps need care.
8b. **Config-parity test — DONE (`da310cf` fixture + `3725175` test; base_lr fix
   `f33862d`).** `tests/test_config_parity.py` (Gemini) compares each of the 12 base
   configs (overlaid on defaults) to a committed SNAPSHOT of the centralized configs
   (`tests/data/centralized_configs/`, so the two codebases stay separate). Checks
   37 applicable ROLAND params + a drift-guard (every shared key must be applicable
   OR in `EXCLUDED_ROLAND_KEYS`). It CAUGHT the base_lr bug. UPDATE (2026-07-07,
   verified): the 17 still-excluded shared ROLAND keys are now PROMOTED to APPLICABLE
   (54 applicable total; nothing left to relay). 12/12 pass.
9. **NEXT / AGENDA (surfaced 2026-07-07 — user to pick):**
   (a) **THE PAPER CONTRIBUTION — federated (C>1) spectral experiments**: multi-client
       node-sharded runs, MRR degradation vs C=1, whether SFV sharing (`federated.
       sfv_share=avg`) helps. Everything so far is C=1 = centralized-equivalent
       BASELINE, not the result. This is the headline.
   (b) **Full multi-seed parity sweeps** across all 12 base configs vs FRESH
       centralized (only uci+bitcoin spot-checked; as733/reddit feature-smoked only).
       The parity test now guards configs, so any residual gap = code, not config.
   (c) **Spectral-section consolidation** (deferred config refactor — fold active
       spectral+smodel params into ONE clean `spectral` section, drop dormant
       `structure_model` keys). Also: force `model.smodel_type` null too? (user floated).
   My recommended order: (a) headline, or (b) finish-the-baseline-first.
10. Deferred research: structure-only smodel subclass (`data_type=structure`);
   SignNet/BasisNet smodel (Lim et al.) replacing sign-fix+procrustes; FedLap
   API-compliance overhaul ([[fedlap-api-compliance]], supervisor req);
   `_forked_rng` eval-seed reproducibility.

## Conventions (user-enforced)

- Confirm before committing; one-line commit messages; no Co-Authored-By.
- Minimal docstrings, no emojis, no verbose headers.
- Test authoring routes to Gemini/antigravity — you write prompts, not tests.
- `roland/` (in the parent repo) is read-only upstream reference.
- FedLap idioms are sacred: extend via subclass + smodel, never fold into one
  class; never call FedLap "old code".
- Config is the global dict-access singleton (`from src import config`);
  YAML overlays at runtime — never bind config values at import time.
