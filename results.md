# FedLap-ROLAND — Federated Temporal Spectral Link Prediction: Results & Analysis

Living record of experimental results and their interpretation, written to be
self-contained for a downstream paper-authoring agent. Last updated 2026-07-29.

> **START HERE — program state (2026-07-29).** Two results carry the paper. (1) §9: federation's
> resilience grows with fragmentation. (2) §10.13–§10.15: a decision-level spectral term over
> rotation-invariant features (`data_type=f+es`), fed by a Chebyshev-filtered solver serving a
> *current* basis, beats the feature-only backbone with null placebos — uci +0.025…+0.047,
> reddit_body +0.053/+0.079, as733 +0.305/+0.374. **§10.15 is the live section**: the first batch
> after the `6ef42a2` leverage-scale fix, and it supersedes §10.13/§10.14's absolute numbers.
> **What §10.15 settles:** the term is NULL on both bitcoin graphs (−0.020…−0.000 vs feature),
> the two datasets with 8% edge recurrence — so the in-model gain is monotone in recurrence with
> no exception, and the honest framing is *a compact encoding of recurrence*, not a structural
> signal a history lookup misses. It also corrects a reporting habit: `real − placebo` overstates
> the effect wherever the placebo damages the model (it does on bitcoin), so **real − feature is
> the column that decides whether the method earns its place**. Still open: the in-model
> repeat/new split (would test the framing directly; touches the eval path, needs sign-off) and
> §10.11's offline NEW-pair probe, the one piece of evidence pointing the other way.
> The paragraph below is the 2026-07-25 state, kept because §10.1–§10.12 are read through it.

> **(2026-07-25).** The paper's positive result is §9 (federation's
> resilience grows with fragmentation). The spectral investigation (§10) has refuted, with controls,
> every implementation-level rescue: fusion style (§10.4), basis quality + injection point (§10.6/7),
> decoder inductive bias and GNN depth (§10.9), temporal-stability confounds (§10.2/8). Scoping
> decisions (user, 2026-07-25): decoder/fusion axes CLOSED (concat established); the Laplace smodel
> SIDELINED (never beat its placebo; sub-baseline under recompute). The decisive result is
> §10.10: the exact basis's CONDITIONAL value given the trained model is +0.007 at C1 (redundant —
> MP computes smoothing) but GROWS ~10x with sharding (+0.022/+0.044/+0.067 at C3/7/9), tracking the
> measured fraction of message-passing edges FL severs (0.67/0.86/0.90). The federated mechanism is
> confirmed, and REPLICATED on bitcoin_otc (+0.004/+0.020/+0.048/+0.087 at C1/3/7/9). Embedding-level
> implementations converted almost none of it; decode-time score fusion does — inside the REAL eval
> protocol it adds **+0.026–0.041 MRR at every C on uci** with null placebos, but is **null on
> bitcoin_otc** (real ≈ placebo), the same dataset split as §10.7's input PE. LIVE DIRECTION: decide
> whether uci or bitcoin_otc is the typical case (cheap breadth pass) BEFORE building the edge-score
> smodel. This supersedes §4.2's gauge mechanism and re-frames §8.

> Status legend: **[confirmed]** = 3-seed means, stable; **[prelim]** = 1–2 seeds or
> small dataset, directionally trusted; **[single]** = one seed, a data point only;
> **[running]** = experiment in flight (numbers partial).

---

## 1. Setup — what is being measured

**The contribution.** We extend FedLap (a federated graph-learning framework with a
learnable *spectral* structure-model) from static node classification to **federated,
temporal, spectral link prediction** with a **ROLAND** recurrent-GNN backbone. The
question: *when a dynamic graph is sharded across C clients (federated, privacy-motivated),
does a shared spectral signal recover the link-prediction accuracy lost to sharding, and
which way of maintaining the spectral basis over time is best?*

**Backbone (fmodel).** ROLAND: per-snapshot GNN whose node embeddings are carried across
snapshots by a GRU (temporal update). Link prediction is "live-update": at each snapshot
`t` we predict `t+1`'s edges, evaluated *before* fine-tuning on `t` (leakage-free).

**Federation.** `C` clients each hold a random node-partition of every snapshot
(intra-client message passing only). Training = FedAvg via a meta `W_init` per snapshot
(clients run local ROLAND fine-tuning; server node-count-weight-averages their weights).
`C=1` is the centralized-equivalent baseline (all prior reproduction work); `C>1` is the
federated regime that this paper is about.

**Spectral structure-model (smodel).** A learned map `S = MLP(relu(Q @ W))`, where
- `Q` = the leading eigenvectors of the (random-walk) graph Laplacian of the **cumulative**
  edge union up to `t` (computed once, server-side, sliced to each client's nodes);
- `W` = a **single, continually-trained** spectral-filter weight (`sfv_share=local` here:
  each client owns its `W`); `spectral_len = 300` eigenpairs.
`S` is fused (`add`) with the fmodel's per-node output before decoding.

**The four conditions (the central axis).**
| condition | spectral basis `Q` over time |
|---|---|
| `feature` | none (plain federated ROLAND; the baseline) |
| `keep` | **frozen** — the `t=0` eigenbasis reused for all snapshots |
| `update` | **tracked** — Rayleigh-Ritz: `H_t = Q_{t-1}ᵀ L_t Q_{t-1}`, `Q_t = Q_{t-1} V_t` |
| `recompute` | **fresh** — a full eigendecomposition (Arnoldi/Lanczos) every snapshot |
`update`/`recompute` optionally align to the `t=0` basis via orthogonal **procrustes**
(`proc-on/off`). `keep` needs no alignment (basis never changes).

**Metrics.** MRR (headline) + AUC / AP / F1 / MCC (secondary, fixed-threshold). Higher = better.

**Datasets.** UCI (college-msg, ~1.9k nodes, weekly), Bitcoin-Alpha/OTC (~weekly),
AS-733 (~7.7k nodes, 733 daily snapshots), Reddit-body/title (~35.8k nodes, weekly, real
300-d node features). Snapshot counts and per-dataset hyperparameters are in
`config/<dataset>_gru.yaml`.

**Non-determinism caveat (load-bearing).** All hardware here is non-deterministic at fixed
seed — Apple MPS and the RTX-4080 CUDA host both vary **±0.008–0.01 MRR** across identical
reruns. **Never read a single seed as exact.** Every headline claim is a 3-seed mean or an
across-C *slope*, not a single number.

---

## 2. Centralized backbone — per-run results (fedlap C=1, feature, gru)

Re-run **2026-07-12 on sim09** as fedlap with `subgraph.num_subgraphs=1` (C=1 = no sharding =
centralized-equivalent) + `model.data_type=feature` (the ROLAND backbone, no spectral). This
**supersedes** the earlier centralized-codebase numbers: fedlap logs **all metrics** (the
centralized codebase logged mostly MRR) and it is the *same* code path as the federated runs
(§3, §6), so C=1 ↔ C>1 is apples-to-apples. All `gru`; seeds 1234/1334/1434; n = snapshots.

| dataset | seed | mean_mrr | std | auc | ap | f1 | mcc | n |
|---|---|---|---|---|---|---|---|---|
| uci | 1234 | 0.11944 | 0.07986 | 0.90269 | 0.90729 | 0.78583 | 0.62554 | 27 |
| uci | 1334 | 0.11926 | 0.08421 | 0.89048 | 0.89712 | 0.78727 | 0.59127 | 27 |
| uci | 1434 | 0.12082 | 0.08522 | 0.89389 | 0.90214 | 0.79418 | 0.64081 | 27 |
| bitcoin_alpha | 1234 | 0.16960 | 0.11919 | 0.94385 | 0.95483 | 0.87395 | 0.77576 | 225 |
| bitcoin_alpha | 1334 | 0.16283 | 0.10339 | 0.94880 | 0.95916 | 0.89130 | 0.79585 | 225 |
| bitcoin_alpha | 1434 | 0.17807 | 0.11170 | 0.95099 | 0.96009 | 0.89320 | 0.80229 | 225 |
| bitcoin_otc | 1234 | 0.20420 | 0.13338 | 0.95312 | 0.96348 | 0.89099 | 0.79003 | 261 |
| bitcoin_otc | 1334 | 0.19858 | 0.11167 | 0.94469 | 0.95750 | 0.88794 | 0.79162 | 261 |
| bitcoin_otc | 1434 | 0.20257 | 0.13703 | 0.94944 | 0.96108 | 0.88601 | 0.79975 | 261 |
| as733 | 1234 | 0.33475 | 0.05144 | 0.97557 | 0.97496 | 0.86877 | 0.76487 | 732 |
| as733 | 1334 | 0.33600 | 0.04858 | 0.97590 | 0.97501 | 0.85364 | 0.73861 | 732 |
| as733 | 1434 | 0.33583 | 0.04892 | 0.97540 | 0.97479 | 0.87789 | 0.76831 | 732 |
| reddit_body | 1234 | 0.39532 | 0.05468 | 0.96931 | 0.97509 | 0.91553 | 0.83725 | 176 |
| reddit_body | 1334 | 0.39263 | 0.05259 | 0.96593 | 0.97280 | 0.91380 | 0.83261 | 176 |
| reddit_body | 1434 | 0.40537 | 0.05027 | 0.96894 | 0.97554 | 0.91595 | 0.83626 | 176 |
| reddit_title | 1234 | 0.41381 | 0.04097 | 0.97726 | 0.98131 | 0.92049 | 0.85111 | 177 |
| reddit_title | 1334 | 0.41109 | 0.04350 | 0.97906 | 0.98237 | 0.92213 | 0.85337 | 177 |
| reddit_title | 1434 | 0.41613 | 0.04302 | 0.97687 | 0.98078 | 0.92232 | 0.84650 | 177 |

**Group 3-seed means (all 6 datasets complete):** uci **0.1198**, bitcoin_alpha **0.1702**,
bitcoin_otc **0.2018**, as733 **0.3355**, reddit_body **0.3978**, reddit_title **0.4137**.
Consistent with the prior centralized-codebase numbers (reddit_body ~0.389, reddit_title ~0.414,
as733 ~0.339, bitcoin_alpha ~0.158, uci ~0.108) within multi-seed noise + the minor code-path
difference. Config parity guarded by `tests/test_config_parity.py`.

---

## 3. Federated spectral experiments — the contribution

### 3.1 UCI — complete per-run results, C{3,5,7} ladder (all metrics)  **[gru, sfv_share=local, 27 snaps]**
Clean re-run 2026-07-12 on sim09: {feature,keep,update,recompute} × C{3,5,7} × proc{on,off}(upd/rec)
× seeds 1234/1334/1434 = **54 runs**, one clean labeled sweep (supersedes the earlier mixed 2026-07-09
ladder). The `can_train` edge-guard fix (30b75b1) enabled C=7 — degenerate 1-edge client-snapshots now
abstain instead of crashing the encoder BN. UCI is tiny (1899 nodes), so high-C shards are degenerate
→ absolute MRR is low and noisy; read the trend, not the absolutes.

| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.08193 | 0.05739 | 0.86765 | 0.87617 | 0.72336 | 0.54681 |
| feature | 3 | – | 1334 | 0.08379 | 0.05260 | 0.86181 | 0.86707 | 0.71679 | 0.53063 |
| feature | 3 | – | 1434 | 0.10007 | 0.07997 | 0.85507 | 0.87365 | 0.73958 | 0.57951 |
| keep | 3 | – | 1234 | 0.08529 | 0.05355 | 0.85675 | 0.87147 | 0.69489 | 0.54269 |
| keep | 3 | – | 1334 | 0.08908 | 0.05056 | 0.85212 | 0.86181 | 0.70569 | 0.53006 |
| keep | 3 | – | 1434 | 0.07712 | 0.04493 | 0.85152 | 0.86784 | 0.69380 | 0.54215 |
| update | 3 | off | 1234 | 0.06849 | 0.03707 | 0.85188 | 0.86219 | 0.70041 | 0.54360 |
| update | 3 | off | 1334 | 0.09256 | 0.07269 | 0.85182 | 0.86176 | 0.70345 | 0.54555 |
| update | 3 | off | 1434 | 0.07186 | 0.05957 | 0.83642 | 0.84993 | 0.69247 | 0.52428 |
| update | 3 | on | 1234 | 0.07887 | 0.05390 | 0.86458 | 0.87606 | 0.69985 | 0.56270 |
| update | 3 | on | 1334 | 0.09669 | 0.06204 | 0.84821 | 0.86527 | 0.70995 | 0.53891 |
| update | 3 | on | 1434 | 0.07065 | 0.05402 | 0.85708 | 0.87431 | 0.70718 | 0.55028 |
| recompute | 3 | off | 1234 | 0.04749 | 0.02243 | 0.82536 | 0.83935 | 0.59897 | 0.47868 |
| recompute | 3 | off | 1334 | 0.06555 | 0.05143 | 0.81901 | 0.83076 | 0.59690 | 0.43690 |
| recompute | 3 | off | 1434 | 0.05157 | 0.02447 | 0.81609 | 0.83158 | 0.58303 | 0.45657 |
| recompute | 3 | on | 1234 | 0.05569 | 0.03339 | 0.84441 | 0.85472 | 0.58366 | 0.46613 |
| recompute | 3 | on | 1334 | 0.07250 | 0.06127 | 0.81716 | 0.83768 | 0.59796 | 0.45408 |
| recompute | 3 | on | 1434 | 0.05152 | 0.02840 | 0.81775 | 0.83455 | 0.58836 | 0.46800 |
| feature | 5 | – | 1234 | 0.07943 | 0.04926 | 0.83274 | 0.84613 | 0.56081 | 0.43421 |
| feature | 5 | – | 1334 | 0.09649 | 0.07538 | 0.82590 | 0.84009 | 0.65046 | 0.48161 |
| feature | 5 | – | 1434 | 0.07672 | 0.05226 | 0.80246 | 0.82796 | 0.63113 | 0.45033 |
| keep | 5 | – | 1234 | 0.07005 | 0.03552 | 0.81555 | 0.83405 | 0.57156 | 0.42872 |
| keep | 5 | – | 1334 | 0.06581 | 0.04253 | 0.81789 | 0.84011 | 0.61190 | 0.45706 |
| keep | 5 | – | 1434 | 0.06318 | 0.03869 | 0.79699 | 0.82990 | 0.56367 | 0.42231 |
| update | 5 | off | 1234 | 0.05232 | 0.03063 | 0.81856 | 0.83255 | 0.50722 | 0.39023 |
| update | 5 | off | 1334 | 0.06605 | 0.03222 | 0.81383 | 0.83547 | 0.61372 | 0.46049 |
| update | 5 | off | 1434 | 0.08002 | 0.04727 | 0.81225 | 0.83373 | 0.59462 | 0.46038 |
| update | 5 | on | 1234 | 0.07135 | 0.04120 | 0.81565 | 0.83364 | 0.55233 | 0.43705 |
| update | 5 | on | 1334 | 0.09120 | 0.06130 | 0.82073 | 0.84096 | 0.61600 | 0.45022 |
| update | 5 | on | 1434 | 0.07108 | 0.04303 | 0.81319 | 0.83422 | 0.58591 | 0.44908 |
| recompute | 5 | off | 1234 | 0.05271 | 0.02751 | 0.78533 | 0.80749 | 0.38615 | 0.29712 |
| recompute | 5 | off | 1334 | 0.05160 | 0.04207 | 0.72864 | 0.75813 | 0.37401 | 0.25588 |
| recompute | 5 | off | 1434 | 0.07500 | 0.07178 | 0.79126 | 0.81485 | 0.45783 | 0.37088 |
| recompute | 5 | on | 1234 | 0.05434 | 0.02410 | 0.78261 | 0.80210 | 0.44956 | 0.37045 |
| recompute | 5 | on | 1334 | 0.05113 | 0.03794 | 0.76333 | 0.78586 | 0.41047 | 0.32031 |
| recompute | 5 | on | 1434 | 0.05611 | 0.03033 | 0.79464 | 0.81667 | 0.48538 | 0.39489 |
| feature | 7 | – | 1234 | 0.05924 | 0.03182 | 0.76565 | 0.79891 | 0.47145 | 0.30780 |
| feature | 7 | – | 1334 | 0.08689 | 0.06765 | 0.79355 | 0.81904 | 0.50399 | 0.37155 |
| feature | 7 | – | 1434 | 0.07148 | 0.05215 | 0.81060 | 0.82865 | 0.60139 | 0.45754 |
| keep | 7 | – | 1234 | 0.07234 | 0.03340 | 0.81556 | 0.83819 | 0.49375 | 0.40256 |
| keep | 7 | – | 1334 | 0.06638 | 0.02978 | 0.79144 | 0.82409 | 0.47073 | 0.37663 |
| keep | 7 | – | 1434 | 0.08778 | 0.06801 | 0.78900 | 0.82108 | 0.51389 | 0.39820 |
| update | 7 | off | 1234 | 0.06537 | 0.03405 | 0.79037 | 0.81801 | 0.42333 | 0.35129 |
| update | 7 | off | 1334 | 0.06904 | 0.05372 | 0.77791 | 0.80768 | 0.46568 | 0.37287 |
| update | 7 | off | 1434 | 0.07931 | 0.07528 | 0.79469 | 0.82707 | 0.52593 | 0.38521 |
| update | 7 | on | 1234 | 0.07324 | 0.04658 | 0.77588 | 0.80712 | 0.44177 | 0.35337 |
| update | 7 | on | 1334 | 0.07142 | 0.03661 | 0.78378 | 0.81819 | 0.51666 | 0.41097 |
| update | 7 | on | 1434 | 0.08508 | 0.07426 | 0.77952 | 0.80463 | 0.52258 | 0.40573 |
| recompute | 7 | off | 1234 | 0.05254 | 0.03440 | 0.72402 | 0.75327 | 0.27551 | 0.23023 |
| recompute | 7 | off | 1334 | 0.06404 | 0.04741 | 0.73453 | 0.76758 | 0.37107 | 0.30090 |
| recompute | 7 | off | 1434 | 0.05711 | 0.04177 | 0.72776 | 0.75425 | 0.31256 | 0.26514 |
| recompute | 7 | on | 1234 | 0.05908 | 0.03584 | 0.75885 | 0.78707 | 0.30621 | 0.26612 |
| recompute | 7 | on | 1334 | 0.06284 | 0.04197 | 0.77811 | 0.80551 | 0.37851 | 0.31118 |
| recompute | 7 | on | 1434 | 0.06188 | 0.04557 | 0.75112 | 0.77542 | 0.35371 | 0.28927 |

**3-seed mean MRR:**
| mode | C=3 | C=5 | C=7 |
|---|---|---|---|
| feature | **0.0886** | **0.0842** | 0.0725 |
| keep | 0.0838 | 0.0663 | **0.0755** |
| update (proc-on) | 0.0821 | 0.0779 | 0.0766 |
| update (proc-off) | 0.0776 | 0.0661 | 0.0712 |
| recompute (proc-on) | 0.0599 | 0.0539 | 0.0613 |
| recompute (proc-off) | 0.0549 | 0.0598 | 0.0579 |

**Reads:** feature **degrades with C** (0.089→0.084→0.073); keep holds and **overtakes feature by
C=7** (0.076 vs 0.073) — the "spectral rescues federated" crossover, here at high C; **recompute is
worst at every C** (0.054–0.061), confirming the UCI recompute-worst story; update sits between.
Procrustes-on mildly helps update/recompute. Absolute MRR is low + noisy (tiny UCI, ~271 nodes/client
at C=7) — the robust signals are the slopes (feature↓, keep flat/up) and recompute-worst, not the
exact values. Mechanism §4.2; cross-dataset contrast (as733 recompute-competitive, reddit
feature-stays-best) in §6.


### 3.2 AS-733 — replicates on a 10× larger graph  **[confirmed]**
`gru`, C{3,5,7} × {feature,keep,update,recompute} × 3 seeds COMPLETE (full per-run tables +
3-seed means in §6). Headline: `keep` stays flat (~0.28–0.31) while `feature` declines, so the
**feature-down / keep-flat crossover replicates** on AS-733 (10× UCI) — keep overtakes feature by
C5 and clearly at C7 (0.282 vs 0.268), the key external-validity check for §3.1. Distinctively,
on this LOW-churn daily topology `recompute`-on is competitive-to-best (procrustes-on matters a
lot: recompute C3 on 0.322 vs off 0.299). Full numbers in §6.

### 3.3 Ablations — procrustes and snapshot density (UCI)  **[prelim]**
- **Procrustes on/off is ~neutral:** |Δ MRR| ≤ 0.01, mixed sign, within 3-seed noise. It
  does **not** lift update/recompute toward keep. ⇒ keep's win is *not* a procrustes
  artifact; the cost is the re-gauging of the basis itself, not a missing alignment.
- **Doubled (2-week) snapshots** (`snapshot_freq=1209600s`): no spectral rescue; the
  `recompute`-worst ordering persists; procrustes actually *hurts* recompute at the coarser
  window. Absolute AUC drops (~0.85→0.78) because a 2-week-ahead forecast is a strictly
  harder task — **2-week and 1-week MRR are not directly comparable.**

---

## 4. Explanation & analysis — for the paper

### 4.1 Headline claim: spectral rescues federated, via a *stable* basis
Sharding removes cross-client edges, so the recurrent fmodel loses global structure and
degrades with C (`feature` slope is down). A **globally-computed** eigenbasis is immune to
sharding (it is built server-side on the cumulative graph, then sliced to clients), so it
re-injects the lost global structure — and the mode that does this most *stably* (`keep`)
holds up and overtakes plain federated as C grows. The paper's headline is the **pair of
opposite slopes** (feature↓, keep↑) reproduced on two graphs of very different size, not
the exact crossover C.

### 4.2 Why the "recompute is best" theory is wrong (and recompute is worst)
> **SUPERSEDED BY §10 (2026-07-21).** The `keep > update > recompute` ORDERING below is real and
> reproducible, but the *gauge* explanation for it is not supported. §10's two controls show the
> branch ignores graph *structure* entirely (a frozen random basis scores like the frozen real one)
> and responds only to the *temporal stability* of the injected per-node code. That RE-CASTS the
> churn-dependence below correctly: recompute is worst where the real basis churns, best on
> low-churn as733 where it is stable — a stability effect, not a spectral one. Read §10 first.

**The a-priori theory.** recompute gives the *exact current* eigenbasis every snapshot, so
it carries the most faithful spectral information → it should win.

**The flaw.** It treats the eigenbasis as *data*, but the model consumes it as a
*coordinate system it must learn against*. `W` is a single filter trained across all
snapshots to make `Q @ W` useful; this only converges if the **meaning of `Q`'s columns is
stable over time**.

**Eigenvectors carry gauge freedom.** For a symmetric Laplacian an eigenvector is defined
only up to **sign** (simple eigenvalues) and up to an arbitrary **orthogonal rotation**
within any (near-)degenerate eigenvalue cluster. Graph-Laplacian spectra are heavily
degenerate (large near-zero multiplicity from structure; tightly packed interior). So a
fresh solve returns an **arbitrary representative** of the eigenspace — sign-flipped,
rotated columns — even when the graph barely changed. The eigen*space* is stable; the
returned eigen*vectors* are not.

**Therefore:**
- **recompute = worst (on high-churn graphs):** it re-draws the gauge every snapshot, so `W`
  chases a target whose coordinate system is reshuffled each step and never settles → the
  spectral branch stays noise. *Fresh eigenvalues buy nothing because the eigenvectors they
  arrive with are re-randomized.* **Churn-dependent corollary (confirmed on as733, §6):** on a
  graph that barely changes snapshot-to-snapshot (as733's daily AS topology), a fresh solve
  returns nearly the *same* gauge, so recompute stops being reshuffled and becomes
  competitive-to-best — the failure is specifically re-randomization under *real* change, which
  is exactly why recompute is worst on high-churn UCI/reddit/bitcoin but not on stable as733.
- **keep = best:** frozen `t=0` basis ⇒ one fixed gauge for the whole run ⇒ `W` sees a
  stationary coordinate system, converges, and learns a genuinely useful filter. The basis
  is "stale," but the fmodel already models temporal change; the smodel's job is a *stable*
  structural signal, so **stability beats freshness**.
- **update = in between:** Rayleigh-Ritz *evolves* the previous basis rather than
  re-drawing it, so the gauge drifts smoothly — a slower-moving target than recompute,
  hence `keep ≥ update ≫ recompute`.

**Why procrustes doesn't rescue it** (consistent with §3.3's neutrality): a single global
orthogonal alignment cannot undo *per-eigenvector* sign flips inside degenerate clusters,
and it aligns to a fixed `t=0` reference that drifts from the current graph as `t` grows —
a partial patch on an intrinsic problem.

**The lesson (and the escape hatch).** "Most accurate current spectrum" is not a
well-defined, learnable representation, because graph→eigenvectors is only defined up to a
sign/rotation gauge; a model trained across time needs a **fixed gauge**, not a fresh one.
This is exactly what **SignNet/BasisNet** (sign-/basis-*invariant* spectral encoders) would
factor out — with such an encoder, recompute's genuinely-fresh spectrum could stop being
scrambled and might become best, *recovering the original intuition*. That is the concrete
follow-up if freshness is to pay off; with the current sign-fix+procrustes smodel it cannot.

### 4.3 Communication story (favorable)
`keep` is *also* the cheapest: one global eigendecomposition at `t=0` and ~zero ongoing
server↔client spectral exchange, vs `recompute`'s full decomposition every snapshot. So on
a utility-vs-communication view, **keep is the Pareto point — best accuracy *and* least
communication.** Freshness is both worse and more expensive. (Communication cost is argued
analytically / big-O; no distributed transport is implemented.)

### 4.4 Threats to validity (state honestly in the paper)
- UCI is small (1899 nodes) and noisy; high-C shards are degenerate — read UCI *trends*, not
  absolutes. The federated matrix is now COMPLETE on all 6 datasets (§6).
- Hardware non-determinism ±0.008–0.01 MRR ⇒ single seeds are unreliable; crossover *point*
  is within noise (the *slopes* and *ordering* are the robust claims).
- `sfv_share=avg` (federating `W`) is now run on **all 6 datasets** (§6): **avg≈local universally**
  (mean Δ +0.001, mean |Δ| 0.006), not load-bearing. Full `ma`-temporal coverage remains un-run.
- Coarse-window and 1-week MRR are not directly comparable (different forecast horizons).

---

## 5. Program state & agenda (rewritten 2026-07-25)

### Closed axes (do not reopen without new evidence; pointers to the closing data)
- **Decoder choice** — CLOSED (§10.9). concat (ROLAND's own choice) is best everywhere; dot/cosine
  lose outright, and the dot arm refuted the product-blind-readout hypothesis (real ≈ placebo under
  a decoder that computes spectral affinity natively; uci + bitcoin_otc).
- **Fusion style (add vs concat)** — CLOSED (§10.4). Null end-to-end; the rank collapse is upstream.
- **GNN depth / receptive field** — CLOSED (§10.9). laplacian−placebo shows no depth trend at L=1..8.
- **Laplace smodel** — SIDELINED (user decision). Never beat its own placebo; below the bare
  backbone under recompute. No further Laplace-smodel runs.
- **Basis maintenance policy (keep/update/recompute) as a spectral story** — recast (§10.2): a
  temporal-stability ordering. **Basis quality** — the Arnoldi estimate is numerically empty
  (§10.6); the exact solver fixes the basis but not the outcome (§10.7). **BasisNet** — predicted
  null, not scheduled (§10.5).

### THE LIVE DIRECTION (2026-07-29) — the method works, and it is recurrence
Gate 3 was built (§10.13), swept (§10.14), re-measured after the leverage fix (§10.15) and
attributed (§10.16). Current state, and what is left:

**Established.** `data_type=f+es` — a decode-time scalar over rotation-invariant features of a
Chebyshev-solved, *current* basis — beats the feature-only backbone with null placebos on uci
(+0.025…+0.047), reddit_body (+0.053/+0.079) and as733 (+0.305/+0.374). Both `update` and
`recompute` give the same result; `keep` is the null arm.

**The limit, now measured (§10.15).** It is NULL on both bitcoin graphs (−0.020…+0.002 vs feature),
whose edge recurrence is 0.08. The in-model gain is monotone in recurrence across all five datasets
with no exception, so the defensible claim is *a compact encoding of recurrence*, not structural
information a history lookup misses. A 1-bit `persist` feature still outscores the spectral term
wherever recurrence is high.

**The mechanism, now attributed (§10.16).** The unfiltered affinity `Σ_i û_ui û_vi` carries all of
it; the learnable heat-kernel bank matches but never exceeds it; the leverage term is inert. The
method reduces to one scalar per pair — which is also the only variant that is EXACTLY gauge-
invariant.

**Open, in order.**
1. **as733/reddit_body attribution** — §10.16 is uci-only; the as733 C9 replication is running
   (`runs/abl_parts`, sim07+sim08). Do not generalise the attribution before it lands.
2. **The in-model repeat/new split** — the one experiment that could overturn the recurrence
   framing, because §10.11's offline probe found a large placebo-null NEW-pair gain on as733
   (+0.37…+0.75) that the recurrence story does not predict. Touches the eval path; needs sign-off.
3. **`spectral.pe_dim` is 50 for every f+es run to date** and §5's older note that 50 is undersized
   still stands; the offline K-sweep showed uci strengthening to K=200–300. The positives may be
   understated. (This does NOT rescue bitcoin — the K-sweep already covered K∈{50…300} there.)
4. **reddit_body has not been re-run post-`6ef42a2`**, so any table mixing it with §10.15 rows
   crosses batches. Needed before docs/spectral_method.tex's results table can be refreshed.
5. Provenance gap: run logs do not record the resolved spectral config (solver/pe_dim); verified
   this session only from runner sources + live `ps`. Worth logging at startup.

### Superseded — the 2026-07-25 spectral direction (kept for the §10.1–§10.12 reading path)
The conditional-information probe, extended to C{1,3,7,9} with per-snapshot cut-edge counts:
the exact global basis's marginal value given the trained model is +0.007 at C1 (redundant, as
MP≈smoothing) but rises to **+0.022 / +0.044 / +0.067** at C3/7/9, tracking the measured severed-MP
fraction (0.67/0.86/0.90 ≈ 1−1/C). Mechanism confirmed; realized embedding-level gains (§10.7)
captured almost none of it; the probe's own score-level fusion does. AGENDA, in order:
1. ~~MRR-style readout of the probe~~ **DONE** — +0.072–0.080 probe-MRR at every C, placebo null.
   **Gate 1 also DONE**: inside the REAL eval protocol (1000-multiplier, test split,
   `mrr_method=max`) the reported MRR moves **+0.026–0.041 at every C** with both leakage-free
   weightings and null placebos (§10.10). One claim retracted: spectral affinity alone does NOT
   beat the model in-protocol.
2. ~~bitcoin_otc replication~~ **DONE, and it SPLITS** (§10.10): the conditional ceiling replicates
   (+0.004/+0.020/+0.048/+0.087 at C1/3/7/9, same cut fractions), but the in-protocol fusion is
   **null there — real ≈ placebo at every C**. Same dataset asymmetry as §10.7's input PE, so it
   is the basis CONTENT on bitcoin_otc that is weak, not the injection point.
3. ~~Breadth before build~~ **DONE for the cheap datasets** (§10.10): bitcoin_alpha null too, and
   the null survives K ∈ {50…300} on both bitcoin graphs while uci STRENGTHENS with K (net +0.046).
   The effect is dataset-conditional, and the condition is measurable in advance (spectral affinity
   alone must rank future partners within reach of the model). as733 running; reddit not attempted.
4. **Gate 3 — the edge-score smodel** [decision pending]: a `decode`-time term in the FedLap
   subclass idiom, learnable λ, exact basis served through the existing `_spectral_step`/`set_QD`
   path, placebo arm free via `spectral.basis_source`. Now framed as a conditional method shipped
   with its diagnostic. The val-edge λ fit is the leakage-free recipe an in-model λ must reproduce;
   `spectral.pe_dim` must be re-tuned (50 is undersized even on uci).
Sidelined as before: Laplace smodel, decoder/fusion/depth axes, SignNet x exact (superseded by the
score-fusion direction).

### Recommended non-spectral agenda
- **§9 x `ma` updater replication** — the headline is currently gru-only; ~54 runs would make the
  resilience claim updater-robust. Highest information per GPU-hour available.
- **Partition-scheme ablation for §9** (pending a code check of `subgraph.partitioning` options) —
  preempts the "IID by construction" reviewer objection.
- FedLap API-compliance overhaul (supervisor requirement, code not compute); paper drafting.

### Done register (details in the cited sections)
Federated matrix C{3,5,7} all 6 datasets (§6); sfv_share avg≈local (§6); coarse + high-C (§7);
SignNet build+sweep (§8); local-only floor / resilience (§9); basis-content controls (§10.2);
input-PE repair (§10.7); per-snapshot analysis (§10.8); decoder + depth ablations (§10.9).
`data_type=structure` remains a stub.

### Aborted / partial (2026-07-25 — deliberately stopped, do NOT read as complete)
The bitcoin_alpha + reddit_body gap-fill sweeps were stopped ~1h in when the program was rescoped
away from matrix-filling: `runs/abl_btcalpha` (21 partial results), `runs/abl2_btcalpha_signnet`
(6), `runs/pe_reddit` (0 results, 3 truncated logs), `runs/pe_btcalpha` (never started). Resume
only if six-dataset control uniformity becomes a paper requirement.

---

## 6. Sweep run log — complete per-run results (updated 2026-07-13)

Every completed run, every metric — raw, one row per run. All `gru`, `sfv_share=local` (except the labeled sfv_share=avg block at the end). `proc` = `use_procrustes` (–: n/a for feature/keep). Every federated table below is now **COMPLETE at C{3,5,7} × {feature,keep,update,recompute} × 3 seeds**; the C=1 centralized baseline is §2, the UCI ladder §3.1. 3-seed mean-MRR summaries follow each dataset; interpretation in the cross-dataset nuance + §4.

### AS-733 (732 snapshots/run) — COMPLETE (54/54, C{3,5,7})
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.31334 | 0.05470 | 0.93544 | 0.94283 | 0.81980 | 0.69939 |
| feature | 3 | – | 1334 | 0.32236 | 0.04837 | 0.94305 | 0.94997 | 0.82760 | 0.71136 |
| feature | 3 | – | 1434 | 0.30730 | 0.05672 | 0.94016 | 0.94560 | 0.81433 | 0.69605 |
| keep | 3 | – | 1234 | 0.30207 | 0.04060 | 0.92623 | 0.94145 | 0.82077 | 0.70559 |
| keep | 3 | – | 1334 | 0.31364 | 0.03202 | 0.93072 | 0.94498 | 0.83922 | 0.72791 |
| keep | 3 | – | 1434 | 0.30607 | 0.03437 | 0.92963 | 0.94318 | 0.82888 | 0.71287 |
| update | 3 | off | 1234 | 0.29808 | 0.04302 | 0.92963 | 0.94162 | 0.82266 | 0.70477 |
| update | 3 | off | 1334 | 0.30961 | 0.03775 | 0.93398 | 0.94501 | 0.83106 | 0.71374 |
| update | 3 | off | 1434 | 0.29510 | 0.04127 | 0.93282 | 0.94353 | 0.82821 | 0.70916 |
| update | 3 | on | 1234 | 0.30192 | 0.03626 | 0.92980 | 0.94246 | 0.81760 | 0.69920 |
| update | 3 | on | 1334 | 0.30459 | 0.03675 | 0.93232 | 0.94426 | 0.82850 | 0.71150 |
| update | 3 | on | 1434 | 0.28801 | 0.04691 | 0.93420 | 0.94333 | 0.81232 | 0.69141 |
| recompute | 3 | off | 1234 | 0.29650 | 0.04025 | 0.93396 | 0.94480 | 0.82108 | 0.70559 |
| recompute | 3 | off | 1334 | 0.29643 | 0.04037 | 0.92984 | 0.94387 | 0.83603 | 0.72317 |
| recompute | 3 | off | 1434 | 0.30293 | 0.03780 | 0.93548 | 0.94607 | 0.82685 | 0.71106 |
| recompute | 3 | on | 1234 | 0.31610 | 0.03146 | 0.93573 | 0.94747 | 0.83255 | 0.72086 |
| recompute | 3 | on | 1334 | 0.32922 | 0.02910 | 0.92880 | 0.94516 | 0.83575 | 0.72667 |
| recompute | 3 | on | 1434 | 0.31992 | 0.02789 | 0.93470 | 0.94753 | 0.83857 | 0.72846 |
| feature | 5 | – | 1234 | 0.29053 | 0.06465 | 0.89266 | 0.91415 | 0.74015 | 0.61286 |
| feature | 5 | – | 1334 | 0.28113 | 0.06116 | 0.90084 | 0.91713 | 0.72654 | 0.60353 |
| feature | 5 | – | 1434 | 0.26840 | 0.06973 | 0.89302 | 0.91138 | 0.72332 | 0.60009 |
| keep | 5 | – | 1234 | 0.28566 | 0.04346 | 0.89511 | 0.91816 | 0.72051 | 0.59825 |
| keep | 5 | – | 1334 | 0.27181 | 0.04714 | 0.89665 | 0.91623 | 0.69715 | 0.57583 |
| keep | 5 | – | 1434 | 0.28917 | 0.03936 | 0.90106 | 0.92191 | 0.72868 | 0.60746 |
| update | 5 | off | 1234 | 0.29093 | 0.04918 | 0.90472 | 0.92237 | 0.72959 | 0.60498 |
| update | 5 | off | 1334 | 0.26838 | 0.05269 | 0.89760 | 0.91634 | 0.68385 | 0.56275 |
| update | 5 | off | 1434 | 0.28115 | 0.04794 | 0.90628 | 0.92298 | 0.69749 | 0.57555 |
| update | 5 | on | 1234 | 0.28821 | 0.04329 | 0.89996 | 0.92010 | 0.72495 | 0.60107 |
| update | 5 | on | 1334 | 0.26071 | 0.04993 | 0.89768 | 0.91621 | 0.66950 | 0.54948 |
| update | 5 | on | 1434 | 0.27435 | 0.04844 | 0.91180 | 0.92620 | 0.70374 | 0.58189 |
| recompute | 5 | off | 1234 | 0.28282 | 0.05139 | 0.90474 | 0.92188 | 0.71508 | 0.59252 |
| recompute | 5 | off | 1334 | 0.26791 | 0.05366 | 0.90842 | 0.92245 | 0.66807 | 0.55025 |
| recompute | 5 | off | 1434 | 0.27611 | 0.04405 | 0.90094 | 0.92081 | 0.71459 | 0.59123 |
| recompute | 5 | on | 1234 | 0.29882 | 0.03770 | 0.90356 | 0.92389 | 0.75075 | 0.62892 |
| recompute | 5 | on | 1334 | 0.27600 | 0.04917 | 0.90145 | 0.91889 | 0.70159 | 0.57824 |
| recompute | 5 | on | 1434 | 0.29708 | 0.04073 | 0.90448 | 0.92323 | 0.72929 | 0.60567 |
| feature | 7 | – | 1234 | 0.23500 | 0.09754 | 0.77890 | 0.83752 | 0.62311 | 0.50282 |
| feature | 7 | – | 1334 | 0.29330 | 0.06030 | 0.86980 | 0.89772 | 0.68030 | 0.56118 |
| feature | 7 | – | 1434 | 0.27429 | 0.07070 | 0.87790 | 0.90073 | 0.66438 | 0.54352 |
| keep | 7 | – | 1234 | 0.28352 | 0.04305 | 0.86736 | 0.89905 | 0.67019 | 0.55307 |
| keep | 7 | – | 1334 | 0.28460 | 0.04803 | 0.86310 | 0.89586 | 0.66003 | 0.54393 |
| keep | 7 | – | 1434 | 0.27705 | 0.04796 | 0.87062 | 0.89879 | 0.63633 | 0.51892 |
| update | 7 | off | 1234 | 0.27622 | 0.05239 | 0.86639 | 0.89786 | 0.64564 | 0.52913 |
| update | 7 | off | 1334 | 0.28679 | 0.05317 | 0.86747 | 0.89810 | 0.64278 | 0.52507 |
| update | 7 | off | 1434 | 0.26451 | 0.05519 | 0.87452 | 0.89926 | 0.61387 | 0.50072 |
| update | 7 | on | 1234 | 0.27432 | 0.05220 | 0.87617 | 0.90158 | 0.61867 | 0.50667 |
| update | 7 | on | 1334 | 0.27778 | 0.05420 | 0.86875 | 0.89852 | 0.63033 | 0.51850 |
| update | 7 | on | 1434 | 0.25130 | 0.05605 | 0.86974 | 0.89552 | 0.61820 | 0.50424 |
| recompute | 7 | off | 1234 | 0.27849 | 0.05810 | 0.87344 | 0.90110 | 0.64955 | 0.53456 |
| recompute | 7 | off | 1334 | 0.28964 | 0.05499 | 0.87580 | 0.90321 | 0.66333 | 0.54620 |
| recompute | 7 | off | 1434 | 0.26238 | 0.05475 | 0.87574 | 0.90082 | 0.62199 | 0.50926 |
| recompute | 7 | on | 1234 | 0.28113 | 0.05139 | 0.87488 | 0.90233 | 0.63611 | 0.52174 |
| recompute | 7 | on | 1334 | 0.28642 | 0.04538 | 0.87892 | 0.90518 | 0.64122 | 0.52867 |
| recompute | 7 | on | 1434 | 0.26552 | 0.04905 | 0.87869 | 0.90144 | 0.62367 | 0.50816 |
_3-seed mean MRR:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| feature | 0.314 | 0.280 | 0.268 |
| keep | 0.307 | 0.282 | 0.282 |
| update proc-off | 0.301 | 0.280 | 0.276 |
| update proc-on | 0.298 | 0.274 | 0.268 |
| recompute proc-off | 0.299 | 0.276 | 0.277 |
| recompute proc-on | 0.322 | 0.291 | 0.278 |

### Reddit-body (176 snapshots/run) — COMPLETE (54/54, C{3,5,7})
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.34104 | 0.05108 | 0.96307 | 0.96966 | 0.88957 | 0.80412 |
| feature | 3 | – | 1334 | 0.33627 | 0.04385 | 0.95909 | 0.96659 | 0.88879 | 0.79897 |
| feature | 3 | – | 1434 | 0.35043 | 0.04245 | 0.96284 | 0.96995 | 0.89043 | 0.80242 |
| keep | 3 | – | 1234 | 0.33719 | 0.05310 | 0.95994 | 0.96767 | 0.89063 | 0.79953 |
| keep | 3 | – | 1334 | 0.33522 | 0.04436 | 0.96158 | 0.96884 | 0.89186 | 0.80762 |
| keep | 3 | – | 1434 | 0.33330 | 0.04890 | 0.96318 | 0.97006 | 0.89050 | 0.80741 |
| update | 3 | off | 1234 | 0.33206 | 0.04882 | 0.96015 | 0.96768 | 0.88989 | 0.80060 |
| update | 3 | off | 1334 | 0.34193 | 0.05067 | 0.96159 | 0.96859 | 0.88999 | 0.80417 |
| update | 3 | off | 1434 | 0.33921 | 0.04859 | 0.96193 | 0.96921 | 0.89017 | 0.80684 |
| update | 3 | on | 1234 | 0.32474 | 0.04884 | 0.96135 | 0.96853 | 0.89196 | 0.80251 |
| update | 3 | on | 1334 | 0.33505 | 0.04808 | 0.96178 | 0.96882 | 0.89149 | 0.80570 |
| update | 3 | on | 1434 | 0.33125 | 0.04461 | 0.96248 | 0.96947 | 0.89177 | 0.80872 |
| recompute | 3 | off | 1234 | 0.31278 | 0.04689 | 0.96167 | 0.96825 | 0.88723 | 0.79587 |
| recompute | 3 | off | 1334 | 0.31879 | 0.05573 | 0.96227 | 0.96869 | 0.88579 | 0.79972 |
| recompute | 3 | off | 1434 | 0.32583 | 0.04690 | 0.96424 | 0.97058 | 0.88903 | 0.80543 |
| recompute | 3 | on | 1234 | 0.32264 | 0.05083 | 0.96153 | 0.96840 | 0.88813 | 0.79678 |
| recompute | 3 | on | 1334 | 0.32395 | 0.05397 | 0.96345 | 0.96981 | 0.88743 | 0.80127 |
| recompute | 3 | on | 1434 | 0.32701 | 0.04464 | 0.96262 | 0.96945 | 0.88658 | 0.80181 |
| feature | 5 | – | 1234 | 0.31320 | 0.04548 | 0.95767 | 0.96516 | 0.85005 | 0.75699 |
| feature | 5 | – | 1334 | 0.29809 | 0.04204 | 0.95428 | 0.96200 | 0.85028 | 0.75259 |
| feature | 5 | – | 1434 | 0.30829 | 0.04196 | 0.95642 | 0.96431 | 0.84805 | 0.75071 |
| keep | 5 | – | 1234 | 0.31298 | 0.04548 | 0.95570 | 0.96384 | 0.85351 | 0.75564 |
| keep | 5 | – | 1334 | 0.29409 | 0.04257 | 0.95645 | 0.96437 | 0.85300 | 0.75890 |
| keep | 5 | – | 1434 | 0.29232 | 0.04113 | 0.95583 | 0.96400 | 0.85320 | 0.76063 |
| update | 5 | off | 1234 | 0.30052 | 0.04299 | 0.95404 | 0.96239 | 0.84117 | 0.74047 |
| update | 5 | off | 1334 | 0.30533 | 0.04134 | 0.95679 | 0.96457 | 0.85290 | 0.75929 |
| update | 5 | off | 1434 | 0.29764 | 0.04297 | 0.95626 | 0.96403 | 0.85603 | 0.76506 |
| update | 5 | on | 1234 | 0.30986 | 0.04779 | 0.95593 | 0.96429 | 0.85715 | 0.75889 |
| update | 5 | on | 1334 | 0.29768 | 0.03904 | 0.95649 | 0.96421 | 0.85616 | 0.76168 |
| update | 5 | on | 1434 | 0.29942 | 0.04193 | 0.95636 | 0.96395 | 0.84952 | 0.75713 |
| recompute | 5 | off | 1234 | 0.29531 | 0.05029 | 0.95855 | 0.96585 | 0.84783 | 0.74984 |
| recompute | 5 | off | 1334 | 0.29078 | 0.05288 | 0.95864 | 0.96581 | 0.84914 | 0.75501 |
| recompute | 5 | off | 1434 | 0.28875 | 0.04289 | 0.95639 | 0.96377 | 0.84640 | 0.75362 |
| recompute | 5 | on | 1234 | 0.30049 | 0.04236 | 0.95754 | 0.96466 | 0.84863 | 0.74885 |
| recompute | 5 | on | 1334 | 0.30415 | 0.04691 | 0.95941 | 0.96639 | 0.85336 | 0.75989 |
| recompute | 5 | on | 1434 | 0.28713 | 0.04486 | 0.95702 | 0.96428 | 0.85021 | 0.75635 |
| feature | 7 | – | 1234 | 0.27738 | 0.04646 | 0.94990 | 0.95848 | 0.77097 | 0.67180 |
| feature | 7 | – | 1334 | 0.27186 | 0.03989 | 0.94937 | 0.95851 | 0.78955 | 0.68750 |
| feature | 7 | – | 1434 | 0.28959 | 0.03710 | 0.95199 | 0.96023 | 0.79578 | 0.69392 |
| keep | 7 | – | 1234 | 0.28252 | 0.04060 | 0.95032 | 0.95945 | 0.79362 | 0.68959 |
| keep | 7 | – | 1334 | 0.28149 | 0.03976 | 0.95413 | 0.96215 | 0.80072 | 0.70272 |
| keep | 7 | – | 1434 | 0.27597 | 0.04003 | 0.95153 | 0.96021 | 0.78810 | 0.69054 |
| update | 7 | off | 1234 | 0.28296 | 0.04237 | 0.95095 | 0.95963 | 0.78900 | 0.68413 |
| update | 7 | off | 1334 | 0.28898 | 0.04148 | 0.95357 | 0.96177 | 0.79547 | 0.69684 |
| update | 7 | off | 1434 | 0.27257 | 0.04086 | 0.95040 | 0.95920 | 0.77348 | 0.67426 |
| update | 7 | on | 1234 | 0.27799 | 0.03827 | 0.94961 | 0.95828 | 0.79538 | 0.68999 |
| update | 7 | on | 1334 | 0.28543 | 0.04175 | 0.95362 | 0.96230 | 0.79899 | 0.70088 |
| update | 7 | on | 1434 | 0.28099 | 0.04016 | 0.95068 | 0.95954 | 0.78258 | 0.68332 |
| recompute | 7 | off | 1234 | 0.27445 | 0.04141 | 0.95188 | 0.95984 | 0.79201 | 0.68686 |
| recompute | 7 | off | 1334 | 0.27177 | 0.04416 | 0.95601 | 0.96359 | 0.79463 | 0.69726 |
| recompute | 7 | off | 1434 | 0.26953 | 0.04088 | 0.95367 | 0.96126 | 0.77449 | 0.67709 |
| recompute | 7 | on | 1234 | 0.27066 | 0.04133 | 0.95234 | 0.96066 | 0.78725 | 0.68346 |
| recompute | 7 | on | 1334 | 0.27423 | 0.04050 | 0.95401 | 0.96219 | 0.78239 | 0.68355 |
| recompute | 7 | on | 1434 | 0.27862 | 0.04193 | 0.95360 | 0.96155 | 0.78511 | 0.68732 |
_3-seed mean MRR:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| feature | 0.343 | 0.307 | 0.280 |
| keep | 0.335 | 0.300 | 0.280 |
| update proc-off | 0.338 | 0.301 | 0.282 |
| update proc-on | 0.330 | 0.302 | 0.281 |
| recompute proc-off | 0.319 | 0.292 | 0.272 |
| recompute proc-on | 0.325 | 0.297 | 0.275 |

### Reddit-title (177 snapshots/run) — COMPLETE (54/54, C{3,5,7})
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.41065 | 0.03745 | 0.97487 | 0.97982 | 0.89255 | 0.82259 |
| feature | 3 | – | 1334 | 0.41694 | 0.04550 | 0.97579 | 0.98022 | 0.88797 | 0.81698 |
| feature | 3 | – | 1434 | 0.42047 | 0.04281 | 0.97424 | 0.97926 | 0.90251 | 0.82871 |
| keep | 3 | – | 1234 | 0.42118 | 0.03577 | 0.97547 | 0.97992 | 0.89748 | 0.82655 |
| keep | 3 | – | 1334 | 0.41443 | 0.03565 | 0.97630 | 0.98052 | 0.89712 | 0.82671 |
| keep | 3 | – | 1434 | 0.41840 | 0.03926 | 0.97638 | 0.98082 | 0.90439 | 0.83698 |
| update | 3 | off | 1234 | 0.41222 | 0.03767 | 0.97502 | 0.97964 | 0.89463 | 0.82355 |
| update | 3 | off | 1334 | 0.41612 | 0.04178 | 0.97642 | 0.98071 | 0.89766 | 0.82928 |
| update | 3 | off | 1434 | 0.42360 | 0.04230 | 0.97680 | 0.98124 | 0.90323 | 0.83672 |
| update | 3 | on | 1234 | 0.41868 | 0.03703 | 0.97498 | 0.97951 | 0.89838 | 0.82859 |
| update | 3 | on | 1334 | 0.41899 | 0.03573 | 0.97682 | 0.98119 | 0.89702 | 0.82858 |
| update | 3 | on | 1434 | 0.41800 | 0.04139 | 0.97679 | 0.98127 | 0.90389 | 0.83784 |
| recompute | 3 | off | 1234 | 0.39294 | 0.04319 | 0.97680 | 0.98047 | 0.89149 | 0.81920 |
| recompute | 3 | off | 1334 | 0.40426 | 0.05106 | 0.97757 | 0.98119 | 0.88994 | 0.81722 |
| recompute | 3 | off | 1434 | 0.39516 | 0.04449 | 0.97782 | 0.98139 | 0.89804 | 0.82785 |
| recompute | 3 | on | 1234 | 0.38985 | 0.04172 | 0.97668 | 0.98058 | 0.88894 | 0.81624 |
| recompute | 3 | on | 1334 | 0.40461 | 0.05369 | 0.97738 | 0.98119 | 0.88886 | 0.81579 |
| recompute | 3 | on | 1434 | 0.39967 | 0.05169 | 0.97764 | 0.98155 | 0.90169 | 0.83346 |
| feature | 5 | – | 1234 | 0.39212 | 0.03937 | 0.97197 | 0.97715 | 0.85752 | 0.78091 |
| feature | 5 | – | 1334 | 0.40089 | 0.04233 | 0.97452 | 0.97880 | 0.86283 | 0.78691 |
| feature | 5 | – | 1434 | 0.39373 | 0.04170 | 0.97290 | 0.97773 | 0.87707 | 0.79582 |
| keep | 5 | – | 1234 | 0.38983 | 0.03924 | 0.97317 | 0.97765 | 0.87072 | 0.79154 |
| keep | 5 | – | 1334 | 0.39686 | 0.04122 | 0.97474 | 0.97898 | 0.86865 | 0.79239 |
| keep | 5 | – | 1434 | 0.38648 | 0.03845 | 0.97450 | 0.97903 | 0.88534 | 0.81225 |
| update | 5 | off | 1234 | 0.39057 | 0.03842 | 0.97254 | 0.97743 | 0.87225 | 0.79375 |
| update | 5 | off | 1334 | 0.39248 | 0.03863 | 0.97380 | 0.97852 | 0.86824 | 0.79322 |
| update | 5 | off | 1434 | 0.39383 | 0.03606 | 0.97460 | 0.97903 | 0.88219 | 0.80808 |
| update | 5 | on | 1234 | 0.39469 | 0.03575 | 0.97315 | 0.97790 | 0.86963 | 0.79027 |
| update | 5 | on | 1334 | 0.39375 | 0.04047 | 0.97432 | 0.97873 | 0.86889 | 0.79416 |
| update | 5 | on | 1434 | 0.39972 | 0.03800 | 0.97480 | 0.97919 | 0.88291 | 0.80891 |
| recompute | 5 | off | 1234 | 0.38468 | 0.04081 | 0.97499 | 0.97908 | 0.86268 | 0.78161 |
| recompute | 5 | off | 1334 | 0.38438 | 0.04261 | 0.97544 | 0.97951 | 0.85584 | 0.77625 |
| recompute | 5 | off | 1434 | 0.38484 | 0.04416 | 0.97578 | 0.97990 | 0.87226 | 0.79476 |
| recompute | 5 | on | 1234 | 0.38020 | 0.04336 | 0.97468 | 0.97869 | 0.86280 | 0.78191 |
| recompute | 5 | on | 1334 | 0.38295 | 0.04486 | 0.97558 | 0.97968 | 0.86300 | 0.78520 |
| recompute | 5 | on | 1434 | 0.38517 | 0.04786 | 0.97547 | 0.97980 | 0.87005 | 0.79200 |
| feature | 7 | – | 1234 | 0.38749 | 0.03913 | 0.96898 | 0.97488 | 0.82368 | 0.73884 |
| feature | 7 | – | 1334 | 0.39282 | 0.03899 | 0.97186 | 0.97628 | 0.79755 | 0.71188 |
| feature | 7 | – | 1434 | 0.38828 | 0.05121 | 0.96838 | 0.97414 | 0.82468 | 0.73281 |
| keep | 7 | – | 1234 | 0.39058 | 0.04597 | 0.96870 | 0.97430 | 0.83506 | 0.75220 |
| keep | 7 | – | 1334 | 0.39217 | 0.04182 | 0.97187 | 0.97664 | 0.81380 | 0.72950 |
| keep | 7 | – | 1434 | 0.39263 | 0.04237 | 0.97289 | 0.97761 | 0.84673 | 0.76505 |
| update | 7 | off | 1234 | 0.38708 | 0.04595 | 0.96972 | 0.97499 | 0.82813 | 0.74379 |
| update | 7 | off | 1334 | 0.38176 | 0.04116 | 0.97172 | 0.97660 | 0.81712 | 0.73502 |
| update | 7 | off | 1434 | 0.38841 | 0.04416 | 0.97214 | 0.97703 | 0.84446 | 0.76139 |
| update | 7 | on | 1234 | 0.39161 | 0.04665 | 0.96950 | 0.97479 | 0.82755 | 0.74348 |
| update | 7 | on | 1334 | 0.38820 | 0.04184 | 0.97170 | 0.97663 | 0.81422 | 0.73098 |
| update | 7 | on | 1434 | 0.38705 | 0.04132 | 0.97198 | 0.97683 | 0.83911 | 0.75576 |
| recompute | 7 | off | 1234 | 0.38122 | 0.04776 | 0.97030 | 0.97540 | 0.80468 | 0.71779 |
| recompute | 7 | off | 1334 | 0.38119 | 0.04065 | 0.97357 | 0.97796 | 0.79529 | 0.71236 |
| recompute | 7 | off | 1434 | 0.37609 | 0.04345 | 0.97365 | 0.97791 | 0.83542 | 0.75115 |
| recompute | 7 | on | 1234 | 0.38202 | 0.04882 | 0.97026 | 0.97524 | 0.79496 | 0.70883 |
| recompute | 7 | on | 1334 | 0.38346 | 0.04126 | 0.97329 | 0.97769 | 0.78666 | 0.70279 |
| recompute | 7 | on | 1434 | 0.37859 | 0.04611 | 0.97329 | 0.97759 | 0.82390 | 0.73796 |
_3-seed mean MRR:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| feature | 0.416 | 0.396 | 0.390 |
| keep | 0.418 | 0.391 | 0.392 |
| update proc-off | 0.417 | 0.392 | 0.386 |
| update proc-on | 0.419 | 0.396 | 0.389 |
| recompute proc-off | 0.397 | 0.385 | 0.380 |
| recompute proc-on | 0.398 | 0.383 | 0.381 |

### Bitcoin-alpha (225 snapshots/run) — COMPLETE (54/54, C{3,5,7})
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.13362 | 0.10480 | 0.88957 | 0.91224 | 0.75520 | 0.61181 |
| feature | 3 | – | 1334 | 0.12236 | 0.09540 | 0.88834 | 0.91041 | 0.80297 | 0.65199 |
| feature | 3 | – | 1434 | 0.12115 | 0.11464 | 0.90979 | 0.92853 | 0.71677 | 0.56103 |
| keep | 3 | – | 1234 | 0.12229 | 0.09752 | 0.86828 | 0.90065 | 0.75296 | 0.63689 |
| keep | 3 | – | 1334 | 0.11384 | 0.09127 | 0.89055 | 0.91331 | 0.76446 | 0.56801 |
| keep | 3 | – | 1434 | 0.11577 | 0.11159 | 0.87712 | 0.90863 | 0.77320 | 0.61568 |
| update | 3 | off | 1234 | 0.11634 | 0.09132 | 0.87845 | 0.90722 | 0.73802 | 0.61225 |
| update | 3 | off | 1334 | 0.11772 | 0.09083 | 0.89940 | 0.91944 | 0.78653 | 0.64362 |
| update | 3 | off | 1434 | 0.10910 | 0.09748 | 0.86740 | 0.90171 | 0.70434 | 0.56102 |
| update | 3 | on | 1234 | 0.12291 | 0.09280 | 0.88770 | 0.91343 | 0.76358 | 0.63669 |
| update | 3 | on | 1334 | 0.12335 | 0.10209 | 0.89651 | 0.91861 | 0.74681 | 0.59719 |
| update | 3 | on | 1434 | 0.11829 | 0.09954 | 0.88680 | 0.91205 | 0.77086 | 0.61874 |
| recompute | 3 | off | 1234 | 0.10425 | 0.09678 | 0.84551 | 0.87941 | 0.64136 | 0.52789 |
| recompute | 3 | off | 1334 | 0.10106 | 0.09069 | 0.86226 | 0.89239 | 0.72823 | 0.56682 |
| recompute | 3 | off | 1434 | 0.10320 | 0.09827 | 0.86591 | 0.89941 | 0.68127 | 0.54962 |
| recompute | 3 | on | 1234 | 0.10332 | 0.09172 | 0.84274 | 0.88276 | 0.67769 | 0.57243 |
| recompute | 3 | on | 1334 | 0.11131 | 0.08994 | 0.85775 | 0.89153 | 0.71322 | 0.58013 |
| recompute | 3 | on | 1434 | 0.09690 | 0.08279 | 0.86578 | 0.89770 | 0.72470 | 0.60955 |
| feature | 5 | – | 1234 | 0.10454 | 0.10664 | 0.83046 | 0.86186 | 0.53037 | 0.27373 |
| feature | 5 | – | 1334 | 0.10319 | 0.09099 | 0.73859 | 0.80998 | 0.61872 | 0.36220 |
| feature | 5 | – | 1434 | 0.09838 | 0.08376 | 0.80749 | 0.84904 | 0.61340 | 0.44698 |
| keep | 5 | – | 1234 | 0.06605 | 0.06774 | 0.80540 | 0.84277 | 0.66751 | 0.49937 |
| keep | 5 | – | 1334 | 0.06040 | 0.06440 | 0.74566 | 0.80303 | 0.59822 | 0.36800 |
| keep | 5 | – | 1434 | 0.09384 | 0.08782 | 0.82147 | 0.86432 | 0.64229 | 0.46243 |
| update | 5 | off | 1234 | 0.08590 | 0.07749 | 0.82536 | 0.85632 | 0.67316 | 0.47149 |
| update | 5 | off | 1334 | 0.07384 | 0.07135 | 0.80720 | 0.84800 | 0.67725 | 0.39061 |
| update | 5 | off | 1434 | 0.07867 | 0.07699 | 0.75423 | 0.81257 | 0.55298 | 0.34530 |
| update | 5 | on | 1234 | 0.08727 | 0.08452 | 0.85133 | 0.87474 | 0.69185 | 0.47981 |
| update | 5 | on | 1334 | 0.08961 | 0.08135 | 0.84894 | 0.87931 | 0.72914 | 0.47749 |
| update | 5 | on | 1434 | 0.10279 | 0.11503 | 0.85183 | 0.87715 | 0.53405 | 0.35664 |
| recompute | 5 | off | 1234 | 0.05323 | 0.06881 | 0.72897 | 0.77959 | 0.49282 | 0.31392 |
| recompute | 5 | off | 1334 | 0.06125 | 0.06795 | 0.76748 | 0.81670 | 0.44905 | 0.33015 |
| recompute | 5 | off | 1434 | 0.05355 | 0.07118 | 0.72892 | 0.77976 | 0.40600 | 0.29384 |
| recompute | 5 | on | 1234 | 0.06878 | 0.07627 | 0.76123 | 0.80489 | 0.48058 | 0.33243 |
| recompute | 5 | on | 1334 | 0.05783 | 0.07091 | 0.72088 | 0.77401 | 0.56942 | 0.27588 |
| recompute | 5 | on | 1434 | 0.06046 | 0.07684 | 0.77552 | 0.81882 | 0.52264 | 0.36459 |
| feature | 7 | – | 1234 | 0.08914 | 0.08460 | 0.78544 | 0.82689 | 0.33352 | 0.24806 |
| feature | 7 | – | 1334 | 0.09184 | 0.08596 | 0.78453 | 0.82814 | 0.44379 | 0.17689 |
| feature | 7 | – | 1434 | 0.09320 | 0.08575 | 0.67682 | 0.76705 | 0.41428 | 0.24682 |
| keep | 7 | – | 1234 | 0.06765 | 0.09382 | 0.75018 | 0.80275 | 0.32485 | 0.24211 |
| keep | 7 | – | 1334 | 0.03041 | 0.04025 | 0.78524 | 0.81096 | 0.56064 | 0.33080 |
| keep | 7 | – | 1434 | 0.05143 | 0.06589 | 0.76489 | 0.80864 | 0.54825 | 0.23127 |
| update | 7 | off | 1234 | 0.07074 | 0.09428 | 0.71713 | 0.78299 | 0.60007 | 0.26371 |
| update | 7 | off | 1334 | 0.03799 | 0.05228 | 0.74499 | 0.78770 | 0.58745 | 0.35328 |
| update | 7 | off | 1434 | 0.05211 | 0.05755 | 0.75645 | 0.80475 | 0.59955 | 0.33270 |
| update | 7 | on | 1234 | 0.09914 | 0.10818 | 0.83637 | 0.87065 | 0.66650 | 0.53457 |
| update | 7 | on | 1334 | 0.04038 | 0.05843 | 0.77878 | 0.80929 | 0.58339 | 0.23555 |
| update | 7 | on | 1434 | 0.05832 | 0.07072 | 0.79157 | 0.82568 | 0.55011 | 0.27917 |
| recompute | 7 | off | 1234 | 0.03711 | 0.07168 | 0.64728 | 0.69916 | 0.23426 | 0.13035 |
| recompute | 7 | off | 1334 | 0.02768 | 0.03822 | 0.63862 | 0.70403 | 0.29879 | 0.15912 |
| recompute | 7 | off | 1434 | 0.04285 | 0.05625 | 0.66506 | 0.72878 | 0.29245 | 0.22923 |
| recompute | 7 | on | 1234 | 0.03780 | 0.07259 | 0.60393 | 0.66834 | 0.15896 | 0.09315 |
| recompute | 7 | on | 1334 | 0.01889 | 0.03104 | 0.61320 | 0.67237 | 0.27396 | 0.09651 |
| recompute | 7 | on | 1434 | 0.02359 | 0.03874 | 0.58261 | 0.65358 | 0.15408 | 0.09683 |
_3-seed mean MRR:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| feature | 0.126 | 0.102 | 0.091 |
| keep | 0.117 | 0.073 | 0.050 |
| update proc-off | 0.114 | 0.079 | 0.054 |
| update proc-on | 0.122 | 0.093 | 0.066 |
| recompute proc-off | 0.103 | 0.056 | 0.036 |
| recompute proc-on | 0.104 | 0.062 | 0.027 |

### Bitcoin-otc (261 snapshots/run) — COMPLETE (54/54, C{3,5,7})
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 3 | – | 1234 | 0.15130 | 0.12743 | 0.89367 | 0.91606 | 0.72848 | 0.58311 |
| feature | 3 | – | 1334 | 0.16857 | 0.11493 | 0.90460 | 0.92681 | 0.77760 | 0.59789 |
| feature | 3 | – | 1434 | 0.13447 | 0.10242 | 0.88843 | 0.91039 | 0.79619 | 0.68273 |
| keep | 3 | – | 1234 | 0.15540 | 0.12041 | 0.87199 | 0.90672 | 0.74497 | 0.63167 |
| keep | 3 | – | 1334 | 0.15822 | 0.11166 | 0.88346 | 0.91569 | 0.77284 | 0.59044 |
| keep | 3 | – | 1434 | 0.13037 | 0.10638 | 0.82368 | 0.87212 | 0.77594 | 0.56034 |
| update | 3 | off | 1234 | 0.13298 | 0.10411 | 0.88487 | 0.91332 | 0.77936 | 0.65285 |
| update | 3 | off | 1334 | 0.14958 | 0.11148 | 0.87915 | 0.91346 | 0.77961 | 0.60530 |
| update | 3 | off | 1434 | 0.12040 | 0.08926 | 0.87099 | 0.90659 | 0.80275 | 0.66152 |
| update | 3 | on | 1234 | 0.14140 | 0.11827 | 0.86767 | 0.90354 | 0.75841 | 0.61389 |
| update | 3 | on | 1334 | 0.15380 | 0.11801 | 0.88735 | 0.91735 | 0.80683 | 0.68742 |
| update | 3 | on | 1434 | 0.12148 | 0.09805 | 0.84776 | 0.88871 | 0.68678 | 0.51193 |
| recompute | 3 | off | 1234 | 0.10103 | 0.09064 | 0.84591 | 0.88209 | 0.64326 | 0.51688 |
| recompute | 3 | off | 1334 | 0.12881 | 0.10999 | 0.88154 | 0.91030 | 0.72525 | 0.60553 |
| recompute | 3 | off | 1434 | 0.11138 | 0.10982 | 0.86177 | 0.89660 | 0.69000 | 0.56593 |
| recompute | 3 | on | 1234 | 0.10698 | 0.10348 | 0.85802 | 0.89206 | 0.69077 | 0.54191 |
| recompute | 3 | on | 1334 | 0.12933 | 0.11301 | 0.89219 | 0.91712 | 0.73101 | 0.58953 |
| recompute | 3 | on | 1434 | 0.12131 | 0.10240 | 0.87538 | 0.90229 | 0.73051 | 0.58829 |
| feature | 5 | – | 1234 | 0.11993 | 0.10508 | 0.76533 | 0.82850 | 0.53173 | 0.37959 |
| feature | 5 | – | 1334 | 0.12995 | 0.10287 | 0.84527 | 0.87338 | 0.55093 | 0.35374 |
| feature | 5 | – | 1434 | 0.14431 | 0.12742 | 0.82058 | 0.86524 | 0.61736 | 0.33958 |
| keep | 5 | – | 1234 | 0.10224 | 0.09444 | 0.73442 | 0.80569 | 0.52745 | 0.37841 |
| keep | 5 | – | 1334 | 0.12511 | 0.10531 | 0.79430 | 0.85016 | 0.54406 | 0.40200 |
| keep | 5 | – | 1434 | 0.09765 | 0.08726 | 0.75865 | 0.83056 | 0.68197 | 0.34754 |
| update | 5 | off | 1234 | 0.11341 | 0.11945 | 0.80876 | 0.85460 | 0.67340 | 0.52399 |
| update | 5 | off | 1334 | 0.10571 | 0.09202 | 0.78966 | 0.84161 | 0.55285 | 0.39388 |
| update | 5 | off | 1434 | 0.10025 | 0.08938 | 0.73017 | 0.80581 | 0.68397 | 0.34486 |
| update | 5 | on | 1234 | 0.13674 | 0.11518 | 0.81302 | 0.85865 | 0.64031 | 0.49167 |
| update | 5 | on | 1334 | 0.10815 | 0.10121 | 0.76126 | 0.82402 | 0.54304 | 0.39469 |
| update | 5 | on | 1434 | 0.12713 | 0.12382 | 0.80969 | 0.86039 | 0.69075 | 0.48581 |
| recompute | 5 | off | 1234 | 0.07334 | 0.08763 | 0.70610 | 0.76736 | 0.51207 | 0.28042 |
| recompute | 5 | off | 1334 | 0.08623 | 0.09152 | 0.80274 | 0.83859 | 0.49337 | 0.36538 |
| recompute | 5 | off | 1434 | 0.09222 | 0.09158 | 0.75412 | 0.81131 | 0.60156 | 0.30794 |
| recompute | 5 | on | 1234 | 0.08834 | 0.08926 | 0.75844 | 0.81050 | 0.58276 | 0.35912 |
| recompute | 5 | on | 1334 | 0.07398 | 0.07642 | 0.77175 | 0.81942 | 0.45702 | 0.34642 |
| recompute | 5 | on | 1434 | 0.08344 | 0.07718 | 0.77423 | 0.82912 | 0.60143 | 0.35167 |
| feature | 7 | – | 1234 | 0.12191 | 0.10919 | 0.83102 | 0.86098 | 0.67707 | 0.51293 |
| feature | 7 | – | 1334 | 0.11894 | 0.10123 | 0.84291 | 0.86779 | 0.61611 | 0.25290 |
| feature | 7 | – | 1434 | 0.09899 | 0.10342 | 0.80056 | 0.82665 | 0.52896 | 0.22670 |
| keep | 7 | – | 1234 | 0.07007 | 0.09334 | 0.70214 | 0.76901 | 0.43175 | 0.24702 |
| keep | 7 | – | 1334 | 0.11295 | 0.10371 | 0.75317 | 0.81765 | 0.59163 | 0.32105 |
| keep | 7 | – | 1434 | 0.04959 | 0.08704 | 0.59935 | 0.67635 | 0.41658 | 0.13119 |
| update | 7 | off | 1234 | 0.06671 | 0.07596 | 0.63887 | 0.72423 | 0.43639 | 0.26135 |
| update | 7 | off | 1334 | 0.07916 | 0.08515 | 0.70039 | 0.77359 | 0.52863 | 0.30902 |
| update | 7 | off | 1434 | 0.05443 | 0.07711 | 0.61731 | 0.68677 | 0.54235 | 0.16325 |
| update | 7 | on | 1234 | 0.09489 | 0.08750 | 0.79951 | 0.84103 | 0.58191 | 0.32058 |
| update | 7 | on | 1334 | 0.08975 | 0.09085 | 0.72656 | 0.79210 | 0.53399 | 0.25534 |
| update | 7 | on | 1434 | 0.06686 | 0.09204 | 0.62045 | 0.70103 | 0.44834 | 0.13384 |
| recompute | 7 | off | 1234 | 0.04550 | 0.06515 | 0.70762 | 0.74898 | 0.28359 | 0.20076 |
| recompute | 7 | off | 1334 | 0.04254 | 0.05495 | 0.73194 | 0.78146 | 0.53070 | 0.28657 |
| recompute | 7 | off | 1434 | 0.06519 | 0.09272 | 0.71796 | 0.76128 | 0.46655 | 0.23154 |
| recompute | 7 | on | 1234 | 0.05075 | 0.07990 | 0.70098 | 0.74983 | 0.25283 | 0.19071 |
| recompute | 7 | on | 1334 | 0.08577 | 0.10733 | 0.74729 | 0.79999 | 0.53200 | 0.39381 |
| recompute | 7 | on | 1434 | 0.04015 | 0.05905 | 0.70771 | 0.75021 | 0.46286 | 0.21522 |
_3-seed mean MRR:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| feature | 0.151 | 0.131 | 0.113 |
| keep | 0.148 | 0.108 | 0.078 |
| update proc-off | 0.134 | 0.106 | 0.067 |
| update proc-on | 0.139 | 0.124 | 0.084 |
| recompute proc-off | 0.114 | 0.084 | 0.051 |
| recompute proc-on | 0.119 | 0.082 | 0.059 |

**Cross-dataset nuance (the paper must not overclaim) — now COMPLETE at C{3,5,7}.** The
mode ordering and the "spectral rescues federated" crossover are dataset-dependent, tracking
(a) whether the frozen global basis survives sharding and (b) how much plain `feature` actually
degrades under it:
- **as733** (7716 nodes, daily, LOW churn): `keep` stays ~flat (0.307/0.282/0.282 over C3/5/7)
  while feature declines (0.314/0.280/0.268), so **keep overtakes feature by C5 and clearly at
  C7** (0.282 vs 0.268). `recompute`-on is competitive-to-best at C3/C5 (0.322/0.291) — a fresh
  basis barely moves on stable AS topology — and procrustes matters a lot for recompute
  (C3 on 0.322 vs off 0.299; C5 on 0.291 vs off 0.276).
- **reddit_body** (35776 nodes, weekly): feature LEADS at C3 (0.343 vs keep 0.335) but the gap
  closes as C grows; by **C7 update≈keep≈feature≈0.280** — a LATE crossover. recompute mildly
  worst throughout.
- **reddit_title** (35776 nodes, weekly): feature is remarkably robust to sharding (C1 0.414 →
  C7 0.390, only −6%), so `keep`/`update` merely **track feature at every C** (keep nominally
  highest at C3 0.418 and C7 0.392); no dramatic crossover because feature never gives up much
  ground. recompute consistently ~0.015 back.
- **bitcoin_alpha/otc** (~4–6k nodes, small): feature is best AND **widens its lead as C grows** —
  keep degrades far faster (alpha keep 0.117→0.073→0.050 vs feature 0.126→0.102→0.091; otc keep
  0.148→0.108→0.078 vs feature 0.151→0.131→0.113), so there is **NO rescue at any C{3,5,7}**; the
  frozen t0 basis itself degenerates once these small graphs are sharded. recompute worst throughout.
  **REVISED by §8.5:** this is a property of the *Laplace* smodel — a higher-capacity encoder (SignNet)
  removes the failure: decisively on bitcoin_otc (+0.026 over feature at C7), to parity on bitcoin_alpha
  (sub-noise). NB §8.5: that gain is general encoder capacity, not the sign-invariance mechanism.
- **UCI** (1899 nodes, weekly, §3.1): keep overtakes feature by C7 (0.076 vs 0.073); recompute worst.

Honest framing: **"spectral rescues federated" is NOT a law.** `keep` matches-or-beats plain
federated where the graph is large/stable enough that a frozen basis survives sharding (UCI,
as733, reddit) — but the *effect size* depends on how much feature degrades (large on
as733/reddit_body/UCI, negligible on the sharding-robust reddit_title). It **fails outright on the
small Bitcoin graphs**, where the t0 basis degrades under sharding and feature only pulls further
ahead (**but see §8.5** — that failure is specific to the Laplace smodel; SignNet reverses it). **`recompute` is churn-dependent**: clearly worst on high-churn UCI/reddit/bitcoin, but
competitive-to-best on low-churn as733 (procrustes-on). Mechanism §4.2.

### sfv_share=avg ablation — ALL 6 datasets (270 runs, C{3,5,7})  **[complete, 2026-07-13]**
Does FedAvg-ing the learnable spectral filter `W` across clients (`federated.sfv_share=avg` — basic-FedLap's SFV-sync) beat the per-client `local` default (each client owns its `W`)? Run on **all 6 datasets**, f+s modes only (feature has no `W`), same grid as the §6 `local` baseline, gru, C{3,5,7} × 3 seeds = 45/dataset. All `device=cuda:0`.

**avg vs local — `mean_mrr(local)→mean_mrr(avg) (Δ)`, 3-seed means:**

_uci:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.084→0.078 (-0.006) | 0.066→0.073 (+0.006) | 0.076→0.075 (-0.000) |
| update proc-off | 0.078→0.077 (-0.001) | 0.066→0.058 (-0.008) | 0.071→0.073 (+0.001) |
| update proc-on | 0.082→0.080 (-0.002) | 0.078→0.070 (-0.008) | 0.077→0.066 (-0.011) |
| recompute proc-off | 0.055→0.065 (+0.010) | 0.060→0.060 (+0.001) | 0.058→0.062 (+0.004) |
| recompute proc-on | 0.060→0.062 (+0.003) | 0.054→0.051 (-0.003) | 0.061→0.058 (-0.004) |

_bitcoin_alpha:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.117→0.123 (+0.005) | 0.073→0.078 (+0.004) | 0.050→0.043 (-0.007) |
| update proc-off | 0.114→0.120 (+0.006) | 0.079→0.074 (-0.005) | 0.054→0.061 (+0.007) |
| update proc-on | 0.122→0.123 (+0.001) | 0.093→0.093 (+0.000) | 0.066→0.062 (-0.004) |
| recompute proc-off | 0.103→0.096 (-0.006) | 0.056→0.072 (+0.016) | 0.036→0.038 (+0.002) |
| recompute proc-on | 0.104→0.105 (+0.001) | 0.062→0.072 (+0.010) | 0.027→0.041 (+0.015) |

_bitcoin_otc:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.148→0.134 (-0.014) | 0.108→0.115 (+0.007) | 0.078→0.095 (+0.017) |
| update proc-off | 0.134→0.131 (-0.003) | 0.106→0.101 (-0.006) | 0.067→0.084 (+0.018) |
| update proc-on | 0.139→0.131 (-0.008) | 0.124→0.109 (-0.015) | 0.084→0.075 (-0.009) |
| recompute proc-off | 0.114→0.122 (+0.008) | 0.084→0.075 (-0.009) | 0.051→0.060 (+0.009) |
| recompute proc-on | 0.119→0.117 (-0.002) | 0.082→0.081 (-0.001) | 0.059→0.041 (-0.018) |

_as733:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.307→0.318 (+0.011) | 0.282→0.286 (+0.004) | 0.282→0.274 (-0.008) |
| update proc-off | 0.301→0.296 (-0.004) | 0.280→0.281 (+0.001) | 0.276→0.269 (-0.006) |
| update proc-on | 0.298→0.301 (+0.003) | 0.274→0.282 (+0.007) | 0.268→0.280 (+0.012) |
| recompute proc-off | 0.299→0.309 (+0.011) | 0.276→0.280 (+0.005) | 0.277→0.269 (-0.008) |
| recompute proc-on | 0.322→0.355 (+0.033) | 0.291→0.303 (+0.012) | 0.278→0.297 (+0.019) |

_reddit_title:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.418→0.412 (-0.006) | 0.391→0.395 (+0.004) | 0.392→0.387 (-0.005) |
| update proc-off | 0.417→0.411 (-0.006) | 0.392→0.399 (+0.007) | 0.386→0.389 (+0.003) |
| update proc-on | 0.419→0.411 (-0.007) | 0.396→0.395 (-0.001) | 0.389→0.391 (+0.002) |
| recompute proc-off | 0.397→0.398 (+0.001) | 0.385→0.389 (+0.004) | 0.380→0.388 (+0.008) |
| recompute proc-on | 0.398→0.399 (+0.001) | 0.383→0.388 (+0.005) | 0.381→0.383 (+0.001) |

_reddit_body:_
| mode | C3 | C5 | C7 |
|---|---|---|---|
| keep | 0.335→0.330 (-0.006) | 0.300→0.298 (-0.002) | 0.280→0.288 (+0.008) |
| update proc-off | 0.338→0.334 (-0.004) | 0.301→0.309 (+0.008) | 0.282→0.284 (+0.003) |
| update proc-on | 0.330→0.339 (+0.009) | 0.302→0.301 (-0.001) | 0.281→0.285 (+0.003) |
| recompute proc-off | 0.319→0.317 (-0.002) | 0.292→0.292 (+0.000) | 0.272→0.275 (+0.003) |
| recompute proc-on | 0.325→0.323 (-0.002) | 0.297→0.296 (-0.002) | 0.275→0.275 (+0.000) |

**Verdict — the avg≈local null is UNIVERSAL.** Across 90 (dataset×mode×C) conditions:
**mean Δ = +0.001, mean |Δ| = 0.006** — federating `W` is statistically indistinguishable from
keeping it `local`, with NO systematic direction (as many + as −). On the two LARGE/stable
datasets (reddit_body, reddit_title) **every** condition is within ±0.01 (max |Δ| 0.009/0.008) — a
clean null. 15/90 conditions exceed ±0.01, concentrated on the SMALL/noisy graphs (bitcoin, uci)
where they are mixed-sign (ordinary multi-seed noise on ~2–6k-node graphs). The one mild
**systematic** exception is **as733 `recompute` proc-on, where avg runs ~0.012–0.033 higher than
local at every C** — a small candidate real effect (federating `W` may stabilize the churny
recompute basis on the low-churn AS graph), but it is noisy and confined to one mode+dataset, so
not overclaimed. **Conclusion (extends the earlier reddit_body result to all 6):** `sfv_share` is
**not load-bearing** — it neither systematically helps nor hurts (refuting the a-priori worry that
averaging `W`s trained on disjoint spectral slices would degrade them). `local` (the default,
zero ongoing `W` exchange) is the sensible pick on communication grounds; `avg` buys nothing
measurable except a possible small stabilization of `recompute` on stable graphs. Confined to gru.

**Per-run avg tables (record, all metrics):**

_UCI avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.06864 | 0.04247 | 0.85183 | 0.86513 | 0.70928 | 0.56308 |
| keep | 3 | – | 1334 | 0.08671 | 0.06159 | 0.86089 | 0.87162 | 0.71694 | 0.54438 |
| keep | 3 | – | 1434 | 0.07920 | 0.05103 | 0.84690 | 0.86405 | 0.68381 | 0.53561 |
| update | 3 | off | 1234 | 0.06831 | 0.03992 | 0.85692 | 0.86392 | 0.68944 | 0.55194 |
| update | 3 | off | 1334 | 0.09004 | 0.07177 | 0.86054 | 0.87403 | 0.70206 | 0.55241 |
| update | 3 | off | 1434 | 0.07285 | 0.06855 | 0.84151 | 0.86154 | 0.71507 | 0.55286 |
| update | 3 | on | 1234 | 0.07577 | 0.03646 | 0.86369 | 0.87574 | 0.68587 | 0.54735 |
| update | 3 | on | 1334 | 0.10480 | 0.07782 | 0.85747 | 0.86830 | 0.71218 | 0.56041 |
| update | 3 | on | 1434 | 0.06053 | 0.03137 | 0.85837 | 0.87410 | 0.70079 | 0.55228 |
| recompute | 3 | off | 1234 | 0.07042 | 0.06075 | 0.82276 | 0.83887 | 0.57621 | 0.44861 |
| recompute | 3 | off | 1334 | 0.06960 | 0.05453 | 0.82512 | 0.84061 | 0.58350 | 0.44548 |
| recompute | 3 | off | 1434 | 0.05542 | 0.03721 | 0.81917 | 0.82748 | 0.56547 | 0.43267 |
| recompute | 3 | on | 1234 | 0.05235 | 0.02657 | 0.83809 | 0.84836 | 0.58338 | 0.45977 |
| recompute | 3 | on | 1334 | 0.07221 | 0.07004 | 0.83391 | 0.84810 | 0.61963 | 0.47467 |
| recompute | 3 | on | 1434 | 0.06287 | 0.05419 | 0.82878 | 0.84288 | 0.57557 | 0.45533 |
| keep | 5 | – | 1234 | 0.06426 | 0.03452 | 0.81521 | 0.83883 | 0.58234 | 0.43806 |
| keep | 5 | – | 1334 | 0.07463 | 0.04982 | 0.81546 | 0.83582 | 0.59529 | 0.45586 |
| keep | 5 | – | 1434 | 0.07932 | 0.04740 | 0.81874 | 0.84455 | 0.58347 | 0.45645 |
| update | 5 | off | 1234 | 0.05727 | 0.03029 | 0.81246 | 0.83038 | 0.54400 | 0.43069 |
| update | 5 | off | 1334 | 0.05443 | 0.02281 | 0.80230 | 0.82623 | 0.59226 | 0.43668 |
| update | 5 | off | 1434 | 0.06279 | 0.03265 | 0.83557 | 0.85604 | 0.60108 | 0.45841 |
| update | 5 | on | 1234 | 0.06401 | 0.03670 | 0.84173 | 0.85708 | 0.55576 | 0.45669 |
| update | 5 | on | 1334 | 0.06712 | 0.04883 | 0.81139 | 0.83634 | 0.62375 | 0.47390 |
| update | 5 | on | 1434 | 0.07843 | 0.04690 | 0.82685 | 0.84754 | 0.57611 | 0.46798 |
| recompute | 5 | off | 1234 | 0.05494 | 0.03090 | 0.80032 | 0.81988 | 0.42252 | 0.34225 |
| recompute | 5 | off | 1334 | 0.06091 | 0.03202 | 0.77074 | 0.80031 | 0.47152 | 0.37591 |
| recompute | 5 | off | 1434 | 0.06544 | 0.04508 | 0.80455 | 0.81786 | 0.44149 | 0.35616 |
| recompute | 5 | on | 1234 | 0.04078 | 0.02286 | 0.77721 | 0.79515 | 0.42125 | 0.32290 |
| recompute | 5 | on | 1334 | 0.05851 | 0.04774 | 0.78279 | 0.79958 | 0.45493 | 0.33359 |
| recompute | 5 | on | 1434 | 0.05331 | 0.02954 | 0.78661 | 0.80586 | 0.46072 | 0.35983 |
| keep | 7 | – | 1234 | 0.06744 | 0.04047 | 0.80838 | 0.83186 | 0.46508 | 0.38453 |
| keep | 7 | – | 1334 | 0.06860 | 0.03158 | 0.77172 | 0.81132 | 0.44940 | 0.36247 |
| keep | 7 | – | 1434 | 0.08915 | 0.07627 | 0.80210 | 0.82447 | 0.49326 | 0.39846 |
| update | 7 | off | 1234 | 0.07956 | 0.07527 | 0.78757 | 0.81490 | 0.44383 | 0.33205 |
| update | 7 | off | 1334 | 0.06324 | 0.03455 | 0.78272 | 0.81389 | 0.45118 | 0.34194 |
| update | 7 | off | 1434 | 0.07486 | 0.06222 | 0.78738 | 0.81416 | 0.45202 | 0.35189 |
| update | 7 | on | 1234 | 0.05573 | 0.02592 | 0.79495 | 0.82320 | 0.45975 | 0.38296 |
| update | 7 | on | 1334 | 0.07693 | 0.05192 | 0.80468 | 0.82852 | 0.49660 | 0.39183 |
| update | 7 | on | 1434 | 0.06482 | 0.03677 | 0.76918 | 0.80023 | 0.46499 | 0.34782 |
| recompute | 7 | off | 1234 | 0.05636 | 0.04491 | 0.72543 | 0.75686 | 0.29935 | 0.24931 |
| recompute | 7 | off | 1334 | 0.05731 | 0.03786 | 0.75878 | 0.78448 | 0.32523 | 0.26404 |
| recompute | 7 | off | 1434 | 0.07138 | 0.07519 | 0.75680 | 0.78620 | 0.29609 | 0.27000 |
| recompute | 7 | on | 1234 | 0.05376 | 0.02977 | 0.75597 | 0.78326 | 0.33051 | 0.26666 |
| recompute | 7 | on | 1334 | 0.04757 | 0.02746 | 0.76058 | 0.79173 | 0.36898 | 0.31127 |
| recompute | 7 | on | 1434 | 0.07195 | 0.05006 | 0.75059 | 0.78464 | 0.34201 | 0.30520 |

_Bitcoin-alpha avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.13174 | 0.11589 | 0.90951 | 0.92821 | 0.78415 | 0.66411 |
| keep | 3 | – | 1334 | 0.11135 | 0.09238 | 0.89307 | 0.91369 | 0.78364 | 0.63877 |
| keep | 3 | – | 1434 | 0.12511 | 0.11416 | 0.89486 | 0.91852 | 0.77070 | 0.63065 |
| update | 3 | off | 1234 | 0.11838 | 0.09450 | 0.87733 | 0.90451 | 0.72327 | 0.59731 |
| update | 3 | off | 1334 | 0.12099 | 0.09187 | 0.89594 | 0.91973 | 0.78402 | 0.63649 |
| update | 3 | off | 1434 | 0.12165 | 0.10214 | 0.88924 | 0.91416 | 0.77718 | 0.64011 |
| update | 3 | on | 1234 | 0.13129 | 0.10091 | 0.90734 | 0.92714 | 0.79213 | 0.67369 |
| update | 3 | on | 1334 | 0.12625 | 0.10183 | 0.89989 | 0.91937 | 0.79445 | 0.64488 |
| update | 3 | on | 1434 | 0.11078 | 0.10543 | 0.89829 | 0.91940 | 0.77958 | 0.65621 |
| recompute | 3 | off | 1234 | 0.10992 | 0.10234 | 0.83812 | 0.87914 | 0.65931 | 0.54302 |
| recompute | 3 | off | 1334 | 0.09047 | 0.08217 | 0.84445 | 0.88135 | 0.64914 | 0.52033 |
| recompute | 3 | off | 1434 | 0.08911 | 0.08577 | 0.80753 | 0.85425 | 0.62024 | 0.49955 |
| recompute | 3 | on | 1234 | 0.10803 | 0.09060 | 0.87325 | 0.90133 | 0.71859 | 0.61178 |
| recompute | 3 | on | 1334 | 0.09157 | 0.08017 | 0.85913 | 0.88897 | 0.63000 | 0.49175 |
| recompute | 3 | on | 1434 | 0.11392 | 0.11170 | 0.85010 | 0.88445 | 0.68205 | 0.54838 |
| keep | 5 | – | 1234 | 0.05739 | 0.06022 | 0.83404 | 0.85545 | 0.65396 | 0.42763 |
| keep | 5 | – | 1334 | 0.08549 | 0.08515 | 0.81363 | 0.85251 | 0.62787 | 0.43929 |
| keep | 5 | – | 1434 | 0.09057 | 0.11166 | 0.81201 | 0.85657 | 0.57305 | 0.44123 |
| update | 5 | off | 1234 | 0.07049 | 0.06750 | 0.83530 | 0.86068 | 0.68717 | 0.49845 |
| update | 5 | off | 1334 | 0.07243 | 0.06903 | 0.78436 | 0.82872 | 0.68225 | 0.38517 |
| update | 5 | off | 1434 | 0.07944 | 0.08373 | 0.80121 | 0.84015 | 0.59163 | 0.38519 |
| update | 5 | on | 1234 | 0.10362 | 0.08970 | 0.86767 | 0.89021 | 0.73234 | 0.57127 |
| update | 5 | on | 1334 | 0.09146 | 0.09413 | 0.79149 | 0.83576 | 0.63591 | 0.35839 |
| update | 5 | on | 1434 | 0.08494 | 0.07761 | 0.83523 | 0.86574 | 0.61533 | 0.42826 |
| recompute | 5 | off | 1234 | 0.09691 | 0.11084 | 0.79563 | 0.83788 | 0.57889 | 0.46373 |
| recompute | 5 | off | 1334 | 0.05198 | 0.05818 | 0.75290 | 0.80043 | 0.54378 | 0.38225 |
| recompute | 5 | off | 1434 | 0.06603 | 0.07315 | 0.75910 | 0.80831 | 0.42895 | 0.29435 |
| recompute | 5 | on | 1234 | 0.06225 | 0.06678 | 0.77170 | 0.81405 | 0.55437 | 0.43039 |
| recompute | 5 | on | 1334 | 0.06548 | 0.07114 | 0.74957 | 0.79549 | 0.49249 | 0.29520 |
| recompute | 5 | on | 1434 | 0.08791 | 0.08767 | 0.79585 | 0.84135 | 0.54534 | 0.39412 |
| keep | 7 | – | 1234 | 0.06117 | 0.08987 | 0.79635 | 0.83893 | 0.58891 | 0.44595 |
| keep | 7 | – | 1334 | 0.03521 | 0.05096 | 0.74408 | 0.79442 | 0.52363 | 0.31576 |
| keep | 7 | – | 1434 | 0.03344 | 0.04244 | 0.72707 | 0.78127 | 0.53368 | 0.33745 |
| update | 7 | off | 1234 | 0.06478 | 0.09332 | 0.58723 | 0.68999 | 0.60606 | 0.23493 |
| update | 7 | off | 1334 | 0.05386 | 0.07961 | 0.78335 | 0.81414 | 0.58045 | 0.31903 |
| update | 7 | off | 1434 | 0.06393 | 0.06776 | 0.81482 | 0.84549 | 0.66946 | 0.44855 |
| update | 7 | on | 1234 | 0.06811 | 0.08878 | 0.77827 | 0.81347 | 0.62271 | 0.26936 |
| update | 7 | on | 1334 | 0.05113 | 0.07424 | 0.80150 | 0.83165 | 0.63835 | 0.39446 |
| update | 7 | on | 1434 | 0.06590 | 0.07706 | 0.79803 | 0.82831 | 0.58308 | 0.38472 |
| recompute | 7 | off | 1234 | 0.03509 | 0.06911 | 0.68050 | 0.72880 | 0.25539 | 0.17458 |
| recompute | 7 | off | 1334 | 0.02277 | 0.03731 | 0.69237 | 0.74782 | 0.42248 | 0.24519 |
| recompute | 7 | off | 1434 | 0.05648 | 0.06838 | 0.68752 | 0.75499 | 0.41856 | 0.27191 |
| recompute | 7 | on | 1234 | 0.03857 | 0.07299 | 0.63484 | 0.70013 | 0.17745 | 0.14000 |
| recompute | 7 | on | 1334 | 0.04139 | 0.05620 | 0.71708 | 0.77452 | 0.41893 | 0.24357 |
| recompute | 7 | on | 1434 | 0.04418 | 0.06189 | 0.68563 | 0.73931 | 0.28786 | 0.17904 |

_Bitcoin-otc avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.12724 | 0.11514 | 0.83933 | 0.88395 | 0.70327 | 0.58597 |
| keep | 3 | – | 1334 | 0.14774 | 0.11409 | 0.87733 | 0.90983 | 0.72924 | 0.55937 |
| keep | 3 | – | 1434 | 0.12656 | 0.10535 | 0.86804 | 0.89992 | 0.79753 | 0.65328 |
| update | 3 | off | 1234 | 0.12183 | 0.10801 | 0.84618 | 0.89045 | 0.73403 | 0.57551 |
| update | 3 | off | 1334 | 0.12568 | 0.09973 | 0.83536 | 0.87867 | 0.72587 | 0.53828 |
| update | 3 | off | 1434 | 0.14496 | 0.11939 | 0.87508 | 0.90918 | 0.74936 | 0.59795 |
| update | 3 | on | 1234 | 0.14849 | 0.11456 | 0.87978 | 0.91279 | 0.78150 | 0.67102 |
| update | 3 | on | 1334 | 0.13100 | 0.10318 | 0.83739 | 0.88256 | 0.72719 | 0.54542 |
| update | 3 | on | 1434 | 0.11411 | 0.09356 | 0.82988 | 0.87481 | 0.70436 | 0.54675 |
| recompute | 3 | off | 1234 | 0.10025 | 0.09334 | 0.87591 | 0.90284 | 0.66457 | 0.54446 |
| recompute | 3 | off | 1334 | 0.12070 | 0.10144 | 0.88254 | 0.91310 | 0.74661 | 0.60984 |
| recompute | 3 | off | 1434 | 0.14388 | 0.13020 | 0.88896 | 0.91279 | 0.74893 | 0.61897 |
| recompute | 3 | on | 1234 | 0.10682 | 0.09434 | 0.84182 | 0.87864 | 0.66221 | 0.54150 |
| recompute | 3 | on | 1334 | 0.12312 | 0.09958 | 0.86693 | 0.89947 | 0.70839 | 0.58970 |
| recompute | 3 | on | 1434 | 0.12225 | 0.11636 | 0.86586 | 0.89516 | 0.70913 | 0.58236 |
| keep | 5 | – | 1234 | 0.11712 | 0.11116 | 0.76209 | 0.82833 | 0.56423 | 0.42297 |
| keep | 5 | – | 1334 | 0.11647 | 0.09882 | 0.81250 | 0.86001 | 0.66334 | 0.50054 |
| keep | 5 | – | 1434 | 0.11257 | 0.10220 | 0.79882 | 0.85502 | 0.53146 | 0.32292 |
| update | 5 | off | 1234 | 0.08710 | 0.09366 | 0.69493 | 0.76278 | 0.61329 | 0.29724 |
| update | 5 | off | 1334 | 0.11753 | 0.09783 | 0.79947 | 0.85310 | 0.52758 | 0.37570 |
| update | 5 | off | 1434 | 0.09807 | 0.09009 | 0.75526 | 0.82600 | 0.58794 | 0.39176 |
| update | 5 | on | 1234 | 0.11496 | 0.11996 | 0.74393 | 0.81178 | 0.54599 | 0.39735 |
| update | 5 | on | 1334 | 0.12642 | 0.11060 | 0.78227 | 0.83875 | 0.61639 | 0.45770 |
| update | 5 | on | 1434 | 0.08498 | 0.08668 | 0.72200 | 0.79362 | 0.66744 | 0.33288 |
| recompute | 5 | off | 1234 | 0.07846 | 0.08445 | 0.75194 | 0.80525 | 0.44218 | 0.32831 |
| recompute | 5 | off | 1334 | 0.07697 | 0.09116 | 0.81482 | 0.85165 | 0.54341 | 0.43041 |
| recompute | 5 | off | 1434 | 0.06842 | 0.08111 | 0.75875 | 0.81087 | 0.59550 | 0.29318 |
| recompute | 5 | on | 1234 | 0.07608 | 0.10194 | 0.74640 | 0.79660 | 0.57139 | 0.32097 |
| recompute | 5 | on | 1334 | 0.05466 | 0.06512 | 0.78551 | 0.82913 | 0.63389 | 0.36862 |
| recompute | 5 | on | 1434 | 0.11266 | 0.10617 | 0.82905 | 0.87027 | 0.66643 | 0.41562 |
| keep | 7 | – | 1234 | 0.10707 | 0.13094 | 0.74319 | 0.80278 | 0.58320 | 0.37358 |
| keep | 7 | – | 1334 | 0.10303 | 0.08620 | 0.74499 | 0.80816 | 0.52930 | 0.25160 |
| keep | 7 | – | 1434 | 0.07484 | 0.09942 | 0.64051 | 0.72007 | 0.49012 | 0.15306 |
| update | 7 | off | 1234 | 0.09944 | 0.10437 | 0.71039 | 0.78374 | 0.50521 | 0.37438 |
| update | 7 | off | 1334 | 0.10301 | 0.09132 | 0.74045 | 0.80806 | 0.46574 | 0.33539 |
| update | 7 | off | 1434 | 0.05058 | 0.06940 | 0.57768 | 0.66754 | 0.51250 | 0.08638 |
| update | 7 | on | 1234 | 0.09401 | 0.10914 | 0.64578 | 0.73192 | 0.54793 | 0.25161 |
| update | 7 | on | 1334 | 0.07090 | 0.07671 | 0.62115 | 0.70927 | 0.55421 | 0.22172 |
| update | 7 | on | 1434 | 0.06036 | 0.08098 | 0.62883 | 0.69743 | 0.44653 | 0.14807 |
| recompute | 7 | off | 1234 | 0.06424 | 0.07938 | 0.74177 | 0.78907 | 0.39052 | 0.29833 |
| recompute | 7 | off | 1334 | 0.08365 | 0.09483 | 0.77052 | 0.81030 | 0.51748 | 0.33625 |
| recompute | 7 | off | 1434 | 0.03088 | 0.06144 | 0.66593 | 0.70923 | 0.32997 | 0.13608 |
| recompute | 7 | on | 1234 | 0.04813 | 0.06077 | 0.71744 | 0.76709 | 0.45935 | 0.32138 |
| recompute | 7 | on | 1334 | 0.04348 | 0.05897 | 0.71001 | 0.76146 | 0.51184 | 0.27529 |
| recompute | 7 | on | 1434 | 0.03033 | 0.05638 | 0.65999 | 0.70985 | 0.43209 | 0.13732 |

_AS-733 avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.31145 | 0.03829 | 0.93332 | 0.94536 | 0.82557 | 0.71078 |
| keep | 3 | – | 1334 | 0.32284 | 0.03212 | 0.93810 | 0.94895 | 0.84180 | 0.72942 |
| keep | 3 | – | 1434 | 0.31943 | 0.04097 | 0.94034 | 0.95012 | 0.83290 | 0.72072 |
| update | 3 | off | 1234 | 0.29608 | 0.04111 | 0.93239 | 0.94310 | 0.81581 | 0.69657 |
| update | 3 | off | 1334 | 0.30149 | 0.03840 | 0.93223 | 0.94343 | 0.82700 | 0.70862 |
| update | 3 | off | 1434 | 0.29177 | 0.04656 | 0.93338 | 0.94342 | 0.81603 | 0.69542 |
| update | 3 | on | 1234 | 0.31039 | 0.03255 | 0.93437 | 0.94593 | 0.82788 | 0.71351 |
| update | 3 | on | 1334 | 0.29849 | 0.04141 | 0.93173 | 0.94239 | 0.80489 | 0.68325 |
| update | 3 | on | 1434 | 0.29474 | 0.04652 | 0.93232 | 0.94306 | 0.81403 | 0.69265 |
| recompute | 3 | off | 1234 | 0.30616 | 0.04220 | 0.94367 | 0.95111 | 0.83197 | 0.72044 |
| recompute | 3 | off | 1334 | 0.30973 | 0.04652 | 0.94527 | 0.95234 | 0.84054 | 0.73041 |
| recompute | 3 | off | 1434 | 0.31154 | 0.04406 | 0.94609 | 0.95270 | 0.83164 | 0.71939 |
| recompute | 3 | on | 1234 | 0.35030 | 0.03166 | 0.95591 | 0.96087 | 0.85470 | 0.75239 |
| recompute | 3 | on | 1334 | 0.34706 | 0.03071 | 0.95573 | 0.96107 | 0.85848 | 0.75721 |
| recompute | 3 | on | 1434 | 0.36684 | 0.02716 | 0.95697 | 0.96179 | 0.85841 | 0.75604 |
| keep | 5 | – | 1234 | 0.30161 | 0.04489 | 0.90098 | 0.92136 | 0.72936 | 0.60587 |
| keep | 5 | – | 1334 | 0.26828 | 0.04388 | 0.89976 | 0.91791 | 0.69252 | 0.56962 |
| keep | 5 | – | 1434 | 0.28768 | 0.04272 | 0.90332 | 0.92210 | 0.71680 | 0.59343 |
| update | 5 | off | 1234 | 0.29118 | 0.05165 | 0.90168 | 0.92050 | 0.73126 | 0.60651 |
| update | 5 | off | 1334 | 0.26925 | 0.05122 | 0.90473 | 0.92054 | 0.67665 | 0.55462 |
| update | 5 | off | 1434 | 0.28156 | 0.05045 | 0.90140 | 0.92054 | 0.72050 | 0.59537 |
| update | 5 | on | 1234 | 0.29278 | 0.04486 | 0.90885 | 0.92483 | 0.72521 | 0.60190 |
| update | 5 | on | 1334 | 0.27565 | 0.04965 | 0.90394 | 0.91982 | 0.67232 | 0.55142 |
| update | 5 | on | 1434 | 0.27698 | 0.04873 | 0.90890 | 0.92466 | 0.69694 | 0.57565 |
| recompute | 5 | off | 1234 | 0.29337 | 0.05107 | 0.89869 | 0.91972 | 0.71342 | 0.59097 |
| recompute | 5 | off | 1334 | 0.27090 | 0.05192 | 0.90564 | 0.92113 | 0.66820 | 0.54881 |
| recompute | 5 | off | 1434 | 0.27675 | 0.04415 | 0.90251 | 0.92080 | 0.68850 | 0.56869 |
| recompute | 5 | on | 1234 | 0.32686 | 0.03231 | 0.93185 | 0.94195 | 0.75760 | 0.64023 |
| recompute | 5 | on | 1334 | 0.27786 | 0.05416 | 0.90694 | 0.92117 | 0.67139 | 0.55072 |
| recompute | 5 | on | 1434 | 0.30446 | 0.03687 | 0.92285 | 0.93515 | 0.73121 | 0.61212 |
| keep | 7 | – | 1234 | 0.28093 | 0.04775 | 0.85313 | 0.89107 | 0.64253 | 0.52607 |
| keep | 7 | – | 1334 | 0.27802 | 0.05295 | 0.86337 | 0.89574 | 0.62033 | 0.50838 |
| keep | 7 | – | 1434 | 0.26348 | 0.05345 | 0.87246 | 0.89911 | 0.62883 | 0.51303 |
| update | 7 | off | 1234 | 0.25827 | 0.05598 | 0.86602 | 0.89658 | 0.61400 | 0.50544 |
| update | 7 | off | 1334 | 0.28661 | 0.05431 | 0.87162 | 0.90066 | 0.64022 | 0.52479 |
| update | 7 | off | 1434 | 0.26340 | 0.05447 | 0.88318 | 0.90556 | 0.62549 | 0.50890 |
| update | 7 | on | 1234 | 0.28461 | 0.04731 | 0.88256 | 0.90636 | 0.65589 | 0.53591 |
| update | 7 | on | 1334 | 0.27791 | 0.05184 | 0.86669 | 0.89730 | 0.63458 | 0.52039 |
| update | 7 | on | 1434 | 0.27649 | 0.04864 | 0.88515 | 0.90648 | 0.63914 | 0.51746 |
| recompute | 7 | off | 1234 | 0.27280 | 0.05971 | 0.88330 | 0.90559 | 0.62277 | 0.50950 |
| recompute | 7 | off | 1334 | 0.27606 | 0.05673 | 0.85924 | 0.89254 | 0.62642 | 0.51357 |
| recompute | 7 | off | 1434 | 0.25790 | 0.06066 | 0.87297 | 0.89878 | 0.61706 | 0.50411 |
| recompute | 7 | on | 1234 | 0.30563 | 0.04341 | 0.90629 | 0.92150 | 0.68568 | 0.56799 |
| recompute | 7 | on | 1334 | 0.30597 | 0.04218 | 0.90018 | 0.91839 | 0.65947 | 0.54638 |
| recompute | 7 | on | 1434 | 0.27996 | 0.04787 | 0.87628 | 0.90239 | 0.62930 | 0.51262 |

_Reddit-title avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.41061 | 0.03654 | 0.97581 | 0.98033 | 0.90035 | 0.83109 |
| keep | 3 | – | 1334 | 0.41243 | 0.03651 | 0.97601 | 0.98040 | 0.89746 | 0.82740 |
| keep | 3 | – | 1434 | 0.41426 | 0.03743 | 0.97656 | 0.98098 | 0.90235 | 0.83474 |
| update | 3 | off | 1234 | 0.41305 | 0.03580 | 0.97593 | 0.98025 | 0.89846 | 0.82953 |
| update | 3 | off | 1334 | 0.41058 | 0.03952 | 0.97591 | 0.98026 | 0.89218 | 0.82077 |
| update | 3 | off | 1434 | 0.40992 | 0.04005 | 0.97623 | 0.98073 | 0.90228 | 0.83486 |
| update | 3 | on | 1234 | 0.40921 | 0.03979 | 0.97535 | 0.97997 | 0.89738 | 0.82757 |
| update | 3 | on | 1334 | 0.41443 | 0.03771 | 0.97573 | 0.98011 | 0.89387 | 0.82280 |
| update | 3 | on | 1434 | 0.41074 | 0.03914 | 0.97657 | 0.98086 | 0.90366 | 0.83662 |
| recompute | 3 | off | 1234 | 0.39276 | 0.04168 | 0.97672 | 0.98027 | 0.89181 | 0.81924 |
| recompute | 3 | off | 1334 | 0.39867 | 0.04927 | 0.97725 | 0.98099 | 0.89125 | 0.81868 |
| recompute | 3 | off | 1434 | 0.40343 | 0.04457 | 0.97782 | 0.98177 | 0.89586 | 0.82528 |
| recompute | 3 | on | 1234 | 0.39582 | 0.04658 | 0.97621 | 0.98028 | 0.89042 | 0.81780 |
| recompute | 3 | on | 1334 | 0.40292 | 0.05262 | 0.97697 | 0.98094 | 0.88571 | 0.81211 |
| recompute | 3 | on | 1434 | 0.39840 | 0.04975 | 0.97737 | 0.98140 | 0.89613 | 0.82584 |
| keep | 5 | – | 1234 | 0.39318 | 0.03712 | 0.97299 | 0.97764 | 0.87255 | 0.79406 |
| keep | 5 | – | 1334 | 0.39324 | 0.04139 | 0.97448 | 0.97888 | 0.86512 | 0.78838 |
| keep | 5 | – | 1434 | 0.39751 | 0.03838 | 0.97505 | 0.97938 | 0.88103 | 0.80681 |
| update | 5 | off | 1234 | 0.39976 | 0.03774 | 0.97284 | 0.97755 | 0.87332 | 0.79477 |
| update | 5 | off | 1334 | 0.39952 | 0.04507 | 0.97432 | 0.97875 | 0.86225 | 0.78747 |
| update | 5 | off | 1434 | 0.39904 | 0.03928 | 0.97436 | 0.97871 | 0.88376 | 0.80976 |
| update | 5 | on | 1234 | 0.39744 | 0.03753 | 0.97333 | 0.97811 | 0.86789 | 0.78860 |
| update | 5 | on | 1334 | 0.39532 | 0.04278 | 0.97443 | 0.97866 | 0.86096 | 0.78432 |
| update | 5 | on | 1434 | 0.39153 | 0.03629 | 0.97453 | 0.97905 | 0.88489 | 0.81129 |
| recompute | 5 | off | 1234 | 0.39060 | 0.04012 | 0.97452 | 0.97871 | 0.85947 | 0.77769 |
| recompute | 5 | off | 1334 | 0.38715 | 0.04461 | 0.97552 | 0.97980 | 0.84919 | 0.76954 |
| recompute | 5 | off | 1434 | 0.38855 | 0.04150 | 0.97508 | 0.97949 | 0.87068 | 0.79250 |
| recompute | 5 | on | 1234 | 0.39124 | 0.04411 | 0.97429 | 0.97853 | 0.85492 | 0.77308 |
| recompute | 5 | on | 1334 | 0.38139 | 0.04604 | 0.97501 | 0.97915 | 0.85019 | 0.76966 |
| recompute | 5 | on | 1434 | 0.39010 | 0.04614 | 0.97519 | 0.97960 | 0.87485 | 0.79720 |
| keep | 7 | – | 1234 | 0.39148 | 0.04595 | 0.96952 | 0.97489 | 0.82982 | 0.74545 |
| keep | 7 | – | 1334 | 0.38530 | 0.03925 | 0.97160 | 0.97631 | 0.80961 | 0.72482 |
| keep | 7 | – | 1434 | 0.38482 | 0.04271 | 0.97162 | 0.97653 | 0.83360 | 0.74932 |
| update | 7 | off | 1234 | 0.39122 | 0.04393 | 0.96898 | 0.97439 | 0.81589 | 0.72919 |
| update | 7 | off | 1334 | 0.38907 | 0.04156 | 0.97150 | 0.97637 | 0.81654 | 0.73397 |
| update | 7 | off | 1434 | 0.38634 | 0.04074 | 0.97157 | 0.97641 | 0.83481 | 0.75031 |
| update | 7 | on | 1234 | 0.39369 | 0.04633 | 0.96852 | 0.97418 | 0.82302 | 0.73846 |
| update | 7 | on | 1334 | 0.39087 | 0.04031 | 0.97159 | 0.97648 | 0.80061 | 0.71624 |
| update | 7 | on | 1434 | 0.38936 | 0.04021 | 0.97155 | 0.97627 | 0.83807 | 0.75379 |
| recompute | 7 | off | 1234 | 0.38879 | 0.04450 | 0.97086 | 0.97574 | 0.80163 | 0.71536 |
| recompute | 7 | off | 1334 | 0.38876 | 0.04133 | 0.97315 | 0.97754 | 0.78525 | 0.70031 |
| recompute | 7 | off | 1434 | 0.38642 | 0.04076 | 0.97345 | 0.97777 | 0.81790 | 0.73077 |
| recompute | 7 | on | 1234 | 0.38321 | 0.04688 | 0.97009 | 0.97508 | 0.80050 | 0.71337 |
| recompute | 7 | on | 1334 | 0.38146 | 0.04301 | 0.97368 | 0.97798 | 0.78049 | 0.69555 |
| recompute | 7 | on | 1434 | 0.38322 | 0.04590 | 0.97453 | 0.97873 | 0.81789 | 0.73236 |

_Reddit-body avg (45 runs):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| keep | 3 | – | 1234 | 0.33225 | 0.05240 | 0.96078 | 0.96855 | 0.89214 | 0.80358 |
| keep | 3 | – | 1334 | 0.32798 | 0.04851 | 0.96118 | 0.96812 | 0.89027 | 0.80521 |
| keep | 3 | – | 1434 | 0.32895 | 0.04579 | 0.96268 | 0.96952 | 0.88958 | 0.80581 |
| update | 3 | off | 1234 | 0.33125 | 0.05139 | 0.95985 | 0.96733 | 0.89101 | 0.80130 |
| update | 3 | off | 1334 | 0.33478 | 0.05054 | 0.96113 | 0.96835 | 0.88987 | 0.80414 |
| update | 3 | off | 1434 | 0.33481 | 0.04521 | 0.96181 | 0.96895 | 0.88730 | 0.80181 |
| update | 3 | on | 1234 | 0.33108 | 0.04866 | 0.96035 | 0.96786 | 0.89170 | 0.80143 |
| update | 3 | on | 1334 | 0.34258 | 0.04921 | 0.96201 | 0.96915 | 0.88952 | 0.80380 |
| update | 3 | on | 1434 | 0.34476 | 0.04822 | 0.96271 | 0.96974 | 0.89073 | 0.80763 |
| recompute | 3 | off | 1234 | 0.31310 | 0.05093 | 0.96264 | 0.96902 | 0.88943 | 0.79869 |
| recompute | 3 | off | 1334 | 0.31679 | 0.05192 | 0.96442 | 0.97041 | 0.88879 | 0.80469 |
| recompute | 3 | off | 1434 | 0.32140 | 0.04427 | 0.96331 | 0.96990 | 0.88797 | 0.80360 |
| recompute | 3 | on | 1234 | 0.32016 | 0.05036 | 0.96277 | 0.96948 | 0.88758 | 0.79758 |
| recompute | 3 | on | 1334 | 0.32638 | 0.05254 | 0.96298 | 0.96950 | 0.88499 | 0.79795 |
| recompute | 3 | on | 1434 | 0.32178 | 0.04423 | 0.96381 | 0.97032 | 0.88936 | 0.80568 |
| keep | 5 | – | 1234 | 0.30340 | 0.04662 | 0.95525 | 0.96347 | 0.85539 | 0.75634 |
| keep | 5 | – | 1334 | 0.30138 | 0.04121 | 0.95824 | 0.96575 | 0.85818 | 0.76525 |
| keep | 5 | – | 1434 | 0.29005 | 0.04055 | 0.95477 | 0.96307 | 0.85132 | 0.75749 |
| update | 5 | off | 1234 | 0.31256 | 0.04851 | 0.95591 | 0.96401 | 0.85345 | 0.75552 |
| update | 5 | off | 1334 | 0.31189 | 0.04129 | 0.95728 | 0.96480 | 0.85229 | 0.75850 |
| update | 5 | off | 1434 | 0.30257 | 0.04027 | 0.95691 | 0.96453 | 0.85356 | 0.76209 |
| update | 5 | on | 1234 | 0.30894 | 0.04699 | 0.95532 | 0.96322 | 0.85034 | 0.75117 |
| update | 5 | on | 1334 | 0.30356 | 0.03979 | 0.95697 | 0.96485 | 0.85634 | 0.76295 |
| update | 5 | on | 1434 | 0.29100 | 0.04144 | 0.95669 | 0.96443 | 0.85619 | 0.76536 |
| recompute | 5 | off | 1234 | 0.29238 | 0.04390 | 0.95791 | 0.96511 | 0.85159 | 0.75326 |
| recompute | 5 | off | 1334 | 0.29239 | 0.04457 | 0.95967 | 0.96655 | 0.84784 | 0.75477 |
| recompute | 5 | off | 1434 | 0.29063 | 0.04533 | 0.95744 | 0.96491 | 0.84540 | 0.75379 |
| recompute | 5 | on | 1234 | 0.29535 | 0.04589 | 0.95729 | 0.96464 | 0.84781 | 0.74814 |
| recompute | 5 | on | 1334 | 0.29642 | 0.04388 | 0.95911 | 0.96606 | 0.84614 | 0.75126 |
| recompute | 5 | on | 1434 | 0.29488 | 0.04111 | 0.95664 | 0.96426 | 0.84843 | 0.75513 |
| keep | 7 | – | 1234 | 0.28809 | 0.03934 | 0.94856 | 0.95813 | 0.78324 | 0.67904 |
| keep | 7 | – | 1334 | 0.28944 | 0.04230 | 0.95421 | 0.96267 | 0.79471 | 0.69765 |
| keep | 7 | – | 1434 | 0.28536 | 0.03961 | 0.95186 | 0.96050 | 0.79070 | 0.69294 |
| update | 7 | off | 1234 | 0.29221 | 0.04008 | 0.95092 | 0.95933 | 0.79469 | 0.69066 |
| update | 7 | off | 1334 | 0.27630 | 0.03888 | 0.95168 | 0.96005 | 0.78639 | 0.68743 |
| update | 7 | off | 1434 | 0.28425 | 0.04288 | 0.95117 | 0.96007 | 0.78095 | 0.68338 |
| update | 7 | on | 1234 | 0.28042 | 0.04055 | 0.94964 | 0.95888 | 0.78353 | 0.67946 |
| update | 7 | on | 1334 | 0.28657 | 0.03848 | 0.95308 | 0.96163 | 0.79322 | 0.69610 |
| update | 7 | on | 1434 | 0.28748 | 0.04092 | 0.95054 | 0.95965 | 0.77604 | 0.67822 |
| recompute | 7 | off | 1234 | 0.27326 | 0.04180 | 0.95206 | 0.96028 | 0.78272 | 0.67888 |
| recompute | 7 | off | 1334 | 0.27811 | 0.04419 | 0.95648 | 0.96356 | 0.78279 | 0.68575 |
| recompute | 7 | off | 1434 | 0.27424 | 0.04229 | 0.95425 | 0.96181 | 0.77254 | 0.67652 |
| recompute | 7 | on | 1234 | 0.27305 | 0.04053 | 0.95180 | 0.95963 | 0.78449 | 0.67973 |
| recompute | 7 | on | 1334 | 0.27241 | 0.04291 | 0.95517 | 0.96244 | 0.78964 | 0.69086 |
| recompute | 7 | on | 1434 | 0.27823 | 0.04141 | 0.95512 | 0.96295 | 0.79064 | 0.69447 |


## 7. Coarse-snapshot experiments (4× duration = 28-day windows) — SEPARATE; NOT comparable to §2–§6

> **DIFFERENT TASK — do not compare these numbers to the weekly tables above.** These runs use `dataset.snapshot_freq=2419200s` (28-day windows, **4× the weekly base**), so each run has far **fewer snapshots** (UCI **~7** vs 27; reddit_body **~44** vs 176) and predicts a **4-week-ahead** graph — a strictly *harder* forecast, so absolute MRR is systematically **lower** and is its own comparison set. wandb tag **`coarse-snap`** / **`coarse-28d`**, group suffix `freq-2419200s`. Full matrix incl **C=1 (centralized)**: {feature,keep,update,recompute} × proc{on,off}(upd/rec) × C{1,3,5,7} × 3 seeds = 72/dataset. gru, sfv_share=local.

**3-seed mean MRR (C=1 = centralized ceiling):**

_UCI (snapshots=7/run):_
| mode | C1 (centralized) | C3 | C5 | C7 |
|---|---|---|---|---|
| feature | 0.127 | 0.089 | 0.078 | 0.069 |
| keep | 0.092 | 0.080 | 0.077 | 0.084 |
| update proc-off | 0.080 | 0.071 | 0.069 | 0.060 |
| update proc-on | 0.100 | 0.081 | 0.074 | 0.075 |
| recompute proc-off | 0.068 | 0.069 | 0.058 | 0.064 |
| recompute proc-on | 0.075 | 0.063 | 0.068 | 0.073 |

_Reddit-body (snapshots=44/run):_
| mode | C1 (centralized) | C3 | C5 | C7 |
|---|---|---|---|---|
| feature | 0.384 | 0.328 | 0.293 | 0.273 |
| keep | 0.378 | 0.318 | 0.294 | 0.276 |
| update proc-off | 0.376 | 0.324 | 0.293 | 0.277 |
| update proc-on | 0.381 | 0.320 | 0.292 | 0.278 |
| recompute proc-off | 0.354 | 0.289 | 0.276 | 0.261 |
| recompute proc-on | 0.351 | 0.295 | 0.274 | 0.262 |

**Read — the story REPLICATES at the coarse window.** Despite ~4× fewer snapshots and the harder
4-week-ahead forecast (absolute MRR ~0.01–0.02 below the weekly tables), the qualitative pattern is
unchanged: `feature` declines with C while `keep`/`update` hold and **catch or overtake it by C7**
(UCI keep **0.084** > feature **0.069** at C7; reddit_body keep≈update≈feature ≈**0.276** at C7), and
`recompute` is worst at every C. The **C=1 centralized** column is the ceiling (feature UCI 0.127,
reddit_body 0.384) and every federated C sits below it, as expected. So the §4 headline findings are
**robust to the forecast horizon**, not an artifact of the weekly window. (UCI at 4× is only ~7
snapshots → noisy; reddit_body ~44 is cleaner.)

### 7.1 HIGH-C extension — C{10,15,20} on the coarse window  **[confirmed, harvested 2026-07-20]**
108 runs (`runs/reddit_{body,title}_coarse_highC`, run 2026-07-14, harvested into this record
2026-07-20). Same 28-day coarse window as the tables above (44 snapshots), so these extend the C
axis of §7 and are **not** comparable to the weekly §2–§6 numbers. 3-seed mean MRR / AUC.

_Reddit-body coarse, high C:_
| mode | C10 | C15 | C20 |
|---|---|---|---|
| feature | 0.257 / 0.943 | 0.254 / 0.928 | 0.243 / 0.927 |
| keep | 0.257 / 0.941 | 0.246 / 0.929 | 0.242 / 0.927 |
| update proc-off | 0.261 / 0.942 | 0.246 / 0.929 | 0.240 / 0.927 |
| update proc-on | 0.258 / 0.942 | 0.250 / 0.929 | 0.240 / 0.926 |
| recompute proc-off | 0.250 / 0.943 | 0.242 / 0.929 | 0.234 / 0.929 |
| recompute proc-on | 0.251 / 0.942 | 0.241 / 0.929 | 0.235 / 0.928 |

_Reddit-title coarse, high C:_
| mode | C10 | C15 | C20 |
|---|---|---|---|
| feature | 0.364 / 0.961 | 0.348 / 0.961 | 0.357 / 0.958 |
| keep | 0.363 / 0.966 | 0.358 / 0.965 | 0.349 / 0.963 |
| update proc-off | 0.362 / 0.966 | 0.351 / 0.965 | 0.354 / 0.962 |
| update proc-on | 0.362 / 0.965 | 0.356 / 0.965 | 0.351 / 0.962 |
| recompute proc-off | 0.351 / 0.967 | 0.345 / 0.967 | 0.351 / 0.964 |
| recompute proc-on | 0.363 / 0.967 | 0.353 / 0.967 | 0.353 / 0.964 |

**Read — degradation FLATTENS at high C, and the modes converge.** Extending C3→C7 (above) to
C10→C20 shows the decline decelerating rather than continuing: reddit_body feature drops 0.273
(C7) → 0.257 (C10) → 0.243 (C20), and reddit_title is nearly flat (0.364 → 0.357 over C10→C20,
non-monotone and inside noise). This supports the paper's **resilience** framing: even at 20-way
sharding the model retains most of its C3 accuracy on these graphs.

**But it also removes the crossover.** At high C every mode is within ~0.01 of `feature` — on
reddit_body all six rows sit in 0.234–0.261 at C10 and 0.234–0.243 at C20. The §4.1 "spectral
rescues federated" crossover does **not** widen with more clients; the modes simply converge.
That is consistent with §10 (the spectral branch does not transport structure) and should be stated
as such: the high-C data is evidence for *resilience of the backbone*, **not** for a spectral benefit.
`recompute` remains nominally worst at nearly every cell, and procrustes is neutral-to-mildly-
positive for recompute on reddit_title (+0.008/+0.008/+0.002) — both consistent with §6.

**Per-run coarse tables (record, all metrics):**

_UCI coarse (72 runs, snapshots=7):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 1 | – | 1234 | 0.10862 | 0.06126 | 0.79251 | 0.81461 | 0.71836 | 0.45666 |
| feature | 1 | – | 1334 | 0.14615 | 0.08893 | 0.80522 | 0.82819 | 0.74709 | 0.47890 |
| feature | 1 | – | 1434 | 0.12534 | 0.08086 | 0.84719 | 0.85617 | 0.74837 | 0.55658 |
| keep | 1 | – | 1234 | 0.10224 | 0.03574 | 0.85848 | 0.86690 | 0.73128 | 0.54420 |
| keep | 1 | – | 1334 | 0.07896 | 0.03297 | 0.85082 | 0.85913 | 0.81275 | 0.58251 |
| keep | 1 | – | 1434 | 0.09393 | 0.03271 | 0.87118 | 0.87386 | 0.78637 | 0.62850 |
| update | 1 | off | 1234 | 0.09072 | 0.04714 | 0.80736 | 0.82638 | 0.71683 | 0.51719 |
| update | 1 | off | 1334 | 0.08190 | 0.04107 | 0.79826 | 0.80702 | 0.71135 | 0.46851 |
| update | 1 | off | 1434 | 0.06706 | 0.03939 | 0.78911 | 0.80633 | 0.67684 | 0.48743 |
| update | 1 | on | 1234 | 0.09991 | 0.04565 | 0.83665 | 0.84866 | 0.72288 | 0.51013 |
| update | 1 | on | 1334 | 0.08313 | 0.03778 | 0.83434 | 0.84440 | 0.78341 | 0.54891 |
| update | 1 | on | 1434 | 0.11657 | 0.07230 | 0.86758 | 0.87025 | 0.75999 | 0.59284 |
| recompute | 1 | off | 1234 | 0.07333 | 0.04654 | 0.80829 | 0.81295 | 0.67058 | 0.46435 |
| recompute | 1 | off | 1334 | 0.07040 | 0.05041 | 0.79812 | 0.79764 | 0.70923 | 0.46349 |
| recompute | 1 | off | 1434 | 0.06006 | 0.01912 | 0.81635 | 0.81500 | 0.66192 | 0.47919 |
| recompute | 1 | on | 1234 | 0.06974 | 0.03671 | 0.78782 | 0.79522 | 0.63120 | 0.43266 |
| recompute | 1 | on | 1334 | 0.07533 | 0.04030 | 0.79407 | 0.79832 | 0.67707 | 0.40328 |
| recompute | 1 | on | 1434 | 0.08040 | 0.04404 | 0.82133 | 0.82539 | 0.63605 | 0.44919 |
| feature | 3 | – | 1234 | 0.08298 | 0.03939 | 0.78020 | 0.79797 | 0.62831 | 0.38674 |
| feature | 3 | – | 1334 | 0.10900 | 0.06170 | 0.81743 | 0.83179 | 0.65255 | 0.44608 |
| feature | 3 | – | 1434 | 0.07355 | 0.04124 | 0.83687 | 0.84141 | 0.70697 | 0.52037 |
| keep | 3 | – | 1234 | 0.09124 | 0.05081 | 0.82630 | 0.83529 | 0.66315 | 0.49837 |
| keep | 3 | – | 1334 | 0.07426 | 0.04411 | 0.79567 | 0.81414 | 0.69325 | 0.44190 |
| keep | 3 | – | 1434 | 0.07409 | 0.03029 | 0.83242 | 0.84011 | 0.66059 | 0.51515 |
| update | 3 | off | 1234 | 0.06982 | 0.03707 | 0.76760 | 0.78245 | 0.59517 | 0.42897 |
| update | 3 | off | 1334 | 0.07454 | 0.04857 | 0.76707 | 0.79007 | 0.61966 | 0.39633 |
| update | 3 | off | 1434 | 0.06966 | 0.02731 | 0.76369 | 0.78772 | 0.52247 | 0.36847 |
| update | 3 | on | 1234 | 0.08187 | 0.03104 | 0.80800 | 0.81964 | 0.68652 | 0.50666 |
| update | 3 | on | 1334 | 0.07563 | 0.04438 | 0.79922 | 0.81286 | 0.70665 | 0.46025 |
| update | 3 | on | 1434 | 0.08425 | 0.03291 | 0.80236 | 0.81482 | 0.62605 | 0.47281 |
| recompute | 3 | off | 1234 | 0.07929 | 0.03551 | 0.78748 | 0.80373 | 0.51961 | 0.40455 |
| recompute | 3 | off | 1334 | 0.05791 | 0.02727 | 0.77646 | 0.79722 | 0.59222 | 0.38861 |
| recompute | 3 | off | 1434 | 0.06908 | 0.03511 | 0.77212 | 0.78572 | 0.42858 | 0.34645 |
| recompute | 3 | on | 1234 | 0.06737 | 0.04189 | 0.76468 | 0.77216 | 0.48494 | 0.34252 |
| recompute | 3 | on | 1334 | 0.06377 | 0.03133 | 0.74538 | 0.77640 | 0.59918 | 0.37479 |
| recompute | 3 | on | 1434 | 0.05851 | 0.03047 | 0.79279 | 0.79986 | 0.48312 | 0.38077 |
| feature | 5 | – | 1234 | 0.06755 | 0.04808 | 0.74884 | 0.76288 | 0.55746 | 0.29754 |
| feature | 5 | – | 1334 | 0.06580 | 0.04783 | 0.73435 | 0.77227 | 0.59888 | 0.36732 |
| feature | 5 | – | 1434 | 0.10154 | 0.04275 | 0.81275 | 0.82846 | 0.67403 | 0.47365 |
| keep | 5 | – | 1234 | 0.09137 | 0.02953 | 0.74869 | 0.77742 | 0.43727 | 0.32191 |
| keep | 5 | – | 1334 | 0.06069 | 0.02967 | 0.78124 | 0.80117 | 0.63429 | 0.39905 |
| keep | 5 | – | 1434 | 0.07961 | 0.03107 | 0.81120 | 0.81349 | 0.48706 | 0.36519 |
| update | 5 | off | 1234 | 0.07217 | 0.02032 | 0.77514 | 0.80350 | 0.43275 | 0.35480 |
| update | 5 | off | 1334 | 0.04788 | 0.01597 | 0.72480 | 0.75518 | 0.53367 | 0.32997 |
| update | 5 | off | 1434 | 0.08749 | 0.03381 | 0.78669 | 0.80765 | 0.52389 | 0.39714 |
| update | 5 | on | 1234 | 0.07113 | 0.01761 | 0.77264 | 0.79778 | 0.52703 | 0.40811 |
| update | 5 | on | 1334 | 0.06193 | 0.03393 | 0.77884 | 0.80172 | 0.65632 | 0.44308 |
| update | 5 | on | 1434 | 0.08782 | 0.03231 | 0.81336 | 0.82462 | 0.56505 | 0.42937 |
| recompute | 5 | off | 1234 | 0.07648 | 0.02395 | 0.71001 | 0.73886 | 0.31834 | 0.27524 |
| recompute | 5 | off | 1334 | 0.04245 | 0.01459 | 0.73336 | 0.76102 | 0.39293 | 0.25354 |
| recompute | 5 | off | 1434 | 0.05578 | 0.02331 | 0.75603 | 0.77264 | 0.29128 | 0.23475 |
| recompute | 5 | on | 1234 | 0.07262 | 0.02791 | 0.71793 | 0.74719 | 0.35456 | 0.28913 |
| recompute | 5 | on | 1334 | 0.05231 | 0.03921 | 0.71856 | 0.73395 | 0.42425 | 0.25452 |
| recompute | 5 | on | 1434 | 0.07852 | 0.02448 | 0.78276 | 0.79382 | 0.38901 | 0.30949 |
| feature | 7 | – | 1234 | 0.07286 | 0.02888 | 0.76261 | 0.78842 | 0.46780 | 0.30490 |
| feature | 7 | – | 1334 | 0.06298 | 0.01954 | 0.76461 | 0.78680 | 0.58258 | 0.38915 |
| feature | 7 | – | 1434 | 0.07020 | 0.04195 | 0.80090 | 0.81260 | 0.56483 | 0.41849 |
| keep | 7 | – | 1234 | 0.09817 | 0.06072 | 0.75212 | 0.77834 | 0.45030 | 0.36921 |
| keep | 7 | – | 1334 | 0.08342 | 0.02514 | 0.77362 | 0.79902 | 0.42231 | 0.29806 |
| keep | 7 | – | 1434 | 0.06915 | 0.03703 | 0.78486 | 0.80203 | 0.44603 | 0.36683 |
| update | 7 | off | 1234 | 0.07100 | 0.03407 | 0.73994 | 0.75891 | 0.44255 | 0.34912 |
| update | 7 | off | 1334 | 0.05112 | 0.02200 | 0.75766 | 0.78864 | 0.33172 | 0.24051 |
| update | 7 | off | 1434 | 0.05805 | 0.02472 | 0.76115 | 0.78191 | 0.43732 | 0.33336 |
| update | 7 | on | 1234 | 0.09127 | 0.04502 | 0.77579 | 0.79579 | 0.45273 | 0.36956 |
| update | 7 | on | 1334 | 0.06327 | 0.01877 | 0.77118 | 0.79564 | 0.42084 | 0.28048 |
| update | 7 | on | 1434 | 0.07138 | 0.03577 | 0.78554 | 0.80498 | 0.44915 | 0.36268 |
| recompute | 7 | off | 1234 | 0.08238 | 0.05855 | 0.72308 | 0.75476 | 0.29311 | 0.26300 |
| recompute | 7 | off | 1334 | 0.05209 | 0.02224 | 0.70100 | 0.72752 | 0.21485 | 0.15355 |
| recompute | 7 | off | 1434 | 0.05659 | 0.02798 | 0.74666 | 0.78285 | 0.33330 | 0.29748 |
| recompute | 7 | on | 1234 | 0.09179 | 0.05583 | 0.69370 | 0.73718 | 0.24448 | 0.22873 |
| recompute | 7 | on | 1334 | 0.06811 | 0.02979 | 0.71143 | 0.74862 | 0.29521 | 0.21803 |
| recompute | 7 | on | 1434 | 0.05860 | 0.02728 | 0.74043 | 0.77512 | 0.38205 | 0.32726 |

_Reddit-body coarse (72 runs, snapshots=44):_
| mode | C | proc | seed | mean_mrr | std | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|---|---|---|
| feature | 1 | – | 1234 | 0.37284 | 0.06858 | 0.96912 | 0.97464 | 0.91350 | 0.84016 |
| feature | 1 | – | 1334 | 0.40063 | 0.07522 | 0.95929 | 0.96748 | 0.91200 | 0.82552 |
| feature | 1 | – | 1434 | 0.37766 | 0.07552 | 0.96488 | 0.97192 | 0.91649 | 0.82608 |
| keep | 1 | – | 1234 | 0.38557 | 0.06299 | 0.96150 | 0.96937 | 0.91616 | 0.82737 |
| keep | 1 | – | 1334 | 0.37819 | 0.07037 | 0.96666 | 0.97310 | 0.90980 | 0.83730 |
| keep | 1 | – | 1434 | 0.37087 | 0.07184 | 0.96213 | 0.97014 | 0.90229 | 0.83001 |
| update | 1 | off | 1234 | 0.37589 | 0.06742 | 0.96036 | 0.96817 | 0.91309 | 0.82157 |
| update | 1 | off | 1334 | 0.37600 | 0.07041 | 0.96567 | 0.97203 | 0.90871 | 0.83478 |
| update | 1 | off | 1434 | 0.37530 | 0.07696 | 0.96246 | 0.97041 | 0.90322 | 0.83263 |
| update | 1 | on | 1234 | 0.37792 | 0.06555 | 0.96159 | 0.96932 | 0.91649 | 0.82812 |
| update | 1 | on | 1334 | 0.38911 | 0.07685 | 0.96564 | 0.97181 | 0.90790 | 0.83330 |
| update | 1 | on | 1434 | 0.37602 | 0.07193 | 0.96200 | 0.97012 | 0.90196 | 0.82951 |
| recompute | 1 | off | 1234 | 0.34576 | 0.06393 | 0.95987 | 0.96693 | 0.90840 | 0.81425 |
| recompute | 1 | off | 1334 | 0.35294 | 0.07487 | 0.96673 | 0.97215 | 0.90645 | 0.83272 |
| recompute | 1 | off | 1434 | 0.36394 | 0.07153 | 0.96217 | 0.96996 | 0.89983 | 0.82801 |
| recompute | 1 | on | 1234 | 0.33113 | 0.06179 | 0.95956 | 0.96698 | 0.90655 | 0.81094 |
| recompute | 1 | on | 1334 | 0.36471 | 0.07611 | 0.96447 | 0.97057 | 0.90313 | 0.82629 |
| recompute | 1 | on | 1434 | 0.35715 | 0.07134 | 0.95974 | 0.96785 | 0.89572 | 0.81984 |
| feature | 3 | – | 1234 | 0.32389 | 0.05696 | 0.96007 | 0.96700 | 0.88217 | 0.79632 |
| feature | 3 | – | 1334 | 0.32885 | 0.04548 | 0.95099 | 0.96046 | 0.87874 | 0.77689 |
| feature | 3 | – | 1434 | 0.33195 | 0.05337 | 0.95756 | 0.96616 | 0.88715 | 0.78720 |
| keep | 3 | – | 1234 | 0.31185 | 0.04368 | 0.95190 | 0.96081 | 0.88504 | 0.78069 |
| keep | 3 | – | 1334 | 0.33285 | 0.05891 | 0.95978 | 0.96727 | 0.88148 | 0.79921 |
| keep | 3 | – | 1434 | 0.31074 | 0.04825 | 0.95366 | 0.96349 | 0.87480 | 0.79260 |
| update | 3 | off | 1234 | 0.32427 | 0.04687 | 0.95253 | 0.96183 | 0.88697 | 0.78436 |
| update | 3 | off | 1334 | 0.33013 | 0.06101 | 0.95974 | 0.96700 | 0.88597 | 0.80375 |
| update | 3 | off | 1434 | 0.31643 | 0.04820 | 0.95354 | 0.96324 | 0.87738 | 0.79623 |
| update | 3 | on | 1234 | 0.31999 | 0.04590 | 0.95298 | 0.96199 | 0.88660 | 0.78451 |
| update | 3 | on | 1334 | 0.32321 | 0.06008 | 0.95982 | 0.96697 | 0.88262 | 0.79992 |
| update | 3 | on | 1434 | 0.31784 | 0.05068 | 0.95397 | 0.96379 | 0.87658 | 0.79471 |
| recompute | 3 | off | 1234 | 0.28918 | 0.04614 | 0.95270 | 0.96119 | 0.87969 | 0.77449 |
| recompute | 3 | off | 1334 | 0.29396 | 0.06280 | 0.95857 | 0.96532 | 0.87274 | 0.78628 |
| recompute | 3 | off | 1434 | 0.28514 | 0.04269 | 0.95301 | 0.96235 | 0.86423 | 0.77816 |
| recompute | 3 | on | 1234 | 0.29154 | 0.04937 | 0.95244 | 0.96150 | 0.88163 | 0.77733 |
| recompute | 3 | on | 1334 | 0.30193 | 0.06115 | 0.95808 | 0.96549 | 0.87040 | 0.78382 |
| recompute | 3 | on | 1434 | 0.29160 | 0.04730 | 0.95288 | 0.96238 | 0.86450 | 0.77821 |
| feature | 5 | – | 1234 | 0.29301 | 0.05242 | 0.95502 | 0.96260 | 0.85080 | 0.75889 |
| feature | 5 | – | 1334 | 0.28191 | 0.04589 | 0.94605 | 0.95587 | 0.84701 | 0.73801 |
| feature | 5 | – | 1434 | 0.30289 | 0.05002 | 0.95201 | 0.96093 | 0.85503 | 0.74507 |
| keep | 5 | – | 1234 | 0.29829 | 0.03886 | 0.94883 | 0.95862 | 0.86163 | 0.75209 |
| keep | 5 | – | 1334 | 0.29714 | 0.05227 | 0.95471 | 0.96241 | 0.84876 | 0.75722 |
| keep | 5 | – | 1434 | 0.28619 | 0.04160 | 0.94812 | 0.95824 | 0.84376 | 0.75349 |
| update | 5 | off | 1234 | 0.29944 | 0.03972 | 0.94855 | 0.95836 | 0.85707 | 0.74830 |
| update | 5 | off | 1334 | 0.29091 | 0.05177 | 0.95500 | 0.96301 | 0.84881 | 0.75877 |
| update | 5 | off | 1434 | 0.28763 | 0.04107 | 0.94889 | 0.95888 | 0.84535 | 0.75576 |
| update | 5 | on | 1234 | 0.29402 | 0.03996 | 0.94872 | 0.95859 | 0.85929 | 0.74997 |
| update | 5 | on | 1334 | 0.29316 | 0.05421 | 0.95438 | 0.96229 | 0.84781 | 0.75663 |
| update | 5 | on | 1434 | 0.28943 | 0.04185 | 0.94848 | 0.95853 | 0.84633 | 0.75596 |
| recompute | 5 | off | 1234 | 0.28343 | 0.03853 | 0.94864 | 0.95811 | 0.84662 | 0.73366 |
| recompute | 5 | off | 1334 | 0.27493 | 0.05554 | 0.95510 | 0.96214 | 0.83410 | 0.73936 |
| recompute | 5 | off | 1434 | 0.26941 | 0.04493 | 0.94780 | 0.95726 | 0.83295 | 0.74034 |
| recompute | 5 | on | 1234 | 0.28064 | 0.03793 | 0.94873 | 0.95842 | 0.84983 | 0.73763 |
| recompute | 5 | on | 1334 | 0.26787 | 0.05266 | 0.95443 | 0.96154 | 0.83580 | 0.74115 |
| recompute | 5 | on | 1434 | 0.27466 | 0.04308 | 0.94786 | 0.95792 | 0.83574 | 0.74317 |
| feature | 7 | – | 1234 | 0.27395 | 0.05034 | 0.95088 | 0.95875 | 0.82667 | 0.72938 |
| feature | 7 | – | 1334 | 0.26967 | 0.03239 | 0.94238 | 0.95316 | 0.82472 | 0.71307 |
| feature | 7 | – | 1434 | 0.27561 | 0.03430 | 0.95065 | 0.95958 | 0.84360 | 0.73115 |
| keep | 7 | – | 1234 | 0.27560 | 0.03744 | 0.94306 | 0.95355 | 0.84157 | 0.72686 |
| keep | 7 | – | 1334 | 0.27385 | 0.04637 | 0.95225 | 0.96006 | 0.82972 | 0.73415 |
| keep | 7 | – | 1434 | 0.27832 | 0.03926 | 0.94492 | 0.95601 | 0.81288 | 0.71962 |
| update | 7 | off | 1234 | 0.27532 | 0.04075 | 0.94255 | 0.95306 | 0.83866 | 0.72261 |
| update | 7 | off | 1334 | 0.27498 | 0.04241 | 0.95165 | 0.95966 | 0.82432 | 0.72884 |
| update | 7 | off | 1434 | 0.27945 | 0.04149 | 0.94402 | 0.95483 | 0.81561 | 0.72180 |
| update | 7 | on | 1234 | 0.27833 | 0.04151 | 0.94367 | 0.95432 | 0.83267 | 0.71806 |
| update | 7 | on | 1334 | 0.27703 | 0.04632 | 0.95139 | 0.95933 | 0.82698 | 0.73139 |
| update | 7 | on | 1434 | 0.27841 | 0.03990 | 0.94470 | 0.95562 | 0.81811 | 0.72398 |
| recompute | 7 | off | 1234 | 0.26028 | 0.04233 | 0.94372 | 0.95315 | 0.82785 | 0.70989 |
| recompute | 7 | off | 1334 | 0.25639 | 0.04758 | 0.95275 | 0.96018 | 0.81465 | 0.71669 |
| recompute | 7 | off | 1434 | 0.26716 | 0.03902 | 0.94502 | 0.95518 | 0.80312 | 0.70765 |
| recompute | 7 | on | 1234 | 0.25974 | 0.03978 | 0.94326 | 0.95320 | 0.82701 | 0.70916 |
| recompute | 7 | on | 1334 | 0.25996 | 0.05164 | 0.95121 | 0.95867 | 0.81490 | 0.71598 |
| recompute | 7 | on | 1434 | 0.26760 | 0.03986 | 0.94497 | 0.95567 | 0.80060 | 0.70501 |


## 8. SignNet — a sign-invariant spectral encoder (the gauge fix)  **[confirmed, 2026-07-14]**

§4.2 argued that `recompute` fails because a fresh solve re-draws the eigenvector **gauge** every
snapshot, so the learned filter chases a coordinate system that never settles — and that a
sign-/basis-*invariant* encoder (SignNet, Lim et al.) was the principled fix. That encoder is now
built (`model.smodel_type=SignNet`, commit `3595fc8`) and swept.

**What it is.** A new smodel subclass alongside `DynamicSLaplace`, not a replacement:
`S = ρ( Σ_i [ φ(u_i) + φ(−u_i) ] )` — a shared per-entry φ maps each eigenvector column *and* its
sign-flip, the pair is summed (invariant to the per-eigenvector sign gauge), summed again over the
`spectral_len` columns, then ρ maps to the fusion width. Invariance is exact (verified:
`max|S(Q) − S(Q·diag(±1))| = 0`). It has no `Q@W` filter, so no SFV — φ/ρ join FedAvg via `state_dict`.
The now-redundant sum-based sign canonicalization is skipped under SignNet
(`calc_eignvalues(canonicalize_sign=False)`); **procrustes is left independent** — it aligns
*rotation*, a different gauge that SignNet does not remove, so the two can co-run.

**Runs.** 135 runs, all `hard_neg=degree` (de-saturates auc/ap ~0.96→~0.86, so the secondary metrics
are informative): reddit_body {SignNet, Laplace} × {keep, update, recompute} × proc{off,on} + feature,
C{3,5,7} × 3 seeds; plus as733 and UCI SignNet {keep, recompute}.

### 8.1 Reddit-body — the matched grid (3-seed means, hard-neg)
| model | mode | proc | mrr C3 | C5 | C7 | auc C3 | ap C3 | f1 C3 | mcc C3 |
|---|---|---|---|---|---|---|---|---|---|
| feature | – | – | **0.341** | 0.303 | 0.280 | 0.864 | 0.883 | 0.889 | 0.801 |
| Laplace | keep | – | 0.331 | 0.299 | 0.283 | 0.860 | 0.880 | 0.891 | 0.804 |
| Laplace | update | off | 0.336 | 0.303 | 0.280 | 0.862 | 0.882 | 0.890 | 0.803 |
| Laplace | update | on | 0.338 | 0.300 | 0.283 | 0.861 | 0.881 | 0.890 | 0.804 |
| Laplace | recompute | off | 0.324 | 0.292 | 0.275 | 0.861 | 0.880 | 0.888 | 0.801 |
| Laplace | recompute | on | 0.324 | 0.296 | 0.276 | 0.862 | 0.882 | 0.889 | 0.802 |
| SignNet | keep | – | 0.338 | **0.308** | 0.276 | 0.865 | 0.884 | 0.890 | 0.803 |
| SignNet | update | off | **0.341** | 0.305 | 0.281 | 0.864 | 0.884 | 0.891 | 0.806 |
| SignNet | update | on | **0.341** | 0.305 | 0.280 | 0.865 | 0.885 | 0.891 | 0.805 |
| SignNet | recompute | off | 0.340 | 0.306 | 0.282 | 0.864 | 0.884 | 0.890 | 0.803 |
| SignNet | recompute | on | 0.340 | 0.301 | **0.285** | 0.866 | 0.885 | 0.890 | 0.804 |

### 8.2 The mechanism — the lift is proportional to gauge churn
Δ MRR = SignNet − Laplace, same mode, proc-off:

| mode | gauge churn across snapshots | Δ C3 | Δ C5 | Δ C7 | mean Δ |
|---|---|---|---|---|---|
| keep | **none** — basis frozen at t=0 | +0.007 | +0.009 | −0.007 | **+0.003** |
| update | mild — basis tracked (Rayleigh-Ritz) | +0.005 | +0.002 | +0.001 | **+0.003** |
| recompute | **maximal** — basis re-solved each snapshot | **+0.016** | **+0.014** | **+0.007** | **+0.012** |

**`keep` is a natural control.** A frozen basis has no sign churn, so a sign-invariant encoder should
do nothing there — and it does nothing (+0.003, mixed sign, inside noise). That rules out the obvious
confound that SignNet simply wins by being a larger/better encoder. The gain appears **only where the
gauge churns, and scales with how much it churns**. This is the load-bearing result of §8.

### 8.3 Cross-dataset (3-seed means; MRR is comparable to §3.1/§6 — `hard_neg` affects only auc/ap)
| dataset | condition | C3 | C5 | C7 | source |
|---|---|---|---|---|---|
| uci | feature | 0.089 | 0.084 | 0.073 | §3.1 |
| uci | Laplace keep | 0.084 | 0.066 | 0.076 | §3.1 |
| uci | Laplace recompute (off) | **0.055** | 0.060 | 0.058 | §3.1 |
| uci | SignNet keep | 0.089 | 0.070 | 0.073 | new |
| uci | SignNet recompute (off) | **0.081** | 0.072 | 0.072 | new |
| as733 | feature | 0.314 | 0.280 | 0.268 | §6 |
| as733 | Laplace keep | 0.307 | 0.282 | **0.282** | §6 |
| as733 | Laplace recompute (off) | 0.299 | 0.276 | 0.277 | §6 |
| as733 | Laplace recompute (**on**) | **0.322** | **0.291** | 0.278 | §6 |
| as733 | SignNet keep | 0.305 | 0.292 | 0.277 | new |
| as733 | SignNet recompute (off) | 0.305 | 0.285 | 0.274 | new |

### 8.4 Breadth — SignNet on all six datasets  **[confirmed, 2026-07-19]**
189/189 runs (`signnet_breadth` + `signnet_rtitle_gap`): the five conditions {keep, update×proc,
recompute×proc} × C{3,5,7} × 3 seeds on reddit_title / bitcoin_alpha / bitcoin_otc (which had no
SignNet at all), plus the missing `update` and proc-on cells on as733 and uci. `hard_neg=degree`
throughout; MRR is unaffected by `hard_neg`, so the Δ against §6's Laplace/feature MRR is valid
without re-running the baselines. (The last hole — `reddit_title update proc-on C3`, which OOM'd
twice under co-scheduling — was filled 2026-07-22 by re-running it alone at MJ=1: MRR **0.414**,
Δ −0.005 vs Laplace, i.e. tracks feature like every other reddit_title cell.)

_SignNet 3-seed mean MRR, and Δ vs the SAME condition under the Laplace smodel (§6/§3.1):_
| dataset | mode | proc | C3 | C5 | C7 | ΔC3 | ΔC5 | ΔC7 |
|---|---|---|---|---|---|---|---|---|
| uci | keep | – | 0.089 | 0.070 | 0.073 | +0.005 | +0.004 | −0.003 |
| uci | update | off | 0.090 | 0.078 | 0.076 | +0.012 | +0.012 | +0.005 |
| uci | update | on | 0.076 | 0.073 | 0.070 | −0.006 | −0.005 | −0.007 |
| uci | recompute | on | 0.082 | 0.074 | 0.070 | **+0.022** | **+0.020** | +0.009 |
| bitcoin_alpha | keep | – | 0.130 | 0.100 | 0.091 | +0.013 | +0.027 | +0.041 |
| bitcoin_alpha | update | off | 0.120 | 0.103 | 0.096 | +0.006 | +0.024 | +0.042 |
| bitcoin_alpha | update | on | 0.116 | 0.108 | 0.086 | −0.006 | +0.015 | +0.020 |
| bitcoin_alpha | recompute | off | 0.123 | 0.108 | 0.098 | **+0.020** | **+0.052** | **+0.062** |
| bitcoin_alpha | recompute | on | 0.121 | 0.106 | 0.089 | +0.017 | +0.044 | +0.062 |
| bitcoin_otc | keep | – | 0.162 | 0.153 | 0.139 | +0.014 | +0.045 | +0.061 |
| bitcoin_otc | update | off | 0.154 | 0.133 | 0.124 | +0.020 | +0.027 | +0.057 |
| bitcoin_otc | update | on | 0.165 | 0.147 | 0.134 | +0.026 | +0.023 | +0.050 |
| bitcoin_otc | recompute | off | 0.163 | 0.150 | 0.134 | **+0.049** | **+0.066** | **+0.083** |
| bitcoin_otc | recompute | on | 0.154 | 0.147 | 0.121 | +0.035 | +0.065 | +0.062 |
| as733 | keep | – | 0.305 | 0.292 | 0.277 | −0.002 | +0.010 | −0.005 |
| as733 | update | off | 0.304 | 0.288 | 0.274 | +0.003 | +0.008 | −0.002 |
| as733 | update | on | 0.305 | 0.288 | 0.277 | +0.007 | +0.014 | +0.009 |
| as733 | recompute | on | 0.302 | 0.293 | 0.271 | −0.020 | +0.002 | −0.007 |
| reddit_title | keep | – | 0.417 | 0.395 | 0.387 | −0.001 | +0.004 | −0.005 |
| reddit_title | update | off | 0.414 | 0.395 | 0.389 | −0.003 | +0.003 | +0.003 |
| reddit_title | update | on | 0.414 | 0.395 | 0.389 | −0.005 | −0.001 | −0.000 |
| reddit_title | recompute | off | 0.414 | 0.392 | 0.389 | **+0.017** | +0.007 | +0.009 |
| reddit_title | recompute | on | 0.417 | 0.393 | 0.388 | **+0.019** | +0.010 | +0.007 |

**The §8.2 mechanism REPLICATES on 4 of 5 new datasets — with one important exception.**
`recompute` receives the largest lift on uci (+0.017 mean), bitcoin_alpha (+0.045), bitcoin_otc (+0.066)
and reddit_title (+0.011), matching reddit_body (+0.012). as733 is the exception: `recompute`-on gets
**−0.008**, i.e. no lift at all — exactly as the low-churn account predicts (a fresh solve barely moves
the gauge there, so there is no sign churn for SignNet to remove; §8.3 point 4).

_Control diagnostic — Δ MRR = SignNet − Laplace (same condition), averaged over C{3,5,7}. If the lift is
gauge-specific, `keep` (frozen basis, no sign churn) must be NULL while `recompute` lifts:_
| dataset | keep | update-off | update-on | recomp-off | recomp-on | recompute mean | control |
|---|---|---|---|---|---|---|---|
| reddit_body | +0.003 | +0.002 | +0.004 | +0.015 | +0.010 | **+0.012** | holds (4.9×) |
| reddit_title | −0.001 | +0.001 | −0.002 | +0.011 | +0.012 | **+0.011** | holds (12.5×) |
| as733 | +0.001 | +0.003 | +0.010 | +0.004 | −0.008 | **−0.002** | holds (low-churn: nothing to fix) |
| uci | +0.002 | +0.010 | −0.006 | +0.017 | +0.017 | **+0.017** | holds (7.6×) |
| **bitcoin_alpha** | **+0.027** | +0.024 | +0.010 | +0.045 | +0.041 | **+0.043** | **FAILS (1.6×)** |
| **bitcoin_otc** | **+0.040** | +0.035 | +0.033 | +0.066 | +0.054 | **+0.060** | **FAILS (1.5×)** |

**HONEST CAVEAT — the `keep`-control null does NOT hold on Bitcoin.** On uci, as733, reddit_title (and
reddit_body, §8.2) `keep` is unmoved by SignNet (mean Δ +0.002, +0.001, −0.001, +0.003) — the control
that pins the gain to the sign gauge. But on **bitcoin_alpha `keep` gains +0.027 and bitcoin_otc `keep`
gains +0.040**, and every mode is lifted substantially. On these small graphs SignNet is therefore a
**broadly better encoder**, not only a gauge fix, and the clean attribution of §8.2 does not carry over.
State this in the paper: the mechanism claim is well supported on the large/high-churn graphs where the
control holds, and is confounded with general capacity on the small Bitcoin graphs.

### 8.5 Bitcoin — SignNet removes the spectral failure, but via capacity not gauge  **[confirmed, 2026-07-19]**
§6 concluded that on the small Bitcoin graphs spectral **fails outright** — plain `feature` was best at
every C and *widened* its lead as C grew. That conclusion was a property of the **Laplace** smodel, not
of the spectral idea:

| dataset | C | feature | best Laplace (§6) | best SignNet | SignNet − feature |
|---|---|---|---|---|---|
| bitcoin_alpha | 3 | 0.126 | 0.122 (−0.004) | 0.130 | **+0.004** |
| bitcoin_alpha | 5 | 0.102 | 0.093 (−0.009) | 0.108 | **+0.006** |
| bitcoin_alpha | 7 | 0.091 | 0.066 (−0.025) | 0.098 | **+0.007** |
| bitcoin_otc | 3 | 0.151 | 0.148 (−0.003) | 0.165 | **+0.014** |
| bitcoin_otc | 5 | 0.131 | 0.124 (−0.007) | 0.153 | **+0.022** |
| bitcoin_otc | 7 | 0.113 | 0.084 (−0.029) | 0.139 | **+0.026** |

**The two Bitcoin datasets must be reported separately — only one is a genuine win.**
- **bitcoin_otc — a real rescue.** All five SignNet modes clear `feature` at C7, and the best-spectral
  margin (+0.014 / +0.022 / **+0.026**) **exceeds the ±0.01 noise floor at C5 and C7**. §6's "no rescue"
  is overturned here, and the margin grows with C (the §4.1 crossover shape).
- **bitcoin_alpha — parity, NOT a win.** The best-spectral margin is +0.004 / +0.006 / +0.007 — all three
  **inside the noise floor**, with individual modes mixed-sign. The honest statement is that SignNet lifts
  spectral from *clearly below* `feature` to *level with* it. **Do not claim a win on bitcoin_alpha.**

**CRITICAL — this is the §8.4 control failure, not the gauge mechanism.** The two datasets where spectral
improves here are *exactly* the two where the `keep`-control fails (§8.4: keep gains +0.027 / +0.040).
Since `keep` freezes the basis and therefore has **no sign churn to remove**, a gain there cannot be
gauge-related. The Bitcoin improvement is therefore SignNet acting as a **better encoder** on small graphs
— more capacity than the old `Q@W` filter — and **not** the sign-invariance mechanism of §8.2.
§8.4's caveat and §8.5's rescue are **the same effect observed twice, not two independent findings**;
reporting them as separate wins would double-count one phenomenon. The §6 nuance should be revised to:
*spectral's failure on the small Bitcoin graphs is specific to the Laplace smodel; a higher-capacity
encoder removes it (decisively on bitcoin_otc, to parity on bitcoin_alpha).*

On the remaining datasets the best SignNet mode merely **tracks** feature (uci +0.001/−0.006/+0.003,
reddit_title +0.001/−0.001/−0.001, as733 −0.009/+0.013/+0.009) — consistent with §8's headline that
SignNet buys parity, not superiority, wherever the Laplace smodel was already competitive.

**Read.**
1. **SignNet removes recompute's deficit.** On reddit_body it lifts `recompute` 0.324 → 0.340 and on
   UCI 0.055 → 0.081 (+47%) — out of last place, to parity with `keep`/`feature`. The gauge *was* the
   cost of freshness, and the cheap fixes of the earlier investigation (deterministic start, robust
   sign) could not recover it while a principled invariant encoder can.
2. **It reaches parity, not superiority.** SignNet-recompute (0.340) ties feature (0.341) and
   SignNet-update (0.341); every gap is inside the ±0.01 noise floor. **The "recompute is superior"
   premise still does not hold** — freshness stopped being harmful, it did not become a net gain.
3. **No metric dissociation.** Under hard negatives all five metrics agree (recompute ≈ keep to three
   decimals at C3). The earlier "recompute wins auc/ap but loses mrr/f1/mcc" split was a saturation
   artifact of 1:1 random negatives, and it is gone.
4. **High-churn vs low-churn is a clean complementary split.** On reddit_body (high churn) SignNet
   helps and procrustes adds nothing on top (0.340 off vs 0.340 on). On as733 (low churn) SignNet adds
   only +0.006 over Laplace-off, while Laplace `recompute`-**on** (0.322) remains the best number on
   that dataset. **High churn → the sign gauge is the problem; low churn → rotation alignment is.**
5. **The as733 proc-on cell is now RUN (§8.4)** and it does *not* rescue as733: SignNet
   `recompute`-on scores 0.302/0.293/0.271 vs Laplace's 0.322/0.291/0.278 (Δ −0.020/+0.002/−0.007), so
   Laplace `recompute`-on remains the best as733 number. Combining sign-invariance with rotation
   alignment buys nothing there — on a low-churn graph the operative gauge is rotation, not sign.
6. **Breadth (§8.4):** the mechanism replicates on 4 of 5 new datasets, but the `keep`-control null
   fails on Bitcoin, where SignNet lifts every mode — there it is a better encoder, not only a gauge fix.
7. **Bitcoin (§8.5):** SignNet overturns §6's "no rescue" on **bitcoin_otc** (best spectral +0.014/
   +0.022/+0.026 vs feature, clearing the noise floor at C5/C7); on **bitcoin_alpha** it reaches only
   parity (+0.004/+0.006/+0.007, all sub-noise). Both are driven by the §8.4 control failure — a
   general encoder gain, NOT the sign-invariance mechanism. Points 6 and 7 describe ONE effect.

---

## 9. Does federation earn its keep? — the local-only lower bound  **[confirmed, 2026-07-16]**

Every result in §2–§8 is federated (`FL=True`). Without a no-aggregation floor there is no evidence
that FedAvg beats "every client for itself". `federated.fl` (commit `f45c3a3`) exposes FedLap's
long-dormant `FL` flag: with `fl=false` each client trains on its own subgraph and is evaluated on its
own next snapshot, with **no broadcast and no aggregation** (verified by instrumentation: 0 `sum_lod`
calls and exactly 1 `share_weights`, the common init).

**A confound found and fixed — read this before comparing.** Eval was historically *hard-wired* to the
`FL` flag: `fl=true` scored the **global stitched graph**, `fl=false` scored **per-client subgraphs**.
That conflates training with the *test set* (the local arm is scored on a smaller, easier task), and it
produced a spurious result — as733 appeared to **lose 0.124 AUC** under federation. `metric.eval_scope`
(commit `a3e907a`) decouples them; on the matched per-client test set that gap is **+0.003**. Everything
below uses `eval_scope=local` for **both** arms, so the two differ **only in training**.
(`eval_scope=global` with `fl=false` is now rejected by `assert_cfg`: the global eval decodes with the
*server* model, which is never aggregated — hence never trained — when `fl=false`.)

Wiring the dormant path also exposed a genuine latent bug (`18cdb6e`, `95ce06a`): `refresh()` inherits
whatever train/eval mode it finds, but a client that **abstains** (`can_train` false) is never touched
by `val_loss`/`local_finetune`, so at t=0 it reaches `refresh` in train mode and crashes the encoder
BatchNorm on a 1-node batch. `FL=True` had masked this via an incidental `eval()` inside `val_loss`.
Both paths now normalize to eval before `refresh` — without this the local baseline is impossible to
run on the Bitcoin graphs at all.

**Runs.** 144 matched runs: 6 datasets × C{3,5,7,9} × 3 seeds × {federated, local}, feature model, gru,
random negatives, identical seeds, all per-client eval.

### 9.1 Δ MRR (federated − local) — noise floor ±0.01
| dataset | C3 | C5 | C7 | C9 |
|---|---|---|---|---|
| uci | +0.019 | +0.018 | +0.013 | −0.001 |
| bitcoin_alpha | −0.000 | −0.005 | +0.022 | +0.023 |
| bitcoin_otc | +0.015 | +0.002 | +0.003 | +0.015 |
| as733 | +0.002 | +0.006 | +0.013 | **+0.019** |
| reddit_body | **+0.060** | **+0.049** | **+0.057** | **+0.065** |
| reddit_title | +0.018 | +0.019 | +0.028 | **+0.039** |

### 9.2 Δ AUC (federated − local)
| dataset | C3 | C5 | C7 | C9 |
|---|---|---|---|---|
| uci | +0.037 | +0.022 | +0.057 | **+0.094** |
| bitcoin_alpha | +0.008 | +0.034 | +0.143 | **+0.202** |
| bitcoin_otc | +0.026 | −0.013 | +0.016 | **+0.050** |
| as733 | +0.002 | +0.003 | +0.003 | +0.004 |
| reddit_body | +0.012 | +0.020 | +0.024 | **+0.030** |
| reddit_title | +0.006 | +0.011 | +0.014 | **+0.018** |

### 9.3 Absolute MRR — local → federated
| dataset | C3 | C5 | C7 | C9 |
|---|---|---|---|---|
| uci | 0.084→0.103 | 0.072→0.089 | 0.073→0.086 | 0.097→0.096 |
| bitcoin_alpha | 0.117→0.117 | 0.102→0.098 | 0.088→0.111 | 0.080→0.103 |
| bitcoin_otc | 0.129→0.144 | 0.110→0.112 | 0.097→0.100 | 0.080→0.095 |
| as733 | 0.258→0.260 | 0.222→0.227 | 0.205→0.218 | 0.178→0.197 |
| reddit_body | 0.297→0.357 | 0.268→0.317 | 0.248→0.305 | 0.224→0.288 |
| reddit_title | 0.407→0.425 | 0.386→0.405 | 0.377→0.405 | 0.377→0.415 |

**Read — federation's benefit is not a fixed offset; it SCALES with fragmentation.**
1. **Δ AUC grows monotonically with C on 5 of 6 datasets** (all but as733, whose AUC is saturated
   ~0.98 with no room to move), and C9 is the largest gap in every one. A four-point monotone trend,
   not a two-point hunch. bitcoin_alpha is the extreme: +0.008 → **+0.202**.
2. **Δ MRR grows with C on 4 of 6** (as733, reddit_body, reddit_title, bitcoin_alpha). uci's gap washes
   into noise by C9 and bitcoin_otc stays small — MRR is the noisier metric here, AUC the cleaner one.
3. **The mechanism is visible in the absolutes (§9.3):** the gap widens because **local collapses
   faster** (reddit_body local 0.297→0.224 over C3→C9, as733 0.258→0.178) while federated holds up
   (reddit_body 0.357→0.288). This is precisely "resilience as the federation fragments".
4. **Federation earns its keep** on reddit_body (largest, +0.05–0.065 at every C), reddit_title, as733
   and bitcoin_alpha; it is **within noise on uci** and marginal on bitcoin_otc. Honest scope: it is
   not a universal win, and the small graphs are where it is weakest.
5. F1/MCC swing much more than MRR/AUC (fixed-threshold metrics over tiny per-client test sets);
   treat **MRR and AUC as the trustworthy columns** in §9.

---

## 10. The basis-content control — structure is never used; only temporal stability is  **[confirmed, 2026-07-21]**

> **This section supersedes the mechanism of §4.2 and re-frames §8.** It is the most
> consequential result in this file. Read it before designing any further spectral encoder.
> **One-line finding:** across two orthogonal controls, the spectral branch does **not** use the
> graph's structural content at all; the only thing it is (mildly) sensitive to is the **temporal
> stability** of whatever per-node code is injected — and the real eigenbasis provides that only
> incidentally, on low-churn graphs.

### 10.1 The two controls
§4.2 explained `recompute`'s deficit as a **gauge** problem, and §8 built SignNet to remove the sign
gauge — parity only. Before building BasisNet we asked the prior question no experiment had tested:
**does the branch use the graph spectrum at all?** `spectral.basis_source` (commit `1690824`) swaps
the eigenbasis for a null basis of the same shape, server-side in `_substitute_basis`, identically
for every smodel and mode:
- `laplacian` — the real Laplacian eigenbasis (default).
- `random` — a Haar-random orthonormal matrix (QR of a Gaussian), **redrawn every snapshot-solve**.
- `shuffled` — the **real** eigenvectors with the node→row order permuted (identical values, exact
  orthonormality, structure destroyed), **re-permuted every snapshot-solve**.

Crossed with `update_mode` this yields two clean, separable controls:
- **STRUCTURE control** = `keep` (basis frozen at t=0). All three sources are then *temporally
  constant*, so they differ **only in structural content**. If structure is used, `laplacian` must
  beat `random`/`shuffled` here.
- **STABILITY effect** = `recompute` (basis re-solved each snapshot). The real basis is re-solved
  but on a low-churn graph barely moves (temporally *stable*); `random`/`shuffled` are redrawn from
  scratch each step (temporally *unstable*). This isolates sensitivity to **temporal stability**.

### 10.2 Result — structure control is NULL everywhere; the only non-null is a stability effect
Grid: {uci, bitcoin_otc, reddit_body, as733, reddit_title} × {keep, recompute} × fusion{add, concat}
× basis{laplacian, random, shuffled} × C{3,7} (C3 only for as733/reddit_title) × 3 seeds, gru,
`sfv_share=local`. Noise floor 0.005–0.007 MRR.

**(a) STRUCTURE control — `keep`, frozen basis. `laplacian` vs `random`/`shuffled`:**
| dataset | lap | random | shuffled | Δrandom | Δshuffled |
|---|---|---|---|---|---|
| reddit_body / keep / add / C3 | 0.337 | 0.313 | 0.337 | −0.024 | −0.000 |
| bitcoin_otc / keep / add / C3 | 0.140 | 0.141 | 0.141 | +0.001 | +0.001 |
| uci / keep / add / C3 | 0.088 | 0.097 | 0.088 | +0.010 | +0.000 |
| **as733 / keep / add / C3** | **0.305** | **0.318** | **0.299** | **+0.013** | **−0.006** |
| reddit_title / keep / add / C3 | 0.417 | 0.416 | 0.416 | −0.001 | −0.001 |

Over all matched `keep` conditions the deltas are mean ≈ 0 with mixed sign and max |Δ| at the noise
floor. **The real eigenbasis never beats a random matrix of the same shape when both are frozen —
including on as733, where `random` is nominally *higher*.** Graph-structural content is not used.

**(b) STABILITY effect — `recompute`. The one real deviation is as733, and it is about stability:**
Per-seed MRR, as733 recompute/add/C3 (3 seeds, no cross-condition overlap):
| basis | seed 1234 | 1334 | 1434 | mean |
|---|---|---|---|---|
| laplacian (stable on low-churn) | 0.341 | 0.345 | 0.339 | **0.342** |
| random (redrawn each snapshot) | 0.297 | 0.305 | 0.297 | 0.300 |
| shuffled (re-permuted each snapshot) | 0.296 | 0.299 | 0.297 | 0.297 |

Here `laplacian` beats both null bases by **~0.042**, cleanly above noise on every seed. But (a)
already proved this is **not** structure: a *frozen* random basis (as733 keep/random 0.318) does as
well as the frozen real one (0.305). The recompute gap therefore isolates **temporal stability** —
the real basis wins only because on low-churn as733 a fresh solve returns nearly the same basis each
step, while `random`/`shuffled` re-randomize the per-node code every snapshot.

**(c) The stability effect vanishes on high churn — as predicted.** as733 is the *only* dataset
where recompute's real basis is temporally stable. On high-churn reddit_body the real basis churns
as much as random, so recompute is null again: reddit_body recompute/add/C3 lap 0.324 / random 0.326
/ shuffled 0.335 (per-seed lap 0.317/0.335/0.321 — fully overlapping with the null bases). Same on
uci and bitcoin_otc.

**Aggregate (Laplace smodel), 5 datasets:**
| comparison | scope | n | mean Δ | mean abs Δ | max abs Δ | verdict |
|---|---|---|---|---|---|---|
| random − laplacian | keep (structure) | 16 | −0.001 | 0.007 | 0.024 | NULL — structure unused |
| shuffled − laplacian | keep (structure) | 16 | −0.003 | 0.006 | 0.026 | NULL — structure unused |
| random − laplacian | recompute, high-churn (uci/btc/reddit) | 14 | +0.003 | 0.004 | 0.009 | NULL — real basis also churns |
| random − laplacian | **recompute, as733 (low-churn)** | 2 | **−0.037** | 0.037 | 0.042 | **real basis wins via stability** |
| concat − add (laplacian) | keep+recompute | 16 | +0.003 | 0.006 | 0.027 | NULL — fusion not the bottleneck |

(as733/reddit_title were C3-only, so their recompute buckets are n=2 per basis; the as733 stability
gain is nonetheless clean — all three seeds separate with no overlap, §10.2b.)

**`shuffled` is the tighter control** (matched on every numeric: byte-identical values, ‖·‖_F 9.899
vs the real 9.899, identical conditioning; `random` differs also in norm ‖·‖_F √300≈17.32 and exact
orthonormality). Since clients receive a **row slice** `U[nid]` (`dynamic_server.py`), permuting rows
severs the node↔coordinate map while holding all numerics fixed — and it too is null under `keep`.

**Plumbing verified, not assumed.** The delivered basis was instrumented at the smodel: `laplacian`
‖·‖_F 9.899, `random` 17.320 (orthonormality residual 1.6e-6), `shuffled` 9.899 with permuted rows —
the model demonstrably receives a different matrix in each arm, so the nulls are real nulls.

### 10.3 The mechanism — the smodel discards the spectrum before fusion
Instrumented effective rank (entropy of the Gram spectrum) through the Laplace smodel, UCI keep
C=1, comparing the first and last 10% of encode calls:

| stage | at init | trained |
|---|---|---|
| `Q` (N,300) eigenbasis | 94.6 | 94.6 |
| `Z = Q@W` (N,512) | 73.4 | 34.7 |
| LayerNorm(512) | 92.8 | 15.8 |
| Linear→128 | 19.1 | 3.4 |
| Linear→64 (`H`) | 4.6 | **1.7** |
| `S = bn(H)` (N,64) | 4.9 | **2.3** |

`Q` carries ~95 effective dimensions; **~2 reach the fusion point**, and the collapse is *learned*
(rank 4.9 → 2.3 over training). A rank-2 channel cannot transport the spectrum, which is exactly
why its content is interchangeable with noise.

Magnitude is **not** the problem — the obvious rival hypothesis is refuted. Because `gnn.l2norm=True`
makes `z` unit-norm while `S` is added raw, ‖S‖/‖z‖ *is* the signal ratio: measured **0.29 (keep)
/ 0.43 (recompute)**, with **99% of S's energy node-varying** (not a constant offset). The branch is
a large, node-specific perturbation that carries no usable information — and `recompute`'s *larger*
‖S‖ with the same rank-2 content is a natural account of why it scores *worse* (0.062 vs 0.101 on
UCI C=1): more noise injected into a unit-norm embedding.

### 10.4 Fusion is not the bottleneck either
`concat` (own 64 dims, independent head weights, no perturbation of `z`) was the alternative
hypothesis — that additive fusion crowds S out. It is refuted twice: end-to-end **mean Δ +0.0007**
(table above), and mechanistically the trained rank under `concat` is **1.8**, no better than
`add`'s 2.3. Giving the branch its own channels does not stop the optimizer throttling it.

### 10.5 What this explains, and what it costs
The two-control result — *structure never used; only temporal stability matters* — explains every
prior finding in this file with a single mechanism:
1. **The four null prototypes** (active-node Laplacian, symmetric Laplacian, count/running-mean edge
   weights, EMA decay) all varied *how Q is built*, i.e. its structural content. Structure is not
   used, so all four had to come back null. This is now a corollary, not a coincidence.
2. **`keep > update > recompute` (§3/§6) is a stability ordering, not a spectral one.** `keep`
   injects a temporally constant code (most stable → best); `recompute` re-randomizes the code each
   snapshot unless churn is low (least stable → worst on high-churn graphs); `update` drifts in
   between. §4.2 attributed this to the eigenvector *gauge*; the correct variable is the temporal
   stability of the injected per-node signal, which the eigenbasis affects only via churn. The
   **churn-dependence §4.2/§6 already documented is real and now has the right cause**: as733 is the
   one low-churn graph, so it is the one place recompute's real basis is stable and beats its own
   random control (§10.2b) — the clean confirmation the churn story needed.
3. **SignNet's parity (§8).** An invariant encoder over a signal whose *content* is unused cannot add
   information; consistent with §8.4/§8.5's own reading (capacity on small graphs, not a gauge fix).
   The reddit_body SignNet basis control is itself null (keep Δ +0.002/+0.003), i.e. on the large
   graph SignNet does not use structure either — **with one flagged exception, bitcoin_otc, below.**

**The one cell that resists the null — bitcoin_otc under SignNet [prelim, 2026-07-22].** The full
SignNet basis control on bitcoin_otc (36 runs, C{3,7} × {keep, recompute} × 3 bases) is NOT clean:
| condition | laplacian | random | shuffled | Δrnd | Δshf |
|---|---|---|---|---|---|
| keep C3 | 0.160 | 0.155 | 0.165 | −0.005 | +0.005 |
| keep C7 | 0.141 | 0.127 | 0.120 | **−0.015** | **−0.021** |
| recompute C3 | 0.171 | 0.159 | 0.160 | −0.012 | −0.011 |
| recompute C7 | 0.140 | 0.121 | 0.120 | **−0.018** | **−0.020** |
Per-seed structure matters here: `laplacian` is *stable on all six C7 seeds* (0.134–0.149) while
each null arm has one collapsed seed (0.091–0.110) plus otherwise slightly-lower values. On the
noisiest dataset (per-run std ~0.09–0.13) three seeds cannot settle whether this is a real
structural signal or occasional training instability under a meaningless injected code — but the
direction is consistent across all four conditions, and notably this is the SAME dataset where
SignNet's win over `feature` was real (§8.5). Two honest readings: (a) on the small sharded graph
the real basis genuinely contributes (then §8.5's "+0.026 over feature at C7" decomposes into a
generic injected-code part — null bases still beat feature 0.120–0.127 vs 0.115 — plus a
basis-specific part ~0.015–0.020); or (b) null-code training is simply less stable on tiny shards
and occasionally collapses a seed. Distinguishing them needs more seeds; **flagged [prelim], not
folded into the §10 headline, which rests on the five clean datasets.** Keep C3's null (mixed sign)
argues against a simple structure story even here.
4. **BasisNet is predicted null** and should not be built. It would make the encoder invariant to the
   basis's rotation/sign — but the basis's structural content is unused, and the one thing that
   matters (temporal stability) is already maximized by `keep` / a deterministic solve, neither of
   which BasisNet touches. §1's degeneracy rationale is also unsupported: on the **cumulative** union
   the model consumes, the Ritz spectrum has **0/299 gaps < 1e-4 and no zero eigenvalues** (uci,
   reddit_body) — no degenerate eigenspaces to be invariant to. The "31k–35k zero eigenvalues" figure
   describes per-window slices, which `_spectral_step` deliberately does not use.

**Honest scope.** Confirmed on 5 datasets (uci, bitcoin_otc, reddit_body at C{3,7}; as733,
reddit_title at C3), gru, `keep`/`recompute`, Laplace smodel + reddit_body SignNet; the one
resisting cell is bitcoin_otc under SignNet at C7 ([prelim] above — needs more seeds). It does **not**
prove graph spectra are useless for this task — it proves **this pipeline does not transport their
structure**; every result in §3–§8 attributed to *basis maintenance policy* must be re-read as a
temporal-stability effect on a rank-2 injected code. The as733 recompute stability gain (~0.04) is
the one place the real basis measurably helps, and even there a stable *random* code would do the
same job.

**Where a real spectral result would have to start:** the bottleneck is the smodel's capacity (the
512→128→64 MLP with `dropout=0.5` and LayerNorm on a 512-wide input collapses Q's ~95 effective dims
to ~2, §10.3), not the basis. Either remove that bottleneck, or bypass the learned filter and feed a
*stable* structural code the fmodel cannot switch off. Until a variant beats its own
`basis_source=random` control under `keep` (i.e. uses structure, not just stability), no encoder
change should be believed. **[This experiment was subsequently run — §10.7. Verdict: still no
consistent gain.]**

### 10.6 The oracle probes — the consumed basis never carried the signal  **[confirmed, 2026-07-22; UCI-only]**

Model-free measurement of what the spectral representation *could* contribute, independent of any
smodel/optimizer: score node pairs by spectral affinity (cosine/dot/heat-kernel of basis rows) and
compute AUC on two tasks — *reconstructing the current cumulative graph* (can the basis encode the
graph it was computed from at all?) and *predicting t+1 links* (the actual task). Random negatives;
27 snapshots; per-t means.

| representation | reconstruct current graph | predict t+1 links |
|---|---|---|
| **Arnoldi-300 of L_rw — what every §3–§8 run consumed** | **0.53** | **0.50–0.52** |
| exact eigh, sym-normalized, 300 lowest nontrivial | 0.965 | 0.663 |
| exact eigh, 50 lowest nontrivial | 0.871 | **0.710** |
| Adamic-Adar / degree-product / previous-edge (baselines) | – | 0.674 / 0.762 / 0.704 |

Two findings, one per row block:
1. **The pipeline's basis is empty.** The Arnoldi/Lanczos estimate (`lanczos_iter=400`,
   `spectral_len=300`, both `LanczosLaplace` and `SignNet` consume it via `estimate=True`) is at
   CHANCE even at reconstructing its own graph. The cause is classical numerics: the informative
   eigenvectors are the smooth low-frequency ones, and the cumulative graph's low spectrum is
   massively clustered (§10.5: 292/299 gaps < 1e-2) — exactly where a plain Krylov run returns
   arbitrary mixtures instead of eigenvectors (corroborated: orthogonality residual 0.37, a negative
   Ritz value on a PSD operator). **§10.2's real≈random null is therefore overdetermined** — the
   "real" basis was itself nearly noise.
2. **Real signal exists, and part of it is complementary.** The exact low-50 basis predicts future
   links at 0.710 — comparable to the local heuristics — and on the **52% of future edges where
   Adamic-Adar is blind** (zero common neighbors) it still scores **0.674**. So on UCI there is
   genuinely complementary structural signal that no local-neighborhood feature carries. (Whether
   the 8-layer message-passing fmodel already captures it is what §10.7 tests.)

Scope: UCI only (dense eigh is cheap at 1.9k nodes); affinity tested under cosine/dot/two
heat-kernel weightings, all agreeing. Probe scripts in the session scratchpad, not committed.

### 10.7 Input-side exact LapPE — the fix-everything experiment, still no consistent gain  **[confirmed, 2026-07-23]**

§10.3–§10.6 identified three stacked causes: empty consumed basis, output-side fusion, learned rank
collapse. `model.data_type=f+pe` (commits `e525928`, `bcc43af`) removes all three at once:
- **exact** k=50 lowest nontrivial eigenpairs of the **sym-normalized** Laplacian
  (`Graph.calc_eigs_exact_sym`: dense eigh ≤3k nodes, sparse shift-invert above, solved on the
  active subgraph — isolated nodes get zero rows), scaled by √N to O(1) entries;
- **injected at the INPUT** (`FedDynamicPEClassifier` concatenates the served per-client slice onto
  node features before the encoder), LapPE-style, so message passing can use the coordinates
  relationally and no learned filter can throttle them;
- judged against **stability-matched structure controls**: `basis_source=shuffled_fixed` (one FIXED
  node-permutation of the drifting real basis — identical values, identical temporal drift,
  node↔structure correspondence severed) and `random_fixed` (one frozen random orthonormal basis).
  These close §10.2's confound: under `recompute` the plain nulls redraw per snapshot and mix
  stability into the comparison.

**Δ MRR = laplacian-PE − shuffled_fixed-PE (the structure signal), 3-seed means:**
| dataset | mode | C1 | C3 | C7 |
|---|---|---|---|---|
| uci | recompute | **+0.013** | **+0.015** | −0.001 |
| uci | keep | +0.007 | +0.006 | +0.007 |
| bitcoin_otc | recompute | +0.007 | **−0.023** | −0.018 |
| bitcoin_otc | keep | – | −0.009 | **−0.013** |
| as733 | recompute | – | +0.006 | **−0.012** |

**Verdict: no consistent structural benefit — the sign flips across datasets.**
- **uci**: the one real positive of the whole program — 5/6 cells positive, mean **+0.008**, and
  laplacian-PE beats plain `feature` at every C (+0.012/+0.004/+0.003; C1 all three seeds separate).
  This is exactly the dataset where §10.6 measured complementary oracle signal, so the effect lands
  where predicted — but it is small (~1–1.5× the noise floor).
- **bitcoin_otc**: the real basis is neutral at C1 and *worse than its own structure-destroyed
  control* federated (mean −0.011). Echoing §8.5/§10.5: the null codes themselves BEAT `feature` at
  C7 (`shuffled_fixed` keep 0.137 vs feature 0.124) — on small sharded graphs *any* extra input
  code helps as capacity, while the real drifting basis actively hurts at high C
  (recompute-laplacian C7 0.093 vs feature 0.124).
- **as733**: within noise at C3 (0.334 vs 0.328, seeds overlap).

**Full-metric record (3-seed means; random negatives, so auc/ap are the saturated variant comparable
to §6, not §8's hard-neg):**

_uci (27 snaps/run):_
| condition | C | mrr | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|
| feature | 1 | 0.1126 | 0.899 | 0.905 | 0.802 | 0.636 |
| feature | 3 | 0.0898 | 0.862 | 0.874 | 0.735 | 0.562 |
| feature | 7 | 0.0762 | 0.801 | 0.822 | 0.508 | 0.369 |
| PE keep/laplacian | 1 | 0.1199 | 0.903 | 0.911 | 0.785 | 0.637 |
| PE keep/laplacian | 3 | 0.0869 | 0.871 | 0.880 | 0.718 | 0.568 |
| PE keep/laplacian | 7 | 0.0832 | 0.792 | 0.820 | 0.490 | 0.387 |
| PE keep/shuffled_fixed | 1 | 0.1134 | 0.901 | 0.908 | 0.783 | 0.641 |
| PE keep/shuffled_fixed | 3 | 0.0812 | 0.859 | 0.870 | 0.707 | 0.544 |
| PE keep/shuffled_fixed | 7 | 0.0762 | 0.794 | 0.819 | 0.511 | 0.391 |
| PE recompute/laplacian | 1 | 0.1242 | 0.901 | 0.909 | 0.787 | 0.644 |
| PE recompute/laplacian | 3 | 0.0942 | 0.863 | 0.875 | 0.706 | 0.544 |
| PE recompute/laplacian | 7 | 0.0789 | 0.790 | 0.818 | 0.471 | 0.367 |
| PE recompute/shuffled_fixed | 1 | 0.1114 | 0.890 | 0.897 | 0.780 | 0.616 |
| PE recompute/shuffled_fixed | 3 | 0.0793 | 0.863 | 0.876 | 0.716 | 0.552 |
| PE recompute/shuffled_fixed | 7 | 0.0798 | 0.809 | 0.830 | 0.490 | 0.372 |
| PE recompute/random_fixed | 1 | 0.1140 | 0.897 | 0.903 | 0.788 | 0.633 |
| PE recompute/random_fixed | 3 | 0.0847 | 0.862 | 0.874 | 0.707 | 0.545 |
| PE recompute/random_fixed | 7 | 0.0756 | 0.797 | 0.822 | 0.471 | 0.371 |

_bitcoin_otc (261 snaps/run):_
| condition | C | mrr | auc | ap | f1 | mcc |
|---|---|---|---|---|---|---|
| feature | 1 | 0.2020 | 0.947 | 0.958 | 0.887 | 0.793 |
| feature | 3 | 0.1502 | 0.850 | 0.889 | 0.756 | 0.578 |
| feature | 7 | 0.1235 | 0.809 | 0.846 | 0.540 | 0.320 |
| PE keep/laplacian | 3 | 0.1502 | 0.887 | 0.915 | 0.785 | 0.617 |
| PE keep/laplacian | 7 | 0.1239 | 0.798 | 0.841 | 0.557 | 0.288 |
| PE keep/shuffled_fixed | 3 | 0.1593 | 0.892 | 0.917 | 0.769 | 0.601 |
| PE keep/shuffled_fixed | 7 | 0.1372 | 0.841 | 0.865 | 0.500 | 0.262 |
| PE recompute/laplacian | 1 | 0.2054 | 0.950 | 0.962 | 0.892 | 0.804 |
| PE recompute/laplacian | 3 | 0.1317 | 0.843 | 0.892 | 0.745 | 0.562 |
| PE recompute/laplacian | 7 | 0.0929 | 0.805 | 0.848 | 0.581 | 0.394 |
| PE recompute/shuffled_fixed | 1 | 0.1989 | 0.948 | 0.960 | 0.887 | 0.796 |
| PE recompute/shuffled_fixed | 3 | 0.1546 | 0.881 | 0.912 | 0.744 | 0.565 |
| PE recompute/shuffled_fixed | 7 | 0.1105 | 0.812 | 0.851 | 0.618 | 0.364 |
| PE recompute/random_fixed | 1 | 0.2059 | 0.952 | 0.965 | 0.887 | 0.797 |
| PE recompute/random_fixed | 3 | 0.1348 | 0.849 | 0.896 | 0.755 | 0.572 |
| PE recompute/random_fixed | 7 | 0.1134 | 0.748 | 0.816 | 0.500 | 0.243 |

_as733 (732 snaps/run; feature row = the fresh `abl_as733` baseline, same code era; C7 harvested
2026-07-25 after the ban was lifted):_
| condition | C | mrr |
|---|---|---|
| feature | 3 | 0.3270 |
| PE recompute/laplacian | 3 | 0.3339 |
| PE recompute/shuffled_fixed | 3 | 0.3280 |
| PE recompute/laplacian | 7 | 0.2699 |
| PE recompute/shuffled_fixed | 7 | 0.2821 |

(as733 C3 secondary metrics: laplacian 0.939/0.948/0.799/0.679 vs shuffled 0.940/0.948/0.798/0.678
— an all-metric null; at C7 the real basis is 0.012 BELOW its placebo.)

Cross-metric notes: (a) on bitcoin_otc at C3/C7 the PE conditions lift auc/ap over feature by
+0.03–0.04 — but for the NULL bases as much as (or more than) the real one (keep/shuffled_fixed C7
auc 0.841 vs keep/laplacian 0.798 vs feature 0.809), so even on secondary metrics the injected-code
benefit on small graphs is capacity, not structure, and it does not convert into MRR; (b) on as733
all five metrics are identical between laplacian and shuffled_fixed to ~3 decimals — a textbook
all-metric null; (c) on uci the secondary metrics track the small MRR gains (keep/laplacian C3 auc
0.871 vs shuffled 0.859) without changing the story.

### 10.8 Per-snapshot analysis — a late-emerging effect the whole-run means dilute  **[prelim, 2026-07-25]**

A methodological objection (user-raised, correct to check): all §10 verdicts are whole-run MEAN
MRR, which could mask time-dependence — e.g. the cumulative graph densifies over the run, so the
basis (and the model's ability to use it) may differ early vs late. Two checks:

**(a) Does the Arnoldi basis become informative late, as density grows? NO — it degrades.**
Re-running the §10.6 oracle probe binned by early/mid/late thirds (uci): reconstruction AUC
0.56 → 0.53 → 0.52, future-link 0.52 → 0.52 → 0.50 (dot variant 0.52 → 0.50 → 0.48). Densification
CLUSTERS the low spectrum further, which hurts a fixed-budget Krylov solve more than density helps
it. The §10.6 conclusion is time-uniform.

**(b) Does the REAL basis's advantage grow late? YES, in the recompute conditions — the means
were diluting it.** Δ MRR (laplacian − shuffled control) from the per-snapshot logs, binned:

| condition | Δ early | Δ mid | Δ late |
|---|---|---|---|
| f+s Arnoldi, uci recompute C3 | +0.004 | −0.006 | **+0.025** |
| f+s Arnoldi, as733 recompute C3 | +0.014 | +0.055 | **+0.065** |
| input-PE exact, uci recompute C1 | +0.010 | +0.021 | +0.008 |
| input-PE exact, uci recompute C3 | −0.001 | −0.004 | **+0.050** |
| input-PE exact, as733 recompute C3 | −0.000 | +0.003 | +0.016 |
| input-PE exact, bitcoin_otc recompute C3 | −0.022 | −0.013 | **−0.034** |
| (keep conditions, all) | ~0 | ~0 | ~0 |

Reads, stated carefully: (1) the late-window growth in the f+s-Arnoldi rows cannot be basis
quality (see (a)) — on as733 it is the §10.2b stability effect compounding over 732 snapshots
(the re-drawn control keeps re-randomizing; the real basis barely moves). (2) The **input-PE uci
late-window +0.050 is against the drift-matched `shuffled_fixed` control**, so it IS
structure-attributable — and it is 3–6× the whole-run mean (+0.008), i.e. the mean genuinely
dilutes a late-emerging structural benefit on uci. With ~9 snapshots × 3 seeds per bin it is
~2.5 standard errors: suggestive, not confirmed. (3) bitcoin_otc stays negative in every bin —
the cross-dataset sign flip is not a timing artifact. (4) `keep` stays null in every bin — the
late effect is specific to a fresh basis tracking the grown graph. Implication if (2) replicates:
on long runs, evaluation windows (early/late splits) should complement whole-run means, and the
honest §10.7 verdict gains a nuance — "no consistent gain *in whole-run means*; a late-window
structural benefit exists on uci/as733 but reverses on bitcoin_otc."

### 10.9 Decoder and depth ablations — the last two architectural outs, both null  **[confirmed, 2026-07-25]**

**Decoder (`model.edge_decoding`).** Motivation: spectral affinity is a product (S_u·S_v); the
concat-MLP head must learn multiplicative interactions, which MLPs do poorly (He et al. 2017 vs
Rendle et al. 2020; Beutel et al. 2018), while a `dot` decoder computes them natively — a possible
"product-blind readout" explanation for §10.7's null. Grid: {dot, cosine_similarity(uci only)} x
{feature, f+pe-recompute laplacian, f+pe-recompute shuffled_fixed} x C{1,3,7} x 3 seeds; concat arm
= the §10.7 runs. Mean±std:

_uci:_
| condition | C | concat | dot | cosine |
|---|---|---|---|---|
| feature | 1 | 0.113±0.005 | 0.061±0.011 | 0.050±0.005 |
| f+pe laplacian | 1 | 0.124±0.008 | 0.066±0.013 | 0.048±0.006 |
| f+pe shuffled_fixed | 1 | 0.111±0.004 | 0.073±0.005 | 0.047±0.006 |
| feature | 3 | 0.090±0.017 | 0.059±0.014 | 0.043±0.007 |
| f+pe laplacian | 3 | 0.094±0.005 | 0.064±0.012 | 0.034±0.005 |
| f+pe shuffled_fixed | 3 | 0.079±0.017 | 0.064±0.007 | 0.040±0.010 |
| feature | 7 | 0.076±0.012 | 0.051±0.005 | 0.030±0.003 |
| f+pe laplacian | 7 | 0.079±0.009 | 0.046±0.005 | 0.033±0.005 |
| f+pe shuffled_fixed | 7 | 0.080±0.003 | 0.045±0.004 | 0.029±0.007 |

_bitcoin_otc (dot arm):_
| condition | C1 | C3 | C7 |
|---|---|---|---|
| feature | 0.201±0.009 | 0.149±0.006 | 0.095±0.010 |
| f+pe laplacian | 0.198±0.010 | 0.144±0.002 | 0.087±0.026 |
| f+pe shuffled_fixed | 0.196±0.011 | 0.146±0.006 | 0.087±0.006 |

Verdicts: (1) **concat is decisively best** — ROLAND's own decoder choice validated (dot loses
0.03–0.05 MRR on uci; on bitcoin dot is free at C1 but loses at C7). (2) **The product-readout
hypothesis is refuted on both datasets**: under dot, real ≈ placebo at every C, and uci's small
concat-arm laplacian edge disappears. The decoder axis is CLOSED.

**Depth (`gnn.dims` length L ∈ {1,2,4,8}).** Motivation: 8 MP layers may already span the graph,
making spectral coordinates redundant — predicting the PE gain should appear at shallow depth.
Grid: uci, {feature, f+pe-recompute laplacian, f+pe-recompute shuffled_fixed} x C{1,3,7} x 3 seeds.

| condition | C | L=1 | L=2 | L=4 | L=8 |
|---|---|---|---|---|---|
| feature | 1 | 0.119±0.003 | 0.121±0.004 | 0.114±0.012 | 0.113±0.012 |
| PE laplacian | 1 | 0.103±0.005 | 0.109±0.003 | 0.111±0.001 | 0.114±0.012 |
| PE shuffled_fixed | 1 | 0.102±0.006 | 0.110±0.009 | 0.110±0.005 | 0.108±0.009 |
| feature | 7 | 0.068±0.006 | 0.069±0.007 | 0.083±0.006 | 0.079±0.006 |
| PE laplacian | 7 | 0.079±0.013 | 0.078±0.009 | 0.077±0.007 | 0.076±0.008 |
| PE shuffled_fixed | 7 | 0.081±0.003 | 0.072±0.003 | 0.074±0.016 | 0.075±0.004 |

(C3 similar, all null.) Verdicts: (1) **the receptive-field hypothesis is refuted** — laplacian −
shuffled_fixed is ~0 at every depth, including L=1 where an undersized receptive field should have
made real coordinates valuable; (2) at C7/L1 BOTH PE arms beat feature (+0.012, seed ranges
separate) — the any-code capacity effect again, not structure; (3) side observation: uci needs only
1–2 MP layers (feature L1 0.119±0.003 ≈ L8 0.113±0.012); the 8-layer Table-3 config is kept for
parity only. The depth axis is CLOSED.

### 10.10 Conversion, not content — C1 answered, then REOPENED federated  **[the live result, 2026-07-25]**

Program scoping (user, 2026-07-25): decoder + fusion axes closed; Laplace smodel sidelined. The one
question the spectral data does NOT answer: **the exact basis is demonstrably informative, yet
nothing converts that information into ranking performance beyond parity. Why?**

The encoder x basis matrix has exactly one untested cell — every SignNet run to date consumed the
numerically-empty Arnoldi basis (`get_spectral_features` routes SignNet through `estimate=True`):

| | Arnoldi basis (empty, §10.6) | exact basis (informative, §10.6) |
|---|---|---|
| Laplace smodel | run — null/harmful (§10.2) | never run; smodel now sidelined |
| SignNet smodel | run — parity + capacity (§8) | **NEVER RUN — the open cell** |
| plain input PE | never run | run — parity, +0.008 uci (§10.7) |

Candidate explanations, ranked:
1. **Marginal redundancy with the trained backbone** (leading; unmeasured): the oracle's 0.71 AUC is
   unconditional; the trained fmodel reaches ~0.90. The decision-relevant quantity is CONDITIONAL
   information — does exact-spectral affinity separate pairs GIVEN the fmodel's z? 8 MP layers + a
   per-node GRU state may span the useful spectral projection. If conditional gain ≈ 0, parity is a
   CEILING no encoder can pass, and the program ends with a complete mechanism.
2. **Feature competition under SGD** (gradient starvation / shortcut learning): the learned rank
   collapse (§10.3) proves the dynamic operates here; §10.8's late-window uci effect (+0.05 in the
   final third) fits a slowly-earned pathway.
3. **Metric mismatch**: the spectral niche (AA-blind, community-level pairs) may barely move
   live-update MRR, which recency/degree dominate.

**Next step — the conditional-information probe [zero GPU, local]:** capture the trained
feature-model's per-snapshot eval embeddings z; score t+1 candidates with (a) a z-based scorer and
(b) z + exact-spectral affinity; compare AUC/MRR overall and on the AA-blind slice. Discriminates
1 from 2/3 before any sweep. **SignNet x exact** (via a `spectral.solver` knob) runs only if the
probe shows headroom — scoped to bitcoin_otc + uci, recompute, laplacian vs shuffled_fixed, C{3,7}.

**RESOLVED (2026-07-25) — the probe was run, and the question is answered.** Two stages, uci C1,
3 seeds, prequential (fit on past snapshots only), fixed candidates across seeds:
- Stage 1 (probe readouts over captured eval-time z): spectral affinity adds +0.04–0.09 AUC over
  z-feature probes — but those probes (0.73–0.79 AUC) are weaker than the model's own head (0.90),
  so the delta conflates headroom with readout weakness.
- Stage 2 (decisive — baseline = the MODEL'S OWN trained scores, captured via a decode hook at
  every eval): model alone **0.9011±0.0007**; model + exact-spectral affinity (2-feature
  prequential logistic) **0.9094±0.0010**; **Δ = +0.0083±0.0005** (per-seed +0.0077/+0.0083/
  +0.0089); AA-blind slice Δ = +0.0070±0.0007. The tightest measurement in this file.

**Verdict — hypothesis 1 confirmed in refined form: near-total marginal redundancy.** The exact
basis's unconditional 0.71 AUC shrinks to **+0.008 conditional** on what the trained backbone
already represents — and §10.7's input-PE realized **+0.008** MRR on uci. The pipeline did not
fail to convert the spectral information; it converted essentially ALL of the conditionally
available information, which is simply small. Parity+ε IS the ceiling, and the implementation
saturates it. Consequences: (a) **SignNet x exact is no longer justified** — the probe bounds any
encoder's possible gain at the same ~+0.008 (an encoder cannot exceed the conditional information
of its input); (b) the spectral thread now ends affirmatively, with a measured
information-accounting: content real (0.71) -> conditionally novel (+0.008) -> realized (+0.008).
Scope caveats: uci C1, random negatives, cosine affinity over the exact low-50 basis; replicating
the probe on bitcoin_otc (where f+pe sat BELOW its ceiling candidate) is the one cheap follow-up
that could still teach something.

**REOPENED BY THE FEDERATED EXTENSION (2026-07-25, same day) — the C1 verdict was an artifact of
centralization.** User objection, correct: at C=1 message passing can compute everything the
low-frequency eigenbasis encodes (MP ~ graph smoothing), so redundancy there is near-tautological;
the paper's hypothesis was always that the GLOBAL basis substitutes for the cross-client message
passing that sharding FORBIDS. The probe was extended to C{1,3,7,9} with, per snapshot, a count of
the edges whose endpoints fall in different clients (= messages a centralized GNN would pass that
FL cannot). uci, 3 seeds, identical global candidates across C, model-score baseline:

| C | cut fraction | lost edges/snap | model alone | model+spec | Δ (conditional ceiling) | Δ AA-blind |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0 | 0.9027±0.0008 | 0.9098±0.0015 | +0.0071±0.0021 | +0.0066±0.0024 |
| 3 | 0.665±0.006 | 464 | 0.8604±0.0040 | 0.8820±0.0034 | **+0.0216±0.0010** | +0.0231±0.0005 |
| 7 | 0.856±0.002 | 594 | 0.7837±0.0116 | 0.8277±0.0095 | **+0.0440±0.0071** | +0.0464±0.0157 |
| 9 | 0.895±0.004 | 620 | 0.7421±0.0328 | 0.8095±0.0089 | **+0.0674±0.0241** | +0.0697±0.0207 |

Three findings:
1. **The federated mechanism is CONFIRMED.** The conditional ceiling grows ~10x from C1 to C9,
   monotone in the measured cut fraction (which itself matches the 1−1/C random-partition theory).
   The global exact basis carries precisely the information the severed message-passing edges would
   have provided — the paper's original premise, validated at the information level.
2. **The federated conversion gap is REAL.** Realized f+pe gains (§10.7: uci C3 +0.015, C7 ~0)
   sit far below the C3/C7 ceilings (+0.022/+0.044) — embedding-level injection fails to convert
   exactly where the information matters most. The C1 "saturation" was the exception, not the rule.
3. **The probe demonstrates a working converter.** Its own readout — a prequential 2-feature
   logistic over [model score, spectral affinity], i.e. SCORE-LEVEL late fusion — realizes the
   ceiling by construction: at C9 it recovers ~40% of the AUC that sharding destroyed
   (0.742 -> 0.810 of the centralized 0.903). Every failed implementation fused at the input or
   embedding level, where training can absorb or ignore the signal; fusing at the DECISION level
   cannot be absorbed.

**MRR-STYLE READOUT + PLACEBO CONTROL — both PASSED (2026-07-25, same day).** Each t+1 positive
ranked against 200 corrupted-endpoint negatives (excluded: true t+1 edges); fusion = prequential
logistic over [model score, spectral affinity]; placebo = the same fusion but with a FIXED
node-row permutation of the same normalized eigenvectors (structure destroyed, numerics and drift
identical), with its own independently fitted weights. uci, 3 seeds:

| C | MRR model | Δ MRR real fusion | Δ MRR placebo fusion | MRR spec alone |
|---|---|---|---|---|
| 1 | 0.1835±0.0019 | **+0.0745±0.0032** | −0.0002±0.0001 | 0.1513 |
| 3 | 0.1455±0.0152 | **+0.0799±0.0103** | +0.0002±0.0003 | 0.1513 |
| 7 | 0.1258±0.0099 | **+0.0717±0.0059** | −0.0001±0.0004 | 0.1513 |
| 9 | 0.1135±0.0046 | **+0.0727±0.0029** | −0.0006±0.0013 | 0.1513 |

Reads:
1. **The fusion gain survives the ranking metric and beats its placebo by construction-grade
   margins**: +0.072–0.080 probe-MRR at every C (~40–70% relative), placebo pinned at zero
   (the prequential fit learns to ignore the useless feature — the honest null behavior).
2. **Two separable effects confirmed**: the C-growing AUC ceiling (cross-shard information; §10.10
   table above) and a C-INDEPENDENT top-rank disambiguation effect — spectral affinity resolves
   which of the model's plausible top candidates is the true partner, worth little in
   threshold-averaged AUC at C1 (+0.009) but a lot in MRR (+0.075). This also retro-explains
   §10.9: the concat-MLP head underuses pair affinity at the top of the ranking, and embedding-level
   injection (§10.7, +0.008) converted almost none of what score-level fusion reaches.
3. **Raw spectral affinity ALONE (MRR 0.151) out-ranks the entire trained federated model at C≥7**
   (0.126/0.114) — the single most striking number in the file: at high sharding, a training-free
   eigenvector cosine beats the full federated ROLAND pipeline on the ranking metric.
4. Caveats that remain before this is a paper claim: (a) probe-MRR (200 corrupted-endpoint
   negatives) is not the codebase's reported protocol (1000-multiplier, test-split positives,
   `mrr_method=max`) — the fusion must be validated INSIDE the real eval; (b) uci only so far;
   (c) fusion weights are prequential-fit — an in-model equivalent (per-snapshot validation-fitted
   λ, or a learned scalar) must reproduce them without leakage.

**GATE 1 — THE FUSION INSIDE THE REAL EVAL PROTOCOL: PASSED (2026-07-25).** The probe-MRR above
uses 200 corrupted-endpoint negatives; the reported metric does not. `analysis/probes/proto_fusion.py`
replaces `compute_mrr_from_z` in the eval path with a copy that is identical in RNG consumption and
return value — verified per run: the probe's own model mean equals the run's reported `mean_mrr` to
4 decimals on all 12 runs — and additionally ranks the SAME candidate set with fused scores. So the
readout below IS the codebase's protocol: `rank_eval_multiplier=1000` negatives per source,
**test-split** positives of t+1, `mrr_method=max`, per-source aggregation, global stitched z.

Two leakage-free weightings, each with its own independently fitted placebo arm (`shuffled_fixed`:
one fixed node-row permutation of the same normalized exact low-50 eigenvectors):
- **preq** — logistic over [model score, spectral affinity] fit on PAST snapshots only (features
  z-scored per snapshot); applied to the reported (train-mode) scores.
- **val** — logistic fit on the CURRENT snapshot's **val-split** edges + 50 sampled negatives each,
  scored in eval mode; applied to eval-mode test scores. This is the one an in-model λ can copy.

uci, gru, feature model, C{1,3,7,9} x 3 seeds; means over the 22 of 27 snapshots where both
weightings are defined (5-snapshot prequential warm-up), the model column on the same window:

| C | MRR model | Δ preq real | Δ preq placebo | Δ val real | Δ val placebo | MRR spec alone | λ val |
|---|---|---|---|---|---|---|---|
| 1 | 0.1292±0.0171 | **+0.0326±0.0124** | −0.0044±0.0014 | **+0.0307±0.0161** | −0.0073±0.0032 | 0.0760±0.0018 | +0.76 |
| 3 | 0.0960±0.0098 | **+0.0406±0.0009** | −0.0017±0.0019 | **+0.0357±0.0043** | −0.0054±0.0043 | 0.0718±0.0029 | +1.12 |
| 7 | 0.0801±0.0167 | **+0.0385±0.0145** | −0.0021±0.0013 | **+0.0257±0.0186** | −0.0063±0.0044 | 0.0688±0.0049 | +1.82 |
| 9 | 0.0751±0.0103 | **+0.0331±0.0043** | +0.0002±0.0015 | **+0.0255±0.0083** | −0.0049±0.0024 | 0.0693±0.0010 | +2.55 |

Reads:
1. **The reported metric moves.** +0.026–0.041 MRR (+25–53% relative) at every C, under both
   leakage-free weightings, while both placebos sit at −0.007..+0.000. The finding survives the
   protocol change; the effect is smaller than probe-MRR's +0.072–0.080 but the same phenomenon.
2. **The val-fitted weight nearly matches the prequential one** (−0.005..−0.013 vs preq), so the
   in-model version needs no history: one snapshot's val edges suffice to set λ.
3. **λ grows with sharding** (+0.76 → +2.55 from C1 to C9): the fit leans harder on spectral
   affinity exactly as federation severs more message-passing edges — the same direction as the
   conditional-ceiling curve, now visible in the fitted weights.
4. **The realized MRR gain is C-INDEPENDENT** (+0.033/+0.041/+0.039/+0.033), consistent with the
   top-rank-disambiguation effect rather than the C-growing cross-shard AUC ceiling. As a fraction
   of what sharding costs (C1→C9 model loses 0.054) it recovers 61% at C9 and 78% at C7 — but it
   also gains the same amount at C1, so this is a decoder-level gain, not a federated repair.
   The federated argument rests on the AUC ceiling table, not on this row.
5. **CORRECTION to the probe-MRR reading (§10.10 point 3): spectral affinity alone does NOT
   out-rank the trained model in-protocol** — 0.069–0.076 vs the model's 0.075–0.129; at C9 it is
   close (0.069 vs 0.075) but still below. That claim was an artifact of the 200-negative
   corrupted-endpoint probe and must not be carried into the paper.

Remaining caveats: uci only (bitcoin_otc replication is gate 2); the fusion is applied at the eval
readout, not learned inside the model (gate 3); λ is fitted per snapshot on val edges, which an
edge-score smodel would have to reproduce.

**GATE 2 — bitcoin_otc: the MECHANISM replicates, the CONVERSION does not (2026-07-25).** Both
probes re-run on bitcoin_otc (261 snapshots, 5881 nodes, gru, feature model, C{1,3,7,9} x 3 seeds,
sim13; ~50 s/run — the earlier 35 min/run estimate was for in-model f+pe runs, not these).

_Conditional-information ceiling (`cond_probe3.py`), model-score baseline, exact Q50:_

| C | cut fraction | lost edges/snap | model alone | model+spec | Δ (ceiling) | Δ AA-blind |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0 | 0.9499±0.0004 | 0.9536±0.0004 | +0.0037±0.0002 | +0.0055±0.0000 |
| 3 | 0.665±0.001 | 61.4 | 0.8767±0.0351 | 0.8963±0.0187 | +0.0197±0.0192 | +0.0107±0.0033 |
| 7 | 0.856±0.003 | 78.8 | 0.7594±0.1127 | 0.8077±0.0206 | +0.0483±0.0922 | +0.0306±0.0535 |
| 9 | 0.891±0.002 | 81.7 | 0.6765±0.1005 | 0.7635±0.0317 | +0.0870±0.0708 | +0.0443±0.0349 |

_In-protocol fusion (`proto_fusion.py`), same protocol as gate 1, 182 of 261 snapshots in window:_

| C | MRR model | Δ preq real | Δ preq placebo | Δ val real | Δ val placebo | MRR spec alone | λ preq | λ val |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1939±0.0009 | −0.0022±0.0025 | −0.0011±0.0003 | −0.0211±0.0025 | −0.0203±0.0043 | 0.0450±0.0036 | −0.02 | +0.44 |
| 3 | 0.1504±0.0068 | −0.0024±0.0039 | −0.0060±0.0013 | −0.0172±0.0021 | −0.0228±0.0027 | 0.0436±0.0017 | −0.04 | +0.90 |
| 7 | 0.1280±0.0039 | −0.0091±0.0064 | −0.0002±0.0022 | −0.0146±0.0039 | −0.0174±0.0007 | 0.0444±0.0025 | +1.20 | +4.54 |
| 9 | 0.1055±0.0087 | −0.0123±0.0020 | −0.0134±0.0062 | −0.0087±0.0053 | −0.0198±0.0046 | 0.0432±0.0025 | +3.56 | +7.34 |

Reads:
1. **The federated ceiling replicates on a second dataset**: +0.004 / +0.020 / +0.048 / +0.087 AUC
   against uci's +0.007 / +0.022 / +0.044 / +0.067, on independently measured cut fractions that
   again match 1−1/C. The mechanism claim (the global basis carries what severed cross-client
   message passing would compute) is no longer uci-only. Caveat: at C7/C9 the seed spread is as
   large as the effect (model-alone AUC itself swings ±0.11 at C7, n=3) — directionally confirmed,
   not tightly measured.
2. **The conversion does NOT replicate.** In-protocol fusion is null-to-negative at every C, and
   crucially **real ≈ placebo** (differences ≤0.011, both signs) — the honest reading of the
   negative Δs is that the refit slightly miscalibrates the model score, identically with and
   without real structure. No spectral gain exists here to convert.
3. **λ still climbs with sharding** (−0.02 → +3.56 preq; +0.44 → +7.34 val) without buying any MRR:
   the fit reaches for spectral affinity as more edges are severed, but on bitcoin_otc that
   affinity carries no ranking-relevant structure — spec alone is 0.043–0.045 versus uci's
   0.069–0.076 against a *stronger* model (0.194 vs 0.129 at C1).
4. **This is the §10.7 dataset split again, now at the decision level** (input PE: uci +0.008,
   bitcoin_otc −0.011). The asymmetry therefore belongs to the SIGNAL — bitcoin_otc's exact
   low-50 basis is weak for future-link ranking — not to the injection point. Score-level fusion
   fixes the *conversion* problem, not the *content* problem.

**BREADTH + BASIS-SIZE SENSITIVITY (2026-07-25, sim13).** Two competing readings of the uci /
bitcoin_otc split had to be separated before building anything: (a) the basis is too SMALL on the
bigger graphs — K=50 was chosen for uci's 1899 nodes, i.e. K/N ≈ 2.6%, versus 0.85% on
bitcoin_otc; (b) the fusion only helps where the BACKBONE is weak (uci's model is the weakest in
the file). `proto_fusion.py` on bitcoin_alpha and on K ∈ {50, 100, 200(, 300)}, all C{1,3,7,9} x
3 seeds. Columns: Δ = mean over C of the prequential fusion delta, net = real − placebo:

| dataset | N | K | Δ real (mean over C) | Δ placebo | **net** | MRR spec alone | model MRR (C1) |
|---|---|---|---|---|---|---|---|
| uci | 1899 | 50 | +0.036 | −0.002 | **+0.038** | 0.070 | 0.129 |
| uci | 1899 | 100 | +0.050 | +0.004 | **+0.046** | 0.091 | 0.119 |
| uci | 1899 | 200 | +0.050 | +0.004 | **+0.046** | 0.104 | 0.132 |
| bitcoin_otc | 5881 | 50 | −0.007 | −0.005 | −0.001 | 0.044 | 0.194 |
| bitcoin_otc | 5881 | 100 | −0.010 | −0.006 | −0.004 | 0.044 | 0.192 |
| bitcoin_otc | 5881 | 200 | −0.013 | −0.005 | −0.008 | 0.044 | 0.191 |
| bitcoin_otc | 5881 | 300 | −0.010 | −0.006 | −0.004 | 0.042 | 0.193 |
| bitcoin_alpha | 3783 | 50 | +0.004 | −0.003 | +0.007 | 0.039 | 0.166 |
| bitcoin_alpha | 3783 | 150 | −0.008 | −0.004 | −0.004 | 0.037 | 0.165 |
| as733 | 7716 | 50 | _running (4 jobs, C1/3/7/9 x 3 seeds, ~1 h per run)_ | | | | |

1. **Hypothesis (a) is refuted for bitcoin, and confirmed for uci.** On uci a bigger basis carries
   strictly more signal — spectral affinity alone climbs 0.070 → 0.091 → 0.104 and the net fusion
   gain grows +0.038 → +0.046 (at K=200, C9 it is +0.060±0.008 on a 0.060±0.005 model, i.e. the
   MRR doubles). On both bitcoin graphs spectral affinity alone is FLAT in K (0.044/0.044/0.044/
   0.042 at K=50/100/200/300, and 0.039/0.037) — a 6x bigger basis adds nothing, so K=50 never was
   the binding constraint there. §10.10's per-C uci table (K=50) therefore UNDERSTATES the effect; K should scale with the
   graph, and 50 was undersized even for uci.
2. **Hypothesis (b) is refuted.** bitcoin_alpha's backbone is as weak as uci's at high sharding
   (0.089–0.098 at C7/C9 vs uci's 0.075–0.088) and the fusion still gains nothing there. What
   separates the datasets is the CONTENT of the basis: spectral affinity ranks future partners on
   uci (0.070–0.104, within reach of the model's own 0.06–0.13) and barely at all on the bitcoin
   graphs (0.037–0.045 against a 0.17–0.19 model).
3. Reading: this is a **communication-graph vs rating-graph** distinction, not a size, capacity or
   basis-resolution one. Who messages whom next week is community-structured, so low-frequency
   eigenvectors predict it; who rates whom next is not.
4. Bookkeeping caveat: the model column moves by up to 0.019 between batches of the SAME condition
   (uci C9: 0.075 / 0.079 / 0.060 across the K batches, 3-seed means each). Only the within-run
   deltas are trustworthy; never compare model columns across batches.

**THE PERSISTENCE CONTROL — the uci gain is EDGE MEMORY, not spectral structure (2026-07-25,
supersedes the gate-1 interpretation).** Prompted by the user's question ("could the bitcoin
datasets just be much sparser?"), the datasets' basic statistics were measured — and the decisive
difference is not density but **edge RECURRENCE**:

| dataset | avg degree (cum.) | median degree | share of next-snapshot edges that ALREADY exist |
|---|---|---|---|
| uci | 14–15 | 6 | **0.52–0.56** |
| bitcoin_otc | 5.5–7.3 | 2 | 0.10–0.20 |
| bitcoin_alpha | 5.4–7.5 | 2 | 0.07–0.10 |

Cosine affinity in a low-frequency eigenbasis is an excellent detector of pairs that are ALREADY
connected in the graph the basis was built from. On uci half the test positives are exactly that;
on the bitcoin graphs almost none are. So `proto_fusion.py` gained two controls: (i) the same
prequential fusion with a **trivial 1-bit `exists` feature** (is the pair already an edge of the
cumulative union?) and with **`cn`** (log common neighbours), and (ii) every arm reported split by
REPEAT vs genuinely NEW positives. uci, exact Q100, 3 seeds:

| C | MRR model | Δ preq spec | Δ preq placebo | **Δ preq exists** | Δ preq cn |
|---|---|---|---|---|---|
| 1 | 0.1279±0.0043 | +0.0519±0.0108 | +0.0050±0.0059 | **+0.1740±0.0137** | −0.0456±0.0033 |
| 3 | 0.0844±0.0131 | +0.0655±0.0109 | +0.0022±0.0031 | **+0.1719±0.0071** | −0.0191±0.0017 |
| 7 | 0.0850±0.0130 | +0.0456±0.0243 | +0.0012±0.0018 | **+0.1543±0.0244** | −0.0214±0.0070 |
| 9 | 0.0735±0.0121 | +0.0475±0.0187 | +0.0029±0.0019 | **+0.1446±0.0190** | −0.0162±0.0047 |

| C | repeat frac | model REP | Δ spec REP | Δ exists REP | model NEW | Δ spec NEW | Δ exists NEW |
|---|---|---|---|---|---|---|---|
| 1 | 0.488 | 0.167 | +0.124±0.014 | +0.340±0.019 | 0.072 | −0.037±0.005 | −0.039±0.003 |
| 3 | 0.488 | 0.109 | +0.137±0.006 | +0.329±0.005 | 0.048 | −0.022±0.010 | −0.025±0.010 |
| 7 | 0.488 | 0.102 | +0.113±0.032 | +0.307±0.028 | 0.061 | −0.038±0.015 | −0.039±0.015 |
| 9 | 0.488 | 0.088 | +0.108±0.023 | +0.287±0.027 | 0.048 | −0.024±0.004 | −0.027±0.004 |

1. **A 1-bit lookup beats the eigenbasis by ~3x at every C** (+0.145..+0.174 vs +0.046..+0.066).
   Whatever the spectral feature contributes, a trivially cheaper feature contributes more.
2. **The entire gain lives on REPEAT positives** (+0.11..+0.14 spec, +0.29..+0.34 exists). On
   genuinely NEW pairs BOTH arms are NEGATIVE (−0.02..−0.04): the fused score is worse than the
   model alone at ranking pairs that have never interacted.
3. Therefore the spectral affinity is acting as a **noisy proxy for "these two are already
   connected"**, i.e. graph MEMORY, not community structure predicting the future. That explains
   the dataset split exactly (uci 49% repeats vs bitcoin 7–20%) — and it explains why K helped on
   uci (a bigger basis resolves existing edges more sharply) but never on bitcoin.
4. **The placebo control was necessary but not sufficient.** `shuffled_fixed` excludes "any
   feature with the same numerics"; it cannot distinguish structure from trivial memory. Any
   future spectral claim in this project must ALSO beat the `exists` baseline, and must be
   reported split by repeat/new.
**DENSITY, TESTED CAUSALLY (`thin`).** The sparsity question was then answered directly rather than
by correlation: the eigenbasis is solved on a randomly THINNED copy of the cumulative graph while
the model, the task and the `exists` feature keep the full graph. uci, K=100, 3 seeds:

| thin | avg degree | Δ preq spec (C1) | Δ preq spec (C7) | spec alone | λ preq (C1) | Δ preq exists (C1) |
|---|---|---|---|---|---|---|
| 1.00 | ~14.6 | +0.0519±0.0108 | +0.0456±0.0243 | 0.099 | +0.60 | +0.1740±0.0137 |
| 0.50 | ~7.3 | +0.0283±0.0101 | +0.0366±0.0128 | 0.073 | +0.35 | +0.1656±0.0147 |
| 0.33 | ~4.8 | +0.0056±0.0059 | +0.0186±0.0162 | 0.058 | +0.29 | +0.1622±0.0123 |

**Sparsity is causally sufficient to destroy the effect.** Thinned to bitcoin-like density
(0.33 → avg degree 4.8 vs bitcoin's 5.5–7.3), uci's spectral gain collapses from +0.052 to
+0.006 — i.e. to bitcoin's null — with the fitted λ falling 0.60 → 0.29. The `exists` arm is
untouched by thinning (it reads the full graph), which confirms the manipulation hit only the
basis. The repeat-split shows the same dose-response (Δ spec on REPEAT positives: +0.124 → +0.087
→ +0.039), tying the two findings together: **the low-frequency eigenbasis stores a smoothed
memory of the edges it was built from, and a sparse graph leaves too little of that memory in the
low-frequency subspace.** Bitcoin loses twice over — a sparser basis AND far fewer repeat test
edges to spend it on.

5. **What the fused features actually inject is HISTORY, and this reframes all of §10.** Snapshots
   are disjoint calendar windows (`_temporal.py::split_by_calendar` slices `g_all` by period), so
   the backbone's message passing at time t sees ONLY window t's edges — everything older survives
   solely in the compressed GRU state. The spectral basis, by contrast, has always been solved on
   the CUMULATIVE union up to t (`_spectral_step`), and so has `exists`. The fused arms therefore
   carry the full edge history EXPLICITLY, which the recurrent state only summarises. Two
   consequences: (a) the honest statement of the backbone finding is not "it ignores persistence
   it already has" but "an explicit 1-bit history lookup retains what its recurrent state loses,
   worth +0.15 MRR"; (b) **the `shuffled_fixed` placebo could never have caught this** — permuting
   node rows destroys history and structure together, so every placebo-validated spectral result
   in §10 was validated against a control that removed BOTH. Separating them needs a positive
   baseline that keeps history and discards structure, which is exactly what `exists` is.
6. Federated reading: at C>1 the global basis injects history AND cross-client edges the client
   never sees. The conditional-AUC ceiling grows with C (the cross-client part), while the MRR
   gain is C-independent (the history part) — the two components the earlier §10.10 tables saw
   separately, now with a mechanism for each.

**bitcoin_otc under the same controls — the mechanism is IDENTICAL, only the opportunity differs
(2026-07-25).** Same probe, exact Q100, 3 seeds; repeat fraction 0.081:

| C | MRR model | Δ preq spec | Δ preq exists | model REP | Δ spec REP | Δ exists REP | model NEW | Δ spec NEW | Δ exists NEW |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1913±0.0035 | −0.0036±0.0016 | −0.0075±0.0022 | 0.221 | +0.133±0.020 | +0.356±0.013 | 0.186 | −0.016±0.004 | −0.041±0.003 |
| 3 | 0.1517±0.0066 | −0.0079±0.0049 | −0.0027±0.0032 | 0.180 | +0.173±0.028 | +0.428±0.024 | 0.148 | −0.025±0.004 | −0.043±0.003 |
| 7 | 0.1279±0.0082 | −0.0162±0.0057 | +0.0066±0.0036 | 0.162 | +0.169±0.009 | +0.428±0.015 | 0.123 | −0.033±0.006 | −0.034±0.001 |
| 9 | 0.0962±0.0186 | −0.0121±0.0081 | +0.0071±0.0064 | 0.150 | +0.147±0.018 | +0.421±0.023 | 0.091 | −0.027±0.009 | −0.032±0.009 |

**On the repeat subset bitcoin_otc gains as much as uci — MORE, in fact** (+0.15..+0.17 spec,
+0.36..+0.43 exists). Its overall null is not a different mechanism; it is the same mechanism with
almost nothing to act on (8% repeats vs uci's 49%), and the small penalty the fusion imposes on the
92% new pairs (−0.02..−0.04) cancels it. This closes the account:

> **Decision-level fusion — spectral or trivial — re-ranks pairs that ALREADY have an edge, and
> slightly damages pairs that do not. The overall gain is set by how many of a dataset's future
> edges are recurrences, not by any property of the spectrum.**

**The resulting predictive model, and its PRE-REGISTERED test.** Both controls point at one
account: `gain(spectral) ≈ f(repeat rate x basis density)` and `gain(exists) ≈ f(repeat rate)`.
Measured statistics for the remaining datasets (CPU-only, no training):

| dataset | N | avg degree | median degree | repeat fraction |
|---|---|---|---|---|
| uci | 1899 | 14.6 | 6 | 0.49 |
| bitcoin_otc | 5881 | 7.3 | 2 | 0.10–0.20 |
| bitcoin_alpha | 3783 | 7.5 | 2 | 0.07–0.10 |
| as733 | 7716 | 5.6 | 2 | **0.997** |
| reddit_body | 35776 | 6.9 | 1 | 0.52 |
| reddit_title | 54075 | 8.1 | 1 | 0.58 |

Predictions stated BEFORE the runs: **as733** — near-total recurrence but a sparse basis, so a
very large `exists` gain and only a small spectral one; **bitcoin_otc** — low recurrence, so a
small `exists` gain; **reddit_*** — uci-like recurrence on a sparser graph, so a large `exists`
gain and a modest spectral one.

**as733 RESULT — recurrence dominates; the "sparse basis ⇒ small spectral gain" half of the
prediction was WRONG (2026-07-26).** as733 C9, 3 seeds, exact Q50 (this run predates the `exists`
arm, so it carries the spectral/placebo columns only):

| C | MRR model | Δ preq spec | Δ preq placebo | MRR spec alone | λ preq | λ val |
|---|---|---|---|---|---|---|
| 9 | 0.2761±0.0073 | **+0.2903±0.0051** | −0.0013±0.0024 | **0.4173±0.0018** | +1.79 | +3.23 |

The fusion **more than doubles** the reported MRR (0.276 → 0.566) with the placebo at zero, and
spectral affinity ALONE (0.417) out-ranks the trained federated model (0.276) by a wide margin.
On a graph where 99.7% of tomorrow's edges already exist, "score pairs by their proximity in the
eigenbasis of today's cumulative graph" is close to the optimal strategy — which is precisely the
memory account, at its maximum. What the prediction got wrong is that sparsity would blunt it:
as733 is as sparse as bitcoin (avg degree 5.6) yet shows the largest gain in the file, so
recurrence dominates density rather than the two multiplying. The persistence baseline is
therefore the outstanding question for as733 — `runs/run_gate2g.sh` re-runs C1 and C9 with the
`exists`/`cn` arms; the account predicts `exists` ≥ `spec` there too.
(Ops note: the first 4-way as733 launch OOMed the 4080 — as733's GPU peak grows with the
cumulative graph. C9 survived and completed; the C1/C7 retries then OOMed against it. Cap as733
at 2 concurrent jobs. Each run is ~40 min.)

Status: the mechanism is confirmed and twice-replicated; decode-time score fusion is
placebo-validated, survives the real eval protocol, and STRENGTHENS with basis size — **on uci**
(net +0.046 at K=100/200, up to a doubled MRR at C9) — while both bitcoin graphs are null at every
K tested. The effect is therefore **dataset-conditional**, and, usefully, its precondition is
measurable BEFORE building anything: spectral affinity alone must rank future partners within
reach of the trained model (uci 0.070–0.104 vs bitcoin 0.037–0.045). That reframes gate 3 (the
edge-score smodel + federated sweeps with placebo arms) as a **conditional method plus its own
go/no-go diagnostic**, not a universal one. Outstanding: the as733 cell (running); reddit_* (not
attempted — 35k nodes makes the exact basis expensive); and, if gate 3 proceeds, `spectral.pe_dim`
must be re-tuned per dataset since 50 is undersized.

**Program conclusion — SUPERSEDED on the fusion axis (2026-07-25).** Written before gates 1–2; it
remains correct for EMBEDDING-level injection (§10.2–10.7) and for the bitcoin/rating graphs at any
K, but "not a reliable lever" is now too strong for uci, where DECISION-level fusion adds up to
+0.06 MRR in-protocol against a null placebo. Read it as the verdict on the implementations tried
before score fusion. Original text: Output-side fusion transported nothing (§10.2–10.4);
the consumed basis carried nothing to transport (§10.6); and with BOTH repaired — an informative
exact basis, injected at the input, un-throttleable — the structural gain is small-and-real on the
one dataset where the oracle predicted it, absent or negative elsewhere. Spectral graph structure is
**not a reliable lever for ROLAND-style temporal link prediction on these datasets**; its predictive
content overlaps what the recurrent message-passing backbone already captures, and the residual
complementary niche (§10.6 point 2) is too small to survive federated noise. The paper's positive
result remains §9 (federation's resilience); the spectral thread closes as a controls-validated
negative with a fully measured causal chain.

### 10.11 The uniform probe matrix — the recurrence law, and the one place structure survives  **[2026-07-26]**

Everything above was assembled from probes of different vintages. This is the whole matrix re-run
under ONE probe version (`proto_fusion.py` with the `exists`/`cn` baselines, the repeat/new split
and the shared basis cache), C{1,3,7,9} x 3 seeds, distributed over seven cluster hosts.
K=100 for uci/bitcoin_*, K=50 for as733/reddit_* (cost). Δ = prequential fusion minus model, on
the reported protocol. `net` = real − placebo. Per-cell mean±std over 3 seeds is in the logs; the
ranges below span C1→C9.

| dataset | repeat frac | Δ spec (overall) | Δ **exists** (overall) | Δ spec on REPEATS | Δ spec on **NEW** | Δ exists on NEW | placebo |
|---|---|---|---|---|---|---|---|
| uci | 0.488 | +0.040…+0.063 | **+0.152…+0.168** | +0.10…+0.13 | **−0.022…−0.042** | −0.024…−0.044 | ~0 |
| bitcoin_alpha | 0.084 | −0.011…−0.000 | −0.009…+0.010 | +0.14…+0.18 | −0.015…−0.026 | −0.029…−0.044 | ~0 |
| bitcoin_otc | 0.081 | −0.019…−0.006 | −0.008…+0.005 | +0.15…+0.18 | −0.020…−0.034 | −0.035…−0.044 | ~0 |
| reddit_body | 0.551 | +0.085…+0.124 | **+0.187…+0.276** | +0.14…+0.20 | **+0.022…+0.034** | −0.025…−0.032 | ~0.000 |
| reddit_title | 0.604 | +0.038…+0.042 | **+0.192…+0.212** | +0.067…+0.070 | **+0.005…+0.007** | −0.036…−0.037 | ~−0.0007 |
| as733 (C1, C9) | 0.952 | +0.294…+0.297 | **+0.499…+0.562** | +0.257…+0.292 | **+0.368…+0.746** | −0.012…−0.017 | −0.002…−0.010 |

**1. The recurrence law.** The per-subset effects are nearly dataset-independent: every dataset
gains a lot on REPEAT positives (spec +0.10..+0.20, exists +0.30..+0.52) and the overall number is
just that gain diluted by the repeat fraction. Order the datasets by recurrence and the overall
`exists` gain follows monotonically (0.08 → ~0.00, 0.49 → +0.16, 0.55 → +0.23). Nothing spectral
is required to explain any overall number.

**2. reddit_body is the one genuine exception, and it is exactly where the federated theory said
to look.** On **genuinely NEW pairs** — pairs that have never interacted, where the `exists`
feature is useless by construction and indeed HURTS (−0.025..−0.032) — the spectral term is
**positive (+0.022 → +0.034), grows monotonically with sharding C1→C9, and its placebo is
−0.0001±0.0002**, i.e. exactly zero at three-seed precision. This is the first evidence in the
whole program that the eigenbasis predicts links it has not already seen, it beats the trivial
baseline where the trivial baseline cannot follow, and its growth with C matches the
conditional-ceiling mechanism (§10.10): the more message passing federation severs, the more the
global basis is worth. The effect is small in absolute terms (~+0.03 MRR on a 0.26–0.40 model)
but it is clean, monotone and placebo-null. **reddit_title (complete, C1/3/7/9) replicates the
SIGN but not the growth**: NEW-pair spec = +0.0071 / +0.0050 / +0.0070 / +0.0061 (±0.0006–0.0015),
placebo ~−0.0008, `exists` −0.036…−0.037 on the same subset. So on a second large communication
graph the spectral term is again the only feature that helps on unseen pairs — but the magnitude
is **1/5 of reddit_body's and it is FLAT in C, not growing**. Honest statement of the claim:
*the NEW-pair spectral gain replicates (2/2 large communication graphs, placebo-null, beats both
trivial baselines there); its growth with sharding does not (1/2)*. Anything written about
"federation makes the basis more valuable" currently rests on reddit_body alone.

**2b. as733 turns the exception into the headline.** With 95% recurrence its overall numbers are
dominated by `exists` (+0.50…+0.56) as the law predicts — but on the 5% of positives that are
genuinely NEW, spectral fusion adds **+0.37 at C1 and +0.75 at C9** against a model that scores
0.038/0.065 there, i.e. a ~10x improvement, while `exists` is NEGATIVE (−0.012/−0.017) and the
placebo is ~0. Spectral affinity ALONE (0.418) also out-ranks the trained model (0.269–0.338)
in-protocol — the claim retracted for uci is simply true here. And the gain nearly DOUBLES from
C1 to C9, the federated mechanism at full strength.

Ordering the NEW-pair effect across all six datasets gives a clean, interpretable gradient:
as733 (+0.37…+0.75) > reddit_body (+0.022…+0.034) > reddit_title (+0.005…+0.007) >
uci (−0.02…−0.04) ≈ bitcoin_alpha ≈ bitcoin_otc (−0.015…−0.034). That is exactly the order of
how much a graph's future is determined by its topology: AS peering (routing hierarchy) >
subreddit hyperlinks > social messaging > trust ratings. **The spectral basis predicts new links
where new links are structurally determined, and not otherwise** — and on the two datasets where
it works at all, the gain grows with sharding.

**3. Common neighbours (`cn`) is strong on reddit and weak elsewhere** (cn alone: 0.353 on
reddit_body vs 0.024 on uci), and on reddit it beats spectral affinity alone (0.353 vs 0.247) —
so on that dataset the honest baseline set for any structural claim is `exists` AND `cn`, not the
placebo alone.

**Consequence for gate 3.** An edge-score smodel is now justified on a specific, falsifiable
claim: *on large communication graphs, a decode-time spectral term improves NEW-pair ranking, and
the improvement grows with federation.* It must be evaluated on the NEW subset, against `exists`
and `cn`, on reddit-scale data — not on overall MRR, which recurrence dominates.

_Status: as733 (repeat 0.997) and reddit_title (0.575) still running; both will sharpen the law
and, for reddit_title, test whether the NEW-pair effect replicates on the second large
communication graph._

### 10.12 The spectrum is clustered, not degenerate — and a Chebyshev filter solves it  **[2026-07-26]**

**Diagnosis.** `analysis/probes/spectrum_stats.py [k] [datasets...]` measures the operator the
pipeline actually consumes (after the active-subgraph + largest-component truncation in
`Graph._active_lsym`). Two windows matter: **k=50** = `spectral.pe_dim`, the input-PE and probe
path; **k=300** = `spectral.spectral_len`, the **f+s smodel / tracking path** (§3–§8).

| dataset | window | λ₁ → λ_k | median gap | gaps<1e-3 | boundary gap λ_k→λ_{k+1} | boundary ÷ median |
|---|---|---|---|---|---|---|
| uci | 50 | 0.132 → 0.527 | 3.7e-3 | 3/50 | 4.3e-3 | 1.16 |
| uci | 300 | 0.132 → 0.759 | 9.2e-4 | 165/300 | 4.8e-4 | 0.53 |
| bitcoin_otc | 50 | 0.040 → 0.276 | 2.2e-3 | 8/50 | 1.9e-3 | 0.85 |
| bitcoin_otc | 300 | 0.040 → 0.480 | 7.4e-4 | 192/300 | 1.5e-4 | 0.22 |
| as733 | 300 | 0.022 → 0.422 | 7.4e-4 | 190/300 | 3.6e-4 | 0.49 |
| reddit_body | 300 | 0.018 → 0.221 | 3.3e-4 | 237/300 | 1.2e-4 | 0.36 |
| reddit_title | 300 | 0.017 → 0.197 | 2.6e-4 | 248/300 | 8.0e-4 | 3.04 |

Findings: (a) **structural degeneracy is real but already handled** — mid-run graphs have up to
**16,188 components** (reddit_body t=88) whose exact-zero eigenpairs would consume the entire
solver request; the truncation drops them (by the end of a run only 7–1065 remain). (b) After
truncation there is **exactly one zero eigenvalue and no gap below 1e-4** — the low spectrum is
**clustered, not degenerate**. (c) Crowding is universal (~1% relative spacing on every dataset)
and **3–4x worse in the k=300 tracking window** than at k=50. (d) The **boundary gap is not
protective**: at k=300 it is 0.1–0.5x the typical internal gap on five of six datasets, so the
300-dim subspace itself is not separated from what sits above the cut — `spectral_len=300` slices
through the densest part of the spectrum. Practical consequence: eigenvector-level identification
is ill-posed here (perturbation ÷ gap ≈ 1 for one snapshot's worth of new edges), which is what
the Procrustes step has been compensating for.

**The fix.** `Graph.calc_eigs_chebyshev` (same return contract as `calc_eigs_exact_sym`):
Jackson-damped Chebyshev expansion of an ideal low-pass over L_sym's range [0,2], applied to a
block by the three-term recurrence (one sparse matvec per degree, no dense matrix, no
factorisation), then QR + Rayleigh–Ritz. A filter never has to separate eigenvalues — it weights
by frequency — so clustering is irrelevant to it. `X0` warm-starts the block from the previous
snapshot's basis, making it a drop-in for the existing tracker.

_Validated against the exact solver on the same snapshots (`analysis/probes/cheb_validate.py`);
subspace overlap = ‖U_exactᵀ U‖²_F / k, recon AUC = the §10.6 oracle probe:_

| dataset | k | solver | seconds | subspace | max Δλ | recon AUC |
|---|---|---|---|---|---|---|
| uci | 50 | exact / arnoldi / **cheb-40** | 0.4 / 0.8 / **0.05** | 1.00 / 0.55 / **0.95** | — | 0.85 / **0.45** / 0.91 |
| uci | 300 | exact / arnoldi / **cheb-40** | 0.4 / 0.8 / **0.4** | 1.00 / **0.27** / **1.00** | 5.7e-5 | 0.96 / **0.51** / **0.96** |
| reddit_body | 50 (t=88) | exact / arnoldi / **cheb-40** | 14.9 / 13.9 / **1.1** | 1.00 / 0.62 / **1.00** | 4.0e-7 | 0.93 / 0.77 / **0.93** |
| reddit_body | 50 (t=175) | exact / arnoldi / **cheb-40** | 58.5 / 16.3 / **2.4** | 1.00 / 0.58 / **1.00** | 4.8e-7 | 0.84 / 0.62 / **0.84** |

1. **Chebyshev reproduces the exact basis** — overlap 1.000, eigenvalues to ~1e-7 — at **13–24x
   less wall-clock** on reddit, and it needs no matrix factorisation (the dominant cost).
2. **The Arnoldi estimate captures 27% of the k=300 subspace on uci** (58–62% at k=50 on reddit),
   with reconstruction AUC 0.45–0.51 at k=300. Every §3–§8 tracking result consumed that.
   The tracker has never seen the subspace it was built to track.
3. **Cutoff rule (measured, counter-intuitive): pin the cutoff AT or just below λ_k.** Raising it
   collapses the result — uci overlap 1.00 → 0.48 → 0.07 at 1.0x / 1.3x / 2.0x λ_k — because a
   dense spectrum puts far more than k modes under a raised cutoff and the block cannot span them.
   Buy accuracy with `oversample` (default 64) and `n_iter` (default 3), never with cutoff.
   Defaults reproduce the exact basis on every case tested here.

### 10.12b Tracking quality — the subspace is stable, the eigenvectors are not  **[2026-07-26]**

`analysis/probes/track_quality.py` measures, between consecutive snapshots and against the exact
solver as ground truth: subspace drift `overlap(span U_t, span U_{t+1})`, vector drift
`mean_i |<u_i(t), u_i(t+1)>|`, the fraction of the movement Procrustes can remove
(‖aligned‖/‖raw‖ Frobenius), and whether a warm-started Chebyshev solve lands on the exact
subspace. uci, k=50, every second snapshot:

| t | Δ edges | subspace drift | vector drift | Procrustes residual | warm cheb vs exact | cold cheb vs exact | arnoldi vs exact |
|---|---|---|---|---|---|---|---|
| 8 | 810 | 0.849 | 0.291 | 0.378 | 1.000 | 1.000 | 0.576 |
| 12 | 313 | 0.937 | 0.159 | 0.227 | 1.000 | 1.000 | 0.566 |
| 16 | 187 | 0.972 | 0.846 | 0.196 | 1.000 | 1.000 | 0.549 |
| 20 | 115 | 0.989 | 0.813 | 0.110 | 1.000 | 1.000 | 0.540 |
| 26 | 90 | 0.967 | 0.452 | 0.186 | 1.000 | 1.000 | 0.560 |

1. **The subspace is stable (0.93–0.99 once the graph settles) while individual eigenvectors are
   not (mean |cos| 0.08–0.85, typically 0.2–0.5 ≈ 60–78° of rotation).** The §10.12 gap prediction
   confirmed directly: crowding makes per-vector identity meaningless between snapshots, and
   leaves the span intact. Tracking is therefore well-founded — but only at the SUBSPACE level.
2. **Procrustes is doing real work, not papering over a bug**: aligning the new basis to the old
   removes 62–89% of the apparent movement (residual 0.11–0.38), i.e. most of the "churn" that
   §3/§4.2 attributed to instability is pure gauge rotation. This also explains `keep > recompute`
   mechanically: a frozen basis has no gauge motion to remove.
3. **Warm-started Chebyshev is free accuracy**: starting from the previous basis with the previous
   λ_k as cutoff lands on the exact subspace (overlap 1.000 at every step) 3–4x faster than a cold
   exact solve — and cold Chebyshev matches it too, so warm-starting costs nothing in quality.
4. **Arnoldi holds at ~0.55 overlap throughout** — the tracking path has been carrying half a
   subspace at every snapshot, consistently, not just occasionally.

### 10.12c The three regimes on a correct basis — better input, same ceiling  **[2026-07-26]**

`spectral.solver` (`arnoldi` | `exact` | `chebyshev`) now selects how the smodel's basis is built;
under `update_mode=update` the chebyshev path warm-starts from the previous basis with the previous
λ_k as cutoff (i.e. it *is* the tracker, done as filtered subspace iteration). uci, C3, f+s, gru,
3 seeds:

| solver | keep | update | recompute |
|---|---|---|---|
| arnoldi (historical) | 0.0756±0.0055 | 0.0783±0.0140 | 0.0670±0.0117 |
| **chebyshev** | **0.0879±0.0076** | **0.0867±0.0048** | 0.0729±0.0095 |
| feature baseline | 0.0892±0.0049 | | |

1. **Chebyshev beats Arnoldi in all three regimes** (+0.012 keep, +0.008 update, +0.006 recompute).
   Direction is consistent, but only `keep` clears the 0.005–0.007 noise floor at n=3.
2. **`keep > update > recompute` SURVIVES an exact-quality basis** — so that ordering was never an
   artifact of the empty Arnoldi estimate. It is the gauge effect measured in §10.12b: recompute
   re-draws the frame every step, `keep` has no frame motion at all.
3. **The ceiling does not move: the best spectral run (0.0879) still does not beat feature-only
   (0.0892).** Fixing the basis does not lift the smodel path above the bare backbone. This is
   §10.2's rank collapse restated — a readout that discards its input does not improve when the
   input improves. The binding constraint is the READOUT, not the basis.
4. Scope: uci C3 only, and uci is the dataset where NEW-pair spectral affinity is NEGATIVE
   (§10.11). as733 / reddit_body are where a better basis has something to be better at.

Consequence, and the direction taken next: everything that worked tonight used a
**rotation-invariant** readout (row inner products = projector entries), and everything that failed
fed raw eigenvector coordinates into a per-coordinate MLP. §10.12 explains why — the coordinates
are not identifiable at ~1e-3 gaps, only the subspace is. So the next smodel consumes invariants.

**DONE: `spectral.solver=chebyshev` is wired into `get_spectral_features`/`_spectral_step`**
with `X0` = previous basis and `cutoff` = previous λ_k, then run the one cell the program has never
tested: **the tracking mechanism on a non-empty basis**. §10.11's NEW-pair gradient (as733 +0.37…
+0.75, reddit_body +0.02…+0.03) says as733 and reddit_body are where it should be tried.

### 10.13 The invariant edge-score smodel (`data_type=f+es`) — built, and null on uci  **[2026-07-26]**

The design the evidence pointed to: contribute at the DECISION level (§10.11: decode-time fusion
moves the metric where embedding-level injection does not) using ROTATION-INVARIANT features
(§10.12b: individual eigenvectors rotate 50–80° between snapshots while the subspace is fixed, so
per-coordinate readouts consume an undetermined quantity).

`DynamicSInvariant` + `FedDynamicEdgeScoreClassifier` (`src/GNN/fed_dynamic_classifier.py`),
FedLap's fmodel/smodel composition, selected by `model.data_type=f+es`. Features per candidate
pair, all functions of the projector `U f(Λ) Uᵀ` and therefore invariant to `U → UR`:
heat-kernel affinities `Σ_i exp(−τ_j λ_i) û_ui û_vi` with **learnable** `τ_j`, the unfiltered
affinity (exactly the probe's feature, the τ→0 case), and a leverage term `‖U_u‖‖U_v‖`. A small
MLP maps them to a scalar added to the decoder logit; its last layer is zero-initialised so the
run starts exactly at the feature-only baseline. `assert_cfg` rejects `solver=arnoldi` here — an
invariant readout over a 0.27–0.62-overlap basis reads noise.

uci, keep, chebyshev solver, 3 seeds, with the stability-matched placebo:

| C | f+es laplacian | f+es shuffled_fixed | feature | real − placebo |
|---|---|---|---|---|
| 3 | 0.0857±0.0050 | **0.0919±0.0066** | 0.0787±0.0059 | **−0.006** |
| 9 | 0.0703±0.0091 | 0.0664±0.0093 | 0.0783±0.0054 | +0.004 |

**Null on uci UNDER `keep` — and that was a flaw in the test, not the model (see below).** The
edge term is genuinely learned (|mean| 0.000 at init → 0.23 mid-run → 0.14 at the end, std up to
0.36), so a dead gradient was ruled out.

**THE FIX AND THE RESULT: `update_mode=update` — the first in-model spectral win in this program
(2026-07-26).** `keep` freezes the basis solved at t=0; on uci that is a nearly-empty early graph.
Every probe that measured the effect used the CURRENT cumulative basis at each t, so the faithful
in-model setting is `update` (warm-started Chebyshev = the tracker, which §10.12b shows lands on
the same subspace as a fresh exact solve). Re-run that way, uci, 3 seeds, in-batch baselines:

| C | feature | **f+es laplacian** | placebo `shuffled_fixed` | placebo `random_fixed` | real − placebo | real − feature |
|---|---|---|---|---|---|---|
| 1 | 0.1110±0.0080 | **0.1506±0.0168** | 0.1142±0.0011 | — | **+0.036** | +0.040 (+36%) |
| 3 | 0.0844±0.0026 | **0.1358±0.0066** | 0.0828±0.0105 | 0.0851±0.0142 | **+0.053** | +0.051 (+61%) |
| 9 | 0.0776±0.0056 | **0.1058±0.0123** | 0.0692±0.0097 | — | **+0.037** | +0.028 (+36%) |

1. **Both placebos sit at the feature baseline** (0.069–0.114 vs feature 0.078–0.111) while the
   real basis is 0.106–0.151. The gain is STRUCTURAL, not capacity — the failure mode that
   explained away §8's SignNet parity and §10.9's L=1 effect does not apply.
2. **No per-seed overlap at any C** (e.g. C3: real [0.128, 0.144, 0.135] vs placebo
   [0.069, 0.094, 0.086]), against a 0.005–0.007 noise floor.
3. Gains are +28…+51 MRR points absolute, +36…+61% relative, largest at C3; not monotone in C.
4. This is the combination the whole §10 arc pointed to and had never been run together:
   **decision-level contribution + rotation-invariant features + a basis that is both correct
   (Chebyshev, §10.12) and current (tracked, `update`)**. Each ingredient alone was null:
   embedding-level (§10.2–10.7), per-coordinate readout (§10.13 `keep` row above and §10.2's rank
   collapse), Arnoldi basis (§10.12), frozen basis (the `keep` rows).
5. **`recompute` reproduces `update` exactly — the regime ordering has collapsed** (uci, same
   batch): real 0.1475/0.1315/0.1035 and placebo 0.1079/0.0818/0.0696 at C1/C3/C9, i.e. within
   0.002–0.003 of the `update` rows above, with the same +0.034…+0.050 margins. This is the
   §10.12b prediction realised end-to-end: a warm-started filtered solve and a fresh one land on
   the same subspace, and an invariant readout cannot tell them apart. Consequences —
   (a) `keep > update > recompute` (§3/§4.2/§10.12c) was an artifact of a gauge-sensitive readout
   over a poorly-resolved basis; with the gauge removed by construction the ordering **inverts**
   (`keep` is now the null arm, because it serves an outdated basis, while both current-basis
   regimes win); (b) the choice is now purely economic — `update` costs 3–25x less wall-clock
   (§10.12) for the same result, so tracking earns its keep as an efficiency mechanism rather than
   a stability one.
6. **THE SWEEPS LANDED (2026-07-27/28) — see §10.14 for the full matrix.** The uci result
   replicates on reddit_body and as733, and the persistence control there reverses the reading.
   Mechanism is still NOT attributed: the §10.11 probe found uci's NEW-pair
   spectral delta NEGATIVE, so the in-model gain is not simply the probe's effect — candidates are
   the learnable heat-kernel filters, the leverage feature, or backbone co-adaptation (the model
   can shift its own scores knowing the spectral term exists, which a post-hoc probe cannot).
   A `recompute` vs `update` comparison and a per-feature ablation are the next diagnostics.

Consistent with §10.11: uci's NEW-pair spectral effect is NEGATIVE, and its probe gain came from
recurrence, which the backbone already handles. The datasets where the basis demonstrably predicts
unseen links — as733 (+0.37…+0.75 on new pairs, growing with C) and reddit_body (+0.02…+0.03) —
are where this model must be judged. **Those runs are now DONE — §10.14.**
Also note (batch-effect caveat, §10.11 point 4): `feature C3` reads 0.0787±0.0059 in this batch and
0.0892±0.0049 in §10.12c's — identical condition, 3-seed means. Compare only within a batch.

### 10.14 The f+es sweep matrix — replication, and the persistence reversal  **[2026-07-27/28]**

28 cells, 3 seeds each, `f+es` + `solver=chebyshev` unless stated, run across sim07–sim15.
**Primary logs: `/nas/lnt/stud/ge27yuv/runs/es_sweep/*.log`** (runners `runs/run_es{,2,3}.sh`,
`run_after.sh`); harvest with `grep "RESULT.*mean_mrr" <dir>/*.log`. uci rows are local
(§10.13 batch). Every value below is a 3-seed mean±population-std taken from those logs.

_Spectral arm (`es_features=spec`), `update_mode=update` unless the row says recompute:_

| dataset | C | feature | real (update) | real (recompute) | placebo (update) | placebo (recompute) |
|---|---|---|---|---|---|---|
| reddit_body | 1 | 0.3927±0.0029 | **0.4452±0.0056** | 0.4440±0.0057 | 0.3931±0.0057 | 0.3933±0.0032 |
| reddit_body | 9 | 0.2678±0.0058 | **0.3469±0.0070** | 0.3408±0.0078 | 0.2659±0.0060 | 0.2595±0.0094 |
| as733 | 1 | 0.3345±0.0016 | **0.6398±0.0009** | 0.6402±0.0009 | 0.3361±0.0010 | 0.3326±0.0010 |
| as733 | 9 | 0.2491±0.0204 | **0.6228±0.0021** | 0.6225±0.0018 | 0.2651±0.0037 | 0.2607±0.0054 |

1. The uci result replicates on both: real beats placebo by +0.052/+0.081 (reddit_body) and
   +0.304/+0.358 (as733); placebos sit on the feature baseline; no per-seed overlap.
2. `recompute` reproduces `update` on both datasets (≤0.006), as §10.12b predicts.
3. Margin grows C1→C9 on reddit_body (+0.052→+0.081) and as733 (+0.304→+0.358). **On uci it does
   NOT** (+0.040/+0.051/+0.028 at C1/C3/C9, §10.13) — so this is 2 of 3, not a law.

_Persistence control (`es_features=persist`, the 1-bit "pair already exists" feature) and the
combination (`both`):_

| dataset | C | spectral | **persist** | both (real) | both (placebo) |
|---|---|---|---|---|---|
| uci | 3 | 0.1358±0.0066 | **0.2184±0.0099** | 0.2245±0.0063 | 0.2150±0.0139 |
| uci | 9 | 0.1058±0.0123 | **0.1852±0.0103** | 0.1869±0.0090 | 0.1611±0.0133 |
| reddit_body | 1 | 0.4452±0.0056 | **0.5567±0.0042** | — | — |
| reddit_body | 9 | 0.3469±0.0070 | **0.4644±0.0039** | 0.4485±0.0094 | 0.4488±0.0109 |
| as733 | 1 | 0.6398±0.0009 | **0.8750±0.0001** | — | — |
| as733 | 9 | 0.6228±0.0021 | **0.8546±0.0044** | 0.8652±0.0016 | 0.8584±0.0006 |

4. **A 1-bit lookup beats the entire spectral apparatus on every dataset and every C measured**
   (+0.08 uci, +0.11 reddit_body, +0.23 as733).
5. **Given persistence, the spectrum adds +0.011 (as733 C9) and −0.016 (reddit_body C9).** Both
   exceed the 0.005–0.007 noise floor and have OPPOSITE signs: the honest statement is "helps
   slightly on one, hurts slightly on the other", not "nothing".
6. `both`(real) vs `both`(placebo) is +0.007 (as733) and −0.000 (reddit_body) — but note the
   reddit_body `both` arms both fall BELOW persist-alone, i.e. adding spectral features (real or
   scrambled) to persistence costs ~0.016 there.
7. Caveat carried from §10.13: the uci rows come from a different batch than §10.12c's; identical
   conditions have differed by 0.010 across batches. Compare only within a batch.

### 10.15 The v2 sweep — first batch after the leverage-scale fix, and the bitcoin test  **[COMPLETE, 2026-07-29]**

**This section supersedes §10.13/§10.14's absolute numbers.** All of those were produced before
commit `6ef42a2`, which fixed a train/serve inconsistency in the `f+es` smodel: the leverage
feature `log1p(‖U_u‖‖U_v‖ · N)` took `N` from the *served block's* row count, so the server (all N
rows) and each client (its own rows only) computed a DIFFERENT feature for the same pair, while
the MLP consuming it is FedAvg-averaged across both. The fix serves the global node count to
server and clients alike. C1 is essentially unaffected (one client, whose block is the whole
graph); every C>1 cell is re-measured here.

**Primary logs: `/nas/lnt/stud/ge27yuv/runs/v2_sweep/*.log`** (runners `runs/run_v2.sh` +
`runs/run_list.sh`, `progress.txt` records rc + host per job); harvest with
`grep "RESULT.*mean_mrr" <dir>/*.log`. Launched 2026-07-29 02:43 across sim07–sim10, sim12,
sim14, sim15. Every value is a 3-seed (1234/1334/1434) mean ± population std unless flagged.
`f+es` + `solver=chebyshev` + `update_mode=update` unless the row says otherwise.

_uci — COMPLETE, C{1,3,7,9}:_

| C | feature | **spec (real)** | placebo `shuffled_fixed` | real − placebo | real − feature |
|---|---|---|---|---|---|
| 1 | 0.1202±0.0017 | **0.1578±0.0059** | 0.1143±0.0042 | **+0.044** | +0.038 (+31%) |
| 3 | 0.0891±0.0119 | **0.1356±0.0060** | 0.0860±0.0133 | **+0.050** | +0.047 (+52%) |
| 7 | 0.0767±0.0080 | **0.1170±0.0016** | 0.0633±0.0028 | **+0.054** | +0.040 (+53%) |
| 9 | 0.0718±0.0129 | **0.0967±0.0138** | 0.0682±0.0138 | **+0.029** | +0.025 (+35%) |

1. **The headline survives the fix.** Real beats both its placebo and the feature baseline at every
   C, placebos sit on the feature baseline (0.063–0.114 vs feature 0.072–0.120), and the margin is
   the same size as §10.13's (+0.036/+0.053/+0.037 there). C7 is new and fits the pattern.
2. Still **not monotone in C** (+0.044/+0.050/+0.054/+0.029): C7 is now the largest margin and C9
   the smallest, so §10.14's "margin grows with sharding" remains 2 of 3 datasets, not a law.
3. `recompute` reproduces `update`: 0.1392±0.0066 vs 0.1356±0.0060 (C3) and 0.1063±0.0118 vs
   0.0967±0.0138 (C9) — within one std, as §10.12b predicts.
4. **Persistence still dominates on uci**: 0.2281±0.0110 (C3) and 0.1906±0.0081 (C9) against
   spectral's 0.1356 / 0.0967, i.e. the 1-bit lookup is worth ~+0.094 more than the entire
   spectral apparatus. The §10.14 caveat is unchanged on this dataset.

_bitcoin — the low-recurrence test (8% edge recurrence). ALL FOUR CELLS COMPLETE:_

| dataset | C | feature | **spec (real)** | placebo | persist | real − placebo | **real − feature** |
|---|---|---|---|---|---|---|---|
| bitcoin_alpha | 1 | 0.1675±0.0045 | 0.1748±0.0032 | 0.1593±0.0028 | 0.1603±0.0079 | +0.016 | **+0.007** |
| bitcoin_alpha | 9 | 0.0982±0.0032 | 0.0785±0.0204 | 0.0676±0.0154 | 0.0911±0.0045 | +0.011 | **−0.020** |
| bitcoin_otc | 1 | 0.2017±0.0034 | 0.2040±0.0034 | 0.1957±0.0147 | 0.1971±0.0069 | +0.008 | **+0.002** |
| bitcoin_otc | 9 | 0.1066±0.0176 | 0.1064±0.0048 | 0.0837±0.0137 | 0.1031±0.0059 | +0.023 | **−0.000** |

5. **The spectral term is NULL on bitcoin.** Against the baseline that decides whether it earns its
   place — feature-only — the four cells read +0.007, −0.020, +0.002, −0.000. The largest positive
   (alpha C1, +0.007) is at the 0.005–0.007 noise floor, ~1.6 pooled std, and does not replicate at
   C9 on the same graph or at either C on bitcoin_otc; the one value clearly outside the floor is
   negative and is a seed artifact (point 5b). Stated conservatively: **no bitcoin cell is a
   convincing win, and the cell-to-cell scatter (−0.020…+0.007) is the size of the effect itself.**
   Contrast uci, where the gain is +0.025…+0.047 at all four C, an order of magnitude larger and
   consistent in sign — and reddit_body (+0.053/+0.079) and as733 (+0.306/+0.360).
5b. **The alpha C9 −0.020 is one collapsed seed, and the placebo collapses with it** — so read it
   as null, not as harm. Per-seed MRR at that cell: feature `[0.0942, 0.1020, 0.0985]` and persist
   `[0.0962, 0.0853, 0.0918]` are both tight, while spec `[0.0500, 0.0888, 0.0967]` and placebo
   `[0.0491, 0.0867, 0.0669]` both collapse on seed 1234. The failure therefore belongs to the
   spectral feature BLOCK (real and scrambled alike) destabilising the edge-score MLP on this
   dataset, not to spectral structure — `persist`, which routes through the same smodel with a
   1-bit input, is unaffected. Dropping seed 1234 leaves spec 0.0928 vs feature 0.1002, i.e. still
   −0.007. Conclusion unchanged, magnitude overstated.
6. **A methodological correction that matters for every earlier claim: `real − placebo` overstates
   the effect wherever the placebo damages the model.** On bitcoin the `shuffled_fixed` arm sits
   0.010–0.031 BELOW the feature baseline, so bitcoin_otc C9's headline-looking "+0.023 over
   placebo" is entirely the placebo's own damage — the real arm is exactly at baseline (0.1064 vs
   0.1066). The two controls answer different questions and both are needed: the placebo answers
   *structure or capacity?*, the feature baseline answers *does it help at all?*. §10.13/§10.14
   led with real−placebo; the real−feature column should lead from here on. (uci is unaffected:
   its placebo sits within 0.003–0.013 of baseline and real−feature is positive at every C.)
7. **Persistence is also null-to-negative on bitcoin** (−0.007 alpha C9, −0.004 otc C9, −0.005
   otc C1) — exactly what 8% recurrence predicts, and the control working as designed.
8. **The one counter-signal, stated so it is not lost: AUC on bitcoin_otc C1.** There the real arm
   reads 0.9615±0.0007 against feature 0.9477±0.0051 and placebo 0.9533±0.0034 — non-overlapping
   ranges, so a genuine +0.014 over the baseline and +0.008 over the placebo. It does not survive
   contact with the rest: otc C9 is +0.008 AUC with fully overlapping stds, and alpha C9 is −0.058.
   MRR is the reported metric of this program and it is flat in all three cells. Read as one cell
   of AUC-only movement, not as a rescue.
9. **Third independent replication of the bitcoin split.** Spectral information has now failed on
   these two graphs through three different injection mechanisms: input-level LapPE (§10.7),
   decode-time probe fusion (§10.10 gate 2, "real ≈ placebo at every C"), and now the in-model
   edge-score smodel. The dataset dependence is a property of the data, not of any one design.

**What this decides (the question posed for this session).** The spectral term does NOT do
something a history lookup cannot: on the two datasets where persistence has almost nothing to
offer, the spectral term has nothing to offer either. Ordering the in-model gain over feature-only
by the dataset's recurrence rate gives a monotone relationship with no exception:

| dataset | recurrence | in-model `f+es` gain over feature |
|---|---|---|
| bitcoin_{alpha,otc} | 0.08 | −0.020 … +0.007 (4 cells, scatter ≈ effect size) |
| uci | 0.49 | +0.025 … +0.047 (4 cells, all positive) |
| reddit_body | 0.55 | +0.053 / +0.079 (§10.14, pre-fix batch) |
| as733 | 0.95 | +0.306 / +0.360 |

So the honest framing is the second of the two the session set out: **the method is a compact
encoding of recurrence**, not a structural signal that a lookup misses. This is consistent with
§10.11's probe, which found the overall spectral gain to be the REPEAT-subset gain diluted by the
repeat fraction. It does not retract §10.15/§10.14's positive result — the gains over the backbone
are real, controlled and large — but it fixes what may be claimed for them. The one piece of
evidence still pointing the other way is §10.11's NEW-pair measurement (as733 +0.37…+0.75,
reddit_body +0.022…+0.034, placebo ~0), which is an offline probe, not the in-model metric; the
in-model repeat/new split that would settle it is deferred pending sign-off (it touches the eval
path).

_as733 — COMPLETE:_

| C | feature | **spec (real)** | placebo | persist | real − placebo | **real − feature** |
|---|---|---|---|---|---|---|
| 1 | 0.3351±0.0004 | **0.6407±0.0003** | 0.3358±0.0006 | — | +0.305 | **+0.306 (+91%)** |
| 9 | 0.2680±0.0078 | **0.6283±0.0016** | 0.2724±0.0037 | 0.8572±0.0058 | +0.356 | **+0.360 (+134%)** |

10. as733 replicates §10.14 to within noise (C1 spec 0.6407 vs 0.6398; C9 spec 0.6283 vs 0.6228),
    placebos sit exactly on the feature baseline, and the margin still grows C1→C9. Persistence
    still beats it there (0.8572 vs 0.6283).
11. _Operational note._ The originally-launched C9 cells (`spec`, `shuffled_fixed`, `persist`) all
    died `rc=1` with CUDA OOM on sim15, whose 48 GB GPU was already holding 45.7 GB across two
    processes belonging to another user. Relaunched on sim12 (verified idle) and completed clean.
    The C1/C9 `feature` baselines were absent from the original job list and were added on sim10 —
    without them no as733 gain-vs-feature could have been stated. **as733 costs ~11 min/seed on the
    Chebyshev path**, not the ~40 min recorded in MEMORY.md, which predates that solver.

_Method note carried forward:_ compare only within this batch. uci `feature` C3 reads 0.0891 here,
0.0844 in §10.13's batch and 0.0892 in §10.12c's — a 0.005 spread on an identical condition.

### 10.16 Attribution — the plain affinity carries it; the filters and leverage are inert  **[confirmed on uci C{1,3,9} + as733 C9, 2026-07-29]**

The experiment §10.13 point 6 and the method doc both flagged as the one that would attribute the
mechanism. `spectral.es_spec_parts` (commit `984bcb7`) selects which of the three invariant blocks
inside `DynamicSInvariant` reach the MLP: `phi` (the J=4 **learnable** heat-kernel affinities
`Σ_i exp(−τ_j λ_i) û_ui û_vi`), `cos` (the **unfiltered** affinity `Σ_i û_ui û_vi` — exactly the
quantity §10.11's probe measured), and `lev` (the leverage term `log1p(‖U_u‖‖U_v‖·N)`). Default
`phi+cos+lev` reproduces the previous behaviour exactly (same feature order and width).

**Logs: `/nas/lnt/stud/ge27yuv/runs/abl_parts/*.log`** (runners `runs/run_abl.sh`,
`runs/run_abl_uci.sh`, `progress.txt` has rc + host), sim13, 2026-07-29 03:12. uci, `f+es`,
`solver=chebyshev`, `update_mode=update`, 3 seeds. Each arm is run against its OWN
`shuffled_fixed` placebo, and the `feature`/`persist` baselines are re-run **inside this batch** so
every comparison is within-batch.

_uci C1 — COMPLETE (in-batch feature = 0.1185±0.0044, persist = 0.2677±0.0127):_

| parts | real | placebo | real − placebo | real − feature |
|---|---|---|---|---|
| `phi` | 0.1556±0.0098 | 0.1174±0.0061 | +0.038 | +0.037 |
| **`cos`** | **0.1603±0.0055** | 0.1128±0.0035 | **+0.048** | **+0.042** |
| `lev` | 0.1116±0.0138 | 0.1120±0.0077 | **−0.000** | −0.007 |
| `phi+cos+lev` | 0.1472±0.0020 | 0.1067±0.0116 | +0.041 | +0.029 |

_uci C3 — COMPLETE (in-batch feature = 0.0881±0.0173; persist 0.2296 at n=1, still running):_

| parts | real | placebo | real − placebo | real − feature |
|---|---|---|---|---|
| `phi` | 0.1302±0.0048 | 0.0868±0.0141 | +0.043 | +0.042 |
| `cos` | 0.1288±0.0019 | 0.0869±0.0240 | +0.042 | +0.041 |
| `lev` | 0.0831±0.0121 | 0.0845±0.0176 | **−0.001** | −0.005 |
| `phi+cos+lev` | 0.1352±0.0033 | 0.0796±0.0156 | +0.056 | +0.047 |

1. **The leverage term is inert.** Real minus its own placebo is −0.000 (C1) and −0.001 (C3), and
   at C1 it sits 0.007 BELOW the feature baseline. It carries no signal at either C. Note the
   irony: commit `6ef42a2` — the fix that this whole v2 batch was run to re-measure — corrected the
   node scale of *this* feature. That is why the v2 uci numbers land on top of §10.13's
   (C3 0.1356 vs 0.1358): the fix repaired a train/serve inconsistency in a term that contributes
   nothing. The fix was still correct; it simply could not move the metric.
2. **The unfiltered affinity carries the entire effect.** `cos` alone is the best single arm at C1
   (+0.042 over feature, +0.048 over its placebo) and matches `phi` at C3.
3. **The learnable heat-kernel filters add nothing over it.** `phi` ≈ `cos` at both C (0.1556 vs
   0.1603 at C1; 0.1302 vs 0.1288 at C3), all within the noise floor. This is the expected
   outcome rather than a surprise: `phi_j → cos` as `τ_j → 0`, so the filter bank's best available
   strategy is to reproduce the unfiltered affinity, and it does. The learnability is not doing
   work.
4. **Combining is not better than the best part.** At C1 the full set (0.1472) is 0.013 BELOW `cos`
   alone (0.1603); at C3 it is 0.005–0.006 above `cos`/`phi` (0.1352 vs 0.1288/0.1302), i.e. at the
   noise floor. Across both C the honest statement is that the three-block combination buys nothing
   over the single best block.
5. **Consequence for the method.** The `f+es` contribution reduces to ONE scalar per candidate
   pair, `Σ_i û_ui û_vi` over the tracked basis — no filter bank, no learnable τ, no leverage term.
   That is a substantial simplification of what §10.13 built, and it is the same quantity the
   offline probe used, which closes the gap §10.13 point 6 opened ("the in-model gain is not
   simply the probe's effect"): on uci it now appears that it IS the probe's feature, consumed
   in-model.
5b. **Dropping `phi` makes the readout EXACTLY gauge-invariant, not approximately.** Write the
   block as `û_u diag(g(λ)) û_vᵀ`. Under `U → UR` this is unchanged for all pairs iff `R` commutes
   with `diag(g(λ))` — i.e. iff `R` is block-diagonal with respect to the eigenvalue multiplicities.
   For `cos` the filter is the identity (`g ≡ 1`), so it commutes with EVERY orthogonal `R`: `cos`
   is exactly invariant. `lev` is too (`‖uR‖ = ‖u‖`). But `phi` with distinct `τ_j` only commutes
   within an eigenspace, which is precisely the "O(|f(λ_i) − f(λ_j)|) sensitive to mixing between
   nearly-equal eigenvalues" caveat in `DynamicSInvariant`'s own docstring. So the ablation does not
   merely simplify the method — it removes the one block whose invariance was approximate, on a
   spectrum §10.12 measured as clustered at ~1e-3 gaps, which is exactly the regime where that
   approximation is weakest. **The empirically-best variant is also the theoretically-clean one.**
6. Persistence still dominates on uci in this batch too: 0.2677±0.0127 against the best spectral
   arm's 0.1603±0.0055.
_uci C9 — COMPLETE (in-batch feature = 0.0743±0.0070, persist = 0.1946±0.0199):_

| parts | real | placebo | real − placebo | real − feature |
|---|---|---|---|---|
| `phi` | 0.1098±0.0046 | 0.0757±0.0054 | +0.034 | +0.036 |
| `cos` | 0.1155±0.0133 | 0.0704±0.0061 | +0.045 | +0.041 |
| `lev` | 0.0706±0.0110 | 0.0764±0.0117 | **−0.006** | −0.004 |
| `phi+cos+lev` | 0.1016±0.0071 | 0.0680±0.0103 | +0.034 | +0.027 |

_as733 C9 — THE REPLICATION, on the dataset with the largest effect (in-batch feature =
0.2656±0.0054); logs `runs/abl_parts/as733_*.log`, sim07 + sim08:_

| parts | real | placebo | real − placebo | real − feature |
|---|---|---|---|---|
| `phi` | 0.5966±0.0079 | — | — | **+0.331** |
| **`cos`** | **0.5898±0.0076** | 0.2785±0.0014 | **+0.311** | **+0.324** |
| `lev` | 0.2474±0.0083 | 0.2653±0.0111 | **−0.018** | **−0.018** |

7. **The attribution replicates, and it is not a small-dataset artifact.** On as733 C9 — where the
   full model scores +0.360 over feature — `cos` ALONE delivers +0.324 of it and `phi` alone
   +0.331, while `lev` is again inert-to-negative (−0.018 against both its own placebo and the
   feature baseline). The same three-way ordering holds on uci at C1, C3 and C9 and on as733 at C9:
   **`cos` ≈ `phi` ≫ `lev` ≈ 0**, four cells, two datasets, three orders of magnitude of effect
   size. `cos` alone recovers ~90% of the full model's gain on as733 and 100%+ of it on uci.
8. **Scope of what is NOT yet shown.** reddit_body has not been attributed, and as733 was measured
   at C9 only (no C1, and `phi`'s placebo was not run — it was the least decisive arm under a fixed
   GPU budget). Nothing here contradicts the uci reading, but "the filter bank is unnecessary" rests
   on `phi ≈ cos` rather than on `phi` failing against a placebo.

---

## 12. The MRR negative-filter asymmetry — measured, and it does not matter  **[2026-08-10]**

**The issue.** The two negative samplers in one evaluation use different forbidden sets. The
classification batch forbids the target snapshot's WHOLE `edge_index`
(`federated_orchestrator.py`), while the MRR ranker forbids only the positives placed in the batch,
i.e. the evaluated split (`src/metrics/mrr.py`). Under `split: [0.8,0.1,0.1]`, `pos_test` is 10% of
the snapshot, so ~90% of the target snapshot's true edges are eligible MRR negatives.

**This is ROLAND's behaviour, not a divergence.** `roland/graphgym/contrib/train/train_utils.py`
:230-235 takes `edge_label_index[:, edge_label == 1]` and `gen_negative_edges` set-differences
against exactly that (`:50`). The default must therefore stay as-is or Table-3 comparability is
lost. New knob `metric.mrr_filter` ∈ `split` (default, unchanged) | `snapshot` | `both`; strict mode
resamples to keep exactly K, so both arms rank against K competitors.

**Tier 1 — contamination rate (model-free, `analysis/probes/mrr_contamination.py`).** Expected
number of a K=1000 draw landing on a true edge of the target snapshot outside the evaluated split:

| dataset | N | eval pairs | sources | contam / K | % of K | sources ≥1 | max |
|---|---|---|---|---|---|---|---|
| `uci`         |  1,899 |  27 |   2,400 | **3.012** | 0.301% | 68.8% |  81.6 |
| `as733`       |  7,716 | 732 | 698,776 | **1.069** | 0.107% | 18.1% | 168.7 |
| `reddit_body` | 35,776 | 176 |  25,071 | **0.051** | 0.005% |  0.2% |   4.3 |

**Tier 2 — paired in-model delta** (`uci`, C=1, `feature`, `mrr_filter=both`, 3 seeds; same model,
same snapshot, same seed — only the forbidden set differs):

| seed | MRR `split` | MRR `snapshot` | delta |
|---|---|---|---|
| 1234 | 0.10786 | 0.10196 | −0.00590 |
| 1334 | 0.11702 | 0.11635 | −0.00067 |
| 1434 | 0.10731 | 0.10534 | −0.00198 |

**delta = −0.0029 ± 0.0022** (mean ± population std, n=3), **inside the documented 0.005–0.007
noise floor**.

**The pre-registered prediction was WRONG on both counts, and is recorded as such.**
It said (a) the delta would be non-negative — removing genuine edges from the negative pool can only
reduce the competitors outranking the positive — and (b) it would scale with edge-recurrence rate,
giving `as733` > `reddit_body` > `uci`. Neither held:

1. The delta is *negative* on all three seeds, though too small to distinguish from draw noise.
2. Contamination is driven by **snapshot density relative to N**, not recurrence. `uci` (recurrence
   0.49) has 59× the contamination of `reddit_body` (0.55) purely because N is 19× smaller. The
   recurrence hypothesis for this quantity is refuted.

**Reading.** Contamination is real but inert. ~3 of 1000 `uci` negatives are genuine edges, yet the
metric barely moves — a contaminating pair only inflates the rank if it outscores the source's best
positive, and at MRR≈0.11 the backbone does not reliably score true edges that highly.

**Tier 3 — the draw-noise control, now RUN** (`analysis/probes/mrr_draw_noise.py`, uci, C=1,
`feature`, 3 seeds). `_extra_forbidden` is neutralised so BOTH arms of `mrr_filter=both` use the
split filter; the only remaining difference between them is the second independent draw, so every
delta it reports is pure sampling noise:

| | per-seed deltas | mean ± std | range |
|---|---|---|---|
| measured effect (Tier 2) | −0.0059, −0.0007, −0.0020 | **−0.0029 ± 0.0022** | 0.0052 |
| draw noise (this control) | +0.0045, +0.0042, −0.0050 | **+0.0012 ± 0.0044** | 0.0095 |

**The noise floor is larger than the effect.** The Tier-2 delta is therefore **not separable from
sampling variance**, and the earlier "consistently negative on 3/3 seeds" observation carries no
weight either — the noise control produced 2 positive and 1 negative of comparable magnitude, so
sign consistency at n=3 is uninformative here.

This does not change §12's conclusion, it grounds it: the filter asymmetry is inert not merely
because the measured effect is small, but because it is indistinguishable from the noise of drawing
K negatives twice. Any future attempt to detect it needs either many more seeds or a design that
holds the draw fixed across arms.

**Caveat on the headline of a `both` run** (measured during the test audit): the strict arm makes a
different number of `randint` calls than the split arm (3 vs 2 for the same input), so enabling
`snapshot` or `both` shifts the run's global RNG stream from the first eval onward. A `both` run's
headline MRR is therefore **not comparable to a banked `split` run's** — visible above as seed
1234 giving 0.10786 under `both` against 0.10791 for the same seed under plain `split`. The *delta*
remains a valid within-run paired quantity (same model state, same snapshot); only the absolute
headline drifts. The `split` default is untouched and is pinned bit-exactly by
`tests/test_mrr_filter.py::test_no_extra_forbidden_is_bit_identical`.

**Consequence for §10.14/§10.16's persistence comparison:** the exposure runs *against* recurrence-
predicting methods, so persistence's measured lead over the spectral term is if anything
understated. It is not an artefact that a protocol fix would remove.

**Cluster runs deliberately NOT spent:** `uci` is the worst case by a factor of 3 over `as733` and
59 over `reddit_body`, and it shows no detectable effect, so the Tier-2 runs on the other two were
dropped rather than executed. Recorded so the gap is not mistaken for an oversight.

### 12b. Re-confirmation and refinement of the known fixed-seed non-determinism

**This is a known property, not a new finding** — the project's standing reference note already
records that the CPU sandbox is non-deterministic at fixed seed ("same-seed fedlap feature C=1 runs
varied 0.076–0.098", multi-threaded float reductions rather than MPS atomics) and already prescribes
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0` for bit-exact reruns. What follows
re-confirms it on the current HEAD, quantifies it on `uci`, and **narrows the remedy**.

Six runs of the **identical command, identical seed (1234), on CPU**, `uci` / `feature` / C=1:

`0.09678, 0.10401, 0.11427, 0.12145, 0.10939, 0.10608` → **0.1087 ± 0.0078** (population std,
n=6), range **0.0247**.

Cause isolated by elimination:

| condition | reproducible? | wall-clock (uci) |
|---|---|---|
| default | **no** — six values spanning 0.0968–0.1215 | 14 s |
| `PYTHONHASHSEED=0`, default threads | **no** — 0.10939 vs 0.10608 | — |
| `torch.set_num_threads(1)` + `OMP/MKL_NUM_THREADS=1` | **yes** — `0.10063452711673798` ×2 | 16 s |
| **`torch.use_deterministic_algorithms(True)`, default threads** | **yes** — `0.11838083517634207` ×3 | 16 s |

**Two refinements to the standing remedy:**

1. `PYTHONHASHSEED=0` is **neither necessary nor sufficient** — alone it leaves the run
   non-reproducible, and both working remedies below are bit-exact with the hash seed left random.
2. **`torch.use_deterministic_algorithms(True)` alone is sufficient on CPU, at full thread count.**
   It was never tried before; it raises no error **on CPU**, so every op exercised there has a
   deterministic kernel. This is the better remedy on CPU: thread pinning serialises the whole run,
   whereas the flag keeps the thread pool. On `uci` both cost the same (~14% over default), but
   `uci` is small enough that serialisation barely bites — the advantage should grow with graph
   size, which is **not measured here** and should not be assumed.

   **CUDA: MEASURED on sim10 (RTX 4080), 2026-08-16.** The prediction was half right.

   | run | outcome |
   |---|---|
   | `deterministic=true`, no env var | **RuntimeError at the first `F.linear`** (`encoder.py:21`): *"this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2 … you must set CUBLAS_WORKSPACE_CONFIG=:4096:8"* |
   | `deterministic=true` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | runs clean; `0.10520243979300614` on **two** runs, bit-identical |
   | `deterministic=false` (control) | `0.11026` vs `0.12371` — differ by **0.0135** |

   So the cuBLAS half of the prediction is **confirmed** and the scatter half is **refuted** — the
   message-passing aggregation has deterministic CUDA kernels and raises nothing. Only the env var
   was missing. Non-determinism on CUDA is also real and larger than on CPU (0.0135 here vs a
   ±0.0078 spread on CPU).

   **Fixed:** `main()` now does `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` when
   and only when the flag is enabled — separately verified on sim10 that setting it *after* torch
   is imported still works, so it can stay conditional. It is deliberately not set unconditionally:
   the workspace config can steer cuBLAS algorithm selection, and non-deterministic runs must stay
   comparable to banked numbers.

   The earlier wording here — "raises no error on this codebase" — was a CPU-only result stated too
   broadly. It was corrected to "CPU-only, CUDA unverified", and is now replaced by the measurement.

**Caveat that matters if either remedy is adopted:** the two converge to *different* fixed points
(`0.10063` thread-pinned vs `0.11838` deterministic-flag). Both are bit-reproducible; neither is
"the" correct value — they are different summation orders. So the choice must be made once and
held, and numbers produced under one are not comparable to numbers produced under the other.

**NOW AVAILABLE AS A KNOB (2026-08-15):** `experimental.deterministic` (bool, default **false**)
calls `torch.use_deterministic_algorithms(True)` once in `main()`, before seeding, dataset load and
partition. Verified: enabled → `0.11838083517634207` on two runs, bit-identical; disabled →
unchanged, and `_run_id()` stays byte-identical (`uci_gru_feature_C1_s1234`) so existing
checkpoints still load, while an enabled run gets `..._det_s1234` and cannot collide with one.
Suite 259/1/1. Default is off precisely because of the caveat above — **every number in this file
and in `docs/spectral_method.tex` was produced with it off**, and enabling it makes future numbers
non-comparable to them.

Quantification worth keeping:

1. On `uci`/`feature`/C=1 the fixed-seed spread is **±0.0078 (n=6), range 0.0247** — the same order
   as the declared noise floor (0.005–0.007), so that floor is not wrong, but a "3-seed mean ± std"
   does not average this away. Thread noise alone contributes ≈0.0045 of SEM to a 3-seed mean
   before any genuine seed variance.
2. It applies to the CUDA cluster runs as well (already recorded: same-seed UCI recompute gave
   0.060 then 0.068).

**This does NOT affect §12's Tier-2 delta.** Both arms of that comparison are computed inside a
single run against a single model state, so the paired delta carries no run-to-run variance. That
is precisely why the paired design was chosen over comparing two separate runs — a two-run design
would have been swamped, since the ±0.0078 run-to-run spread is ~3× the −0.0029 effect being
measured.

**Not yet done:** decide whether to pin threads for future reported runs (reproducibility vs
wall-clock), and whether to restate the noise floor's attribution in §5 and in
`docs/spectral_method.tex`'s reporting paragraph.

---

## 13. Run-identity gap — an `f+s` placebo could have shared its real arm's checkpoint  **[2026-08-16]**

`DynamicServer._run_id()` is what stops a re-launched run from loading a foreign checkpoint. Until
today it omitted knobs that change the numbers:

| data type | identity carried | omitted |
|---|---|---|
| `f+s` / `structure` | `um`, `sfv`, `proc` | **`basis_source`**, `solver` |
| `f+pe` | `um`, `pe`, `basis` | `solver` |
| `f+es` | *nothing beyond dataset/update-method/type/C/seed* | everything spectral |

**The `f+s` row is the one that matters.** `basis_source` is the placebo switch, so a `laplacian`
run and its `shuffled_fixed` control at the same dataset/C/seed produced the **same identity** — and
`f+s` checkpointing works. Under `train.auto_resume=true` the placebo arm could therefore load the
real arm's `.ckpt`, or return its `.done` results outright, manufacturing agreement between an arm
and its own control.

**Whether this actually happened cannot be settled from this checkout.** `train.auto_resume`
defaults to `false`, and the sweep runner scripts are not committed here. **Before any `f+s` placebo
comparison in §10 is quoted as an independent control, check the cluster job scripts for
`auto_resume`.** If it was off throughout, nothing is affected.

The `f+es` row looks worse but was unreachable: `_get_sfv` dereferenced `smodel.graph.x` and
`DynamicSInvariant` has no `graph`, so `f+es` + `auto_resume` raised `AttributeError` before any
collision. A crash is not a safeguard, and a strict xfail in `tests/test_edge_score_smodel.py`
had already documented it.

**Fixed.** Identity now carries `basis_source` on `f+s`, a full `f+es` branch
(`um`/`pe`/`basis`/`es_features`/`es_spec_parts`/`proc`), and `solver` — the last appended only when
it is not the default, so default-solver identities stay byte-identical and existing checkpoints
still load. `spectral.solver` is now validated against `{arnoldi, exact, chebyshev}` instead of
silently falling through to the Krylov path on a typo. `_get_sfv`/`_set_sfv` go through the
smodel's `get_SFV`/`set_SFV` protocol rather than reaching into `graph`, which also makes `f+es`
checkpointing work (the xfail is now a passing test). Suite 290 passed / 1 skipped.

---

## 11. Provenance & reproduction
- Code: fedlap repo, branch `roland-dev`. Runs on TUM CUDA hosts `tueilnt-sim{08,09,10,12,13,14}`.
- Commits behind §8/§9: `3595fc8` SignNet smodel; `f45c3a3` `federated.fl`; `a3e907a`
  `metric.eval_scope`; `18cdb6e` + `95ce06a` the refresh train/eval-mode fix (§9).
- Run pattern (one condition): `python main.py -c config/<dataset>_gru.yaml --repeat 3
  --set model.data_type=f+s spectral.update_mode=<keep|update|recompute>
  spectral.use_procrustes=<true|false> subgraph.num_subgraphs=<C> wandb.mode=online`
  (feature: `--set model.data_type=feature subgraph.num_subgraphs=<C>`).
- For large/long runs add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `--set train.auto_resume=true train.ckpt_period=100 train.ckpt_dir=<dir>`.
- Durable narrative + history: auto-memory `fedlap-spectral-federated-results`,
  `fedlap-roland-migration`, `fedlap-remote-host`; repo `fedlap/MEMORY.md`.
- Numbers here are copied from those runs; wandb `cod-tum/dynamic-fedlap` holds the plots.
- §8/§9 raw logs (NOT committed; harvest with `grep "RESULT dataset" <dir>/*.log`):
  `/nas/lnt/stud/ge27yuv/runs/{reddit_body_signnet, reddit_body_baselines_hardneg, as733_signnet,
  uci_signnet, signnet_breadth}` (§8) and `{fl_localeval, local_baseline_fl_false}` (§9). §8/§9 report 3-seed MEANS;
  per-run rows live in those logs rather than inline, unlike §6.
- Commits behind §10: `3291fce` (`spectral.deterministic_start` default → True),
  `1690824` (`spectral.basis_source` ablation + `_substitute_basis`).
- §7.1 high-C logs: `/nas/lnt/stud/ge27yuv/runs/reddit_{body,title}_coarse_highC` (108 runs,
  executed 2026-07-14, harvested into this record 2026-07-20).
- §10.9 logs: `runs/{dec_abl_uci, dec_abl_btc, depth_abl}` (189 runs, wandb-tagged `dec-abl` /
  `depth-abl`). Commits: `680688a` (wandb group_suffix/extra_tags + f+pe group parts), `ee9fbf4`
  (exact solver restricted to the largest connected component — many-component graphs like reddit
  otherwise return an all-zero basis; caught in pre-launch validation, no runs affected).
- §10.10 probes (`analysis/probes/`, all zero-GPU, runnable locally): `cond_probe3.py` (federated
  conditional ceiling + cut-edge counting), `cond_probe5.py` (probe-MRR + placebo),
  `proto_fusion.py` (gate 1: fusion inside the real eval protocol). All three take
  `[dataset] [config] [Cs] [seeds]` positionally, resolve the repo root from `__file__`, and print
  a mean±std table; e.g. `python analysis/probes/proto_fusion.py uci config/uci_gru.yaml 1,3,7,9
  1234,1334,1434` (12 uci runs, ~5 min on the Mac).
- Gate-2 logs: `/nas/lnt/stud/ge27yuv/runs/gate2_btcotc` (8 probe logs + `progress.txt`, runner
  `runs/run_gate2.sh`, sim13, 2026-07-25). Gate 1 (uci) ran locally on the Mac, ~5 min total.
- Breadth + K-sensitivity logs: `/nas/lnt/stud/ge27yuv/runs/gate2b_breadth` (runners
  `runs/run_gate2b.sh` bitcoin_alpha + uci/bitcoin_otc K sweep, `run_gate2c.sh` the larger-K
  bitcoin cells, `run_gate2d.sh` as733 parallel by C). Each log ends in its own mean±std table;
  `progress.txt` records completion + rc per job.
- Analysis tooling: `analysis/compile_results.py` (RESULT-line harvest -> mean±std + coverage);
  master CSV of the control program: `runs/master_results.csv` (661 runs, 217 conditions,
  216/217 seed-complete as of 2026-07-25).
- §10.8 per-snapshot lines parsed from the same §10.7 logs (`grep 't=[0-9]* mrr='`); oracle
  probe re-run binned (scratchpad `oracle_probe2b.py`).
- Commits behind §10.7: `e525928` (`data_type=f+pe` input LapPE + `*_fixed` stability-matched
  basis controls), `bcc43af` (exact solver on the active subgraph — ARPACK stalls on the
  isolated-node clusters of early cumulative graphs). §10.7 logs:
  `/nas/lnt/stud/ge27yuv/runs/{pe_uci, pe_btcotc, pe_btcotc_keep, pe_as733}` (runner
  `runs/run_pe.sh`). Run pattern: `--set model.data_type=f+pe spectral.update_mode=<m>
  spectral.basis_source=<laplacian|shuffled_fixed|random_fixed> subgraph.num_subgraphs=<C>`.
- §10 raw logs (runners `runs/run_ablation.sh` fusion×basis and `runs/run_ablation2.sh` smodel×basis,
  not committed): Laplace basis control = `{abl_uci, abl_btcotc, abl_rb_c3, abl_rb_c7, abl_as733,
  abl_rtitle}`; SignNet basis control = `{abl2_rb_signnet, abl2_btcotc_signnet}`. §10 uses default
  random negatives. All under `spectral.deterministic_start=True` (the new default). All §10 sweeps
  are COMPLETE (2026-07-22): reddit_body SignNet keep-control null; bitcoin_otc SignNet control
  36/36 (the [prelim] non-null cell, §10.5); `signnet_rtitle_gap` 3/3 (closed the §8.4 hole).
- **NOTE:** every number in §2–§9 was produced with `spectral.deterministic_start=False`,
  which is no longer the default — exact reproduction of those rows needs it set back to False.
- §8 runs use `metric.hard_neg=degree` (auc/ap de-saturated ~0.86); §9 uses the default random
  negatives, so §9 auc is comparable to §6 but §8 auc is NOT.
