from config.registry import Registry

# Default configuration for the federated ROLAND pipeline. Built as a nested
# Registry (dict-style access; attribute access also works via the shim for the
# kept spectral/structural code). `get_default_config()` returns a fresh tree so
# repeated builds (tests, the import-time singleton) never collide.
#
# Sections fall into three groups:
#   - ROLAND (ported from the centralized codebase): dataset/train/model/gnn/
#     optim/meta/experimental/metric/wandb — drive the recurrent model + the
#     live-update loop.
#   - FedLap (kept for the spectral contribution): subgraph/spectral/
#     structure_model/feature_model + the FedLap-only keys merged onto dataset
#     and model.
#   - federated (new): cross-client aggregation knobs.


def _dataset() -> Registry:
    d = Registry("dataset")
    # ROLAND core
    d["name"] = None
    d["path"] = "data"
    d["format"] = "custom"
    d["task"] = "link_pred"
    d["task_type"] = "classification"
    d["transductive"] = True
    d["shuffle"] = True
    d["split"] = [0.8, 0.1, 0.1]            # edge-level per snapshot (live_update)
    d["split_seed"] = None                  # None -> follow global seed
    d["edge_negative_sampling_ratio"] = 1.0
    d["edge_dim"] = 0                       # raw edge-feature dim; >0 enables edge-aware convs
    d["edge_encoder"] = False
    d["edge_encoder_dim"] = 32
    d["edge_encoder_bn"] = True
    d["node_encoder"] = False
    d["node_encoder_dim"] = 128
    d["node_encoder_bn"] = True
    d["negative_sample_weight"] = "uniform"
    d["is_hetero"] = False
    d["split_method"] = "default"
    d["snapshot"] = True
    d["snapshot_freq"] = "W"                # 'D'/'W'/'M' or 'NNNs'
    # FedLap-kept keys (read by the spectral/partition code)
    d["multi_label"] = False
    d["normalize"] = True
    d["num_classes"] = None                 # set at runtime
    d["shape"] = None                       # set at runtime
    return d


def _subgraph() -> Registry:
    s = Registry("subgraph")
    s["num_subgraphs"] = 3                  # number of federated clients
    s["partitioning"] = "random"
    s["delta"] = None
    s["train_ratio"] = None
    s["test_ratio"] = None
    s["prune"] = None
    s["pruning_th"] = None
    return s


def _train() -> Registry:
    t = Registry("train")
    t["num_epochs"] = 0                     # per-snapshot max inner-loop iters
    t["batch_size"] = 512
    t["mode"] = "live_update"
    t["early_stopping"] = False
    t["eval_period"] = 10
    t["ckpt_period"] = 100
    t["auto_resume"] = False                # gate: when True, joint_train_w checkpoints + resumes
    t["ckpt_clean"] = True                  # delete the partial ckpt on successful completion
    t["ckpt_dir"] = "checkpoints"           # where {run_id}.ckpt / .done live
    t["internal_validation_tolerance"] = 5
    t["stop_live_update_after"] = 99999999
    t["num_runs"] = 1                       # FedLap multi-seed (also overridable via --repeat)
    return t


def _model() -> Registry:
    m = Registry("model")
    # ROLAND
    m["type"] = "recurrent_gnn"             # 'gnn', 'recurrent_gnn'
    m["loss_fun"] = "bce_with_logits"
    m["match_upper"] = True
    m["thresh"] = 0.5
    m["edge_decoding"] = "concat"           # 'dot', 'cosine_similarity', 'concat'
    m["edge_pred_shape"] = "label_index"
    # FedLap-kept (spectral smodel dispatch + structural reg)
    m["smodel_type"] = "LanczosLaplace"     # structure model; '' or None to disable spectral
    m["fmodel_type"] = "GNN"
    m["data_type"] = "f+s"                  # 'feature', 'structure', 'f+s', 'f+pe' (input LapPE)
    m["fusion"] = "add"                     # combine node(z)+spectral(S) at output: 'add' (FedLap-native,
                                            # smodel MLP matches z width) | 'concat' (head widened to 2d)
    m["weight_decay"] = 5e-4
    # FedLap-kept (static, non-temporal path): the original ModelConfig knobs the
    # static federated training reads. Distinct from the ROLAND optim/gnn/train
    # keys (model.lr vs optim.base_lr, model.dropout vs gnn.dropout,
    # model.iterations vs train.num_epochs) — both coexist during the migration.
    m["iterations"] = 200
    m["lr"] = 1e-3
    m["gnn_layer_type"] = "custom-gat"
    m["dropout"] = 0.1
    m["metric"] = "auc"
    m["num_samples"] = [5, 10]
    m["batch"] = False
    m["batch_size"] = 64
    m["local_epochs"] = 1
    return m


def _gnn() -> Registry:
    g = Registry("gnn")
    g["layer_type"] = "residual_edge_conv"
    g["dims_pre_mp"] = []                    # pre-MP MLP layer widths ([] = no pre-MP)
    g["dims"] = [64, 64]                     # MP layer widths (one entry per layer)
    g["dims_post_mp"] = []                   # post-MP head intermediate widths ([] = single head layer)
    g["stage_type"] = "stack"
    g["skip_every"] = 1
    g["batchnorm"] = True
    g["act"] = "prelu"
    g["dropout"] = 0.0
    g["agg"] = "add"
    g["normalize_adj"] = False
    g["msg_direction"] = "single"
    g["l2norm"] = True
    g["keep_edge"] = 0.5
    g["att_heads"] = 1
    g["att_final_linear"] = False
    g["att_final_linear_bn"] = False
    g["only_update_top_state"] = False
    g["embed_update_method"] = "gru"        # 'moving_average', 'masked_gru', 'gru'
    g["keep_ratio_mode"] = "linear"
    g["gru_kernel"] = "linear"
    g["mlp_update_layers"] = 2
    g["skip_connection"] = "affine"
    g["encoder_edge_drop"] = 0.0            # ABLATION: fraction of each snapshot's edges hidden
                                            # from the ENCODER's message passing, and from nothing
                                            # else. 0.0 = current behaviour. The kept subset is
                                            # fixed per (owner, snapshot) for the whole run, so this
                                            # starves message passing rather than acting as a
                                            # stochastic DropEdge regularizer. Evaluation targets,
                                            # negatives, the per-snapshot train/val/test edge split,
                                            # keep_ratio, client abstention and the cumulative union
                                            # the spectral basis is built on all keep the FULL edge
                                            # set -- see _precompute_encoder_edge_drop.
    return g


def _optim() -> Registry:
    o = Registry("optim")
    o["optimizer"] = "adam"
    o["base_lr"] = 1e-2
    o["weight_decay"] = 5e-4
    o["momentum"] = 0.9
    o["scheduler"] = "none"
    o["steps"] = [30, 60, 90]
    o["lr_decay"] = 0.1
    return o


def _meta() -> Registry:
    m = Registry("meta")
    m["is_meta"] = False
    m["method"] = "moving_average"          # 'moving_average', 'online_mean'
    m["alpha"] = 0.9
    return m


def _experimental() -> Registry:
    e = Registry("experimental")
    e["rank_eval_multiplier"] = 1000
    e["restrict_training_set"] = -1
    e["eval_seed"] = None
    e["deterministic"] = False              # torch.use_deterministic_algorithms(True) at startup.
                                            # Works on CPU and CUDA (measured: sim10 RTX 4080,
                                            # bit-identical x2; main() sets CUBLAS_WORKSPACE_CONFIG
                                            # for the CUDA path, which is mandatory there).
                                            # Off by default: same seed reruns vary ~±0.008 MRR
                                            # (results.md §12b), enabling it makes them bit-exact
                                            # at full thread count for ~14% wall-clock. BUT it
                                            # converges to a DIFFERENT fixed point than the
                                            # non-deterministic path (different summation order),
                                            # so numbers produced with it on are NOT comparable to
                                            # the existing tables. Flip it deliberately, not
                                            # incidentally.
    return e


def _metric() -> Registry:
    m = Registry("metric")
    m["eval_scope"] = "auto"                # which test set the reported metrics are scored on:
                                            # 'auto'  = global when federated.fl else per-client (the
                                            #           historical coupling; keeps prior runs reproducible)
                                            # 'local' = always per-client (each client scored on its OWN
                                            #           subgraph) -- the apples-to-apples setting for
                                            #           comparing fl=true vs fl=false, since only the
                                            #           TRAINING then differs, not the test set
                                            # 'global'= always the stitched global graph. Needs fl=true:
                                            #           _eval_mrr decodes with the SERVER model, which is
                                            #           never aggregated (so never trained) when fl=false.
    m["mrr_method"] = "max"                 # 'min', 'max', 'mean'
    m["hard_neg"] = "random"                # discrimination negatives for auc/ap: 'random'
                                            # (deepsnap ~1:1, saturates ~0.96) or 'degree'
                                            # (degree-weighted hard, de-saturates auc/ap)
    m["repeat_new_split"] = False           # also report MRR split by whether the positive pair is
                                            # already in the cumulative union. Separates "the term
                                            # supplies structure lost to sharding" from "the term
                                            # encodes which pairs have interacted before" -- the
                                            # aggregate metric cannot tell those apart. Off by
                                            # default; enabling it costs one extra rank pass, no
                                            # extra decode, and does not change the headline value.
    m["mrr_filter"] = "split"               # what MRR negatives are forbidden from hitting:
                                            # 'split'    = the evaluated split's positives only
                                            #              (ROLAND-faithful, train_utils.py:230-235)
                                            # 'snapshot' = also every edge of the target snapshot,
                                            #              so its train/val positives stop being
                                            #              eligible negatives. Resamples to keep K.
                                            # 'both'     = report each; headline stays 'split'.
                                            # The classification metrics already forbid the whole
                                            # target snapshot -- this knob measures that asymmetry.
    return m


def _feature_model() -> Registry:
    f = Registry("feature_model")
    f["gnn_layer_sizes"] = [256, 128]
    f["heads"] = [32, 16]
    f["DGCN_layer_sizes"] = [64]
    f["mlp_layer_sizes"] = [256]
    f["DGCN_layers"] = 2
    f["dropout"] = 0.1
    f["residual"] = False
    f["use_edge_features"] = False
    f["edge_dimension"] = None
    return f


def _structure_model() -> Registry:
    s = Registry("structure_model")
    s["GNN_structure_layers_sizes"] = []
    s["DGCN_structure_layers_sizes"] = [128]   # code reads the plural attr name
    s["DGCN_layers"] = 10
    s["structure_type"] = "hop2vec"
    s["num_structural_features"] = 512
    # SFV lifetime. The learnable W is built once in initialize_FL, before the
    # snapshot loop, and nothing resets it — so it already carries its trained
    # values across every snapshot. Both knobs default to that behaviour; they
    # exist to take it away, which is the only way to measure what it is worth.
    #   reset_per_snapshot: re-draw W at every snapshot boundary, every owner.
    #                       Removes accumulation across snapshots; still trains
    #                       within one.
    #   freeze:             hold W at its random init for the whole run — out of
    #                       the optimizer, out of structural FedAvg. Separates
    #                       "trained at all" from "trained and carried", which
    #                       the reset arm alone confounds.
    s["sfv_reset_per_snapshot"] = False
    s["freeze_sfv"] = False
    s["estimate"] = False
    s["num_mp_vectors"] = 10
    s["rw_len"] = 50
    s["gnn_epochs"] = 500
    s["mlp_epochs"] = 50
    return s


def _spectral() -> Registry:
    s = Registry("spectral")
    s["spectral_len"] = 300
    s["lanczos_iter"] = 400
    s["L_type"] = "sym"                     # 'normal', 'rw', 'sym'. sym is the default because the
                                            # Rayleigh-Ritz tracker projects as H = Q^T L Q and then
                                            # calls eigh, which assumes symmetry. L_rw is NOT
                                            # symmetric, so H picks up complex eigenvalues
                                            # (max|Im| ~2e-2, residuals ~1e-2) and the code discards
                                            # the imaginary part -- an approximation, not an
                                            # identity. Under sym that branch never fires and the
                                            # tracking is exact (residual ~1e-7). See graph.py:444.
                                            # Only the tracking path reads this; exact and chebyshev
                                            # always build L_sym on the giant component.
    s["method"] = "arnoldi"                 # 'arnoldi', 'lanczos'
    s["deterministic_start"] = True         # arnoldi start: fixed-seed vector (stable gauge across
                                            # snapshots -> recompute eigvecs move only as the graph
                                            # moves) vs a re-drawn random start each solve
    s["basis_source"] = "laplacian"         # ABLATION: 'laplacian' (real eigenbasis) vs null controls
                                            # 'random' (Haar orthonormal, same shape) / 'shuffled'
                                            # (real eigvecs, node rows permuted); '*_fixed' variants
                                            # freeze the draw across snapshots (stability-matched
                                            # structure controls) -- see _substitute_basis
    s["pe_dim"] = 50                        # data_type=f+pe: k lowest exact sym-Laplacian eigvecs
                                            # concatenated to node features at the INPUT (LapPE)
    s["solver"] = "arnoldi"                 # how the smodel's basis is computed: 'arnoldi' (Krylov
                                            # estimate, the historical path -- ~0.27-0.62 subspace
                                            # overlap with the truth), 'exact' (sym-Laplacian eigh /
                                            # shift-invert), 'chebyshev' (filtered subspace iteration:
                                            # exact-quality at 3-24x less wall-clock, warm-started
                                            # from the previous basis under update_mode=update)
    s["es_features"] = "spec"               # data_type=f+es feature set: 'spec' (rotation-invariant
                                            # spectral affinities), 'persist' (the 1-bit "pair already
                                            # exists" control -- §10.11 scored it ABOVE spectral on
                                            # every dataset), 'both' (does the spectrum add anything
                                            # on top of explicit history access?), 'cn' (log1p common
                                            # neighbours on the same cumulative graph -- the OTHER
                                            # baseline §10.11 pre-registered; an offline probe puts it
                                            # ABOVE the spectral affinity on both reddit graphs, so a
                                            # new-pair claim is not attributable to the spectrum until
                                            # it clears this arm. NOTE persist is NOT an informative
                                            # control on the new subset: it is the split label handed
                                            # back as a feature, so its new-pair penalty is forced by
                                            # rank arithmetic (results.md §20.4a). cn is.
    s["es_spec_parts"] = "phi+cos+lev"      # ATTRIBUTION ABLATION over the 'spec' feature block:
                                            # '+'-joined subset of 'phi' (learnable heat-kernel
                                            # affinities), 'cos' (the unfiltered affinity the probes
                                            # measured) and 'lev' (the leverage term). The default is
                                            # all three; single-part arms say which one carries the
                                            # gain. Ignored when es_features='persist'
    s["robust_sign"] = False                # eigvec sign canon: by largest-|component| entry (stable)
                                            # vs the near-zero column-sum (noisy -> flips across snaps)
    s["regularizer_coef"] = 0
    s["matrix"] = "lap"                     # 'adj', 'lap', 'inc'
    s["decompose"] = "eigh"                 # 'svd', 'eigh'
    s["update_mode"] = "update"             # 'keep', 'update', 'recompute'
    s["recompute_prob"] = 0.0               # update mode: per-snapshot Bernoulli full re-Lanczos (basis refresh)
    # 'auto' | True | False. Procrustes rotates U toward snapshot 0's basis to
    # stabilise the gauge across snapshots. It is worth it wherever the readout
    # sees U's COORDINATES (f+s, f+pe, structure).
    #
    # It is NOT worth it on f+es, which is why 'auto' turns it off there: that
    # path's features are rotation-invariant by construction (cos and lev are
    # exactly invariant under any orthogonal R), so the rotation buys nothing --
    # and it actively breaks the phi block, because procrustes_project rotates U
    # without rotating D, so (U, Lambda) no longer satisfies L = U Lambda U^T and
    # the filtered block stops being an entry of any M_f.
    #
    # An explicit True/False still wins on every path, so the on/off A/B on f+es
    # stays runnable via --set spectral.use_procrustes=true.
    s["use_procrustes"] = "auto"
    # Age kernel weighting the CUMULATIVE adjacency the eigenbasis is built on.
    # Today every edge that ever appeared counts once and forever: graph.py
    # binarizes with ((A + A.T) > 0), so multiplicity is discarded too, and only
    # 14% of the cumulative edge mass belongs to the current snapshot
    # (|E_t|/|cum_t| = 0.138, analysis/probes/encoder_edge_budget.py). The basis
    # therefore describes accumulated history, not the current graph.
    #
    # Weight of an undirected edge e at snapshot t:
    #     w_t(e) = sum over snapshots s <= t containing e of  f(t - s)
    #
    #   none      f = 1 if ever seen        DEFAULT, bit-identical to the binary union
    #   count     f = 1 per appearance      frequency only, no recency (isolates re-weighting)
    #   harmonic  f = 1 / (age + 1)
    #   exp       f = cum_decay_param ** age
    #   window    f = 1 if age < cum_decay_param else 0
    #
    # count/harmonic/exp keep every ever-seen edge strictly positive, so the
    # active set (deg > 0) and the giant component are unchanged -- they alter
    # the metric, not the support. window can zero an edge outright and so also
    # changes coverage; read it against exp at a matched horizon, not against none.
    #
    # L_sym = I - D^-1/2 A D^-1/2 is invariant to A -> cA, so only the SHAPE of
    # the kernel matters; there is no normalisation to tune.
    #
    # Applies to the eigenbasis ONLY. The unweighted union still defines the
    # repeat/new split and the 'persist' feature -- see DynamicServer.
    s["cum_decay"] = "none"
    s["cum_decay_param"] = 0.9              # exp: decay factor in (0,1]; window: int horizon >=1
    s["output_bn"] = True                   # BatchNorm on smodel output S (bounds spectral amplification; gamma zero-init)
    # SignNet smodel (model.smodel_type=SignNet): S = rho(sum_i [phi(u_i)+phi(-u_i)]).
    # phi is the shared per-entry map R^1->phi_dims (last = phi_out); rho maps
    # phi_out -> rho_dims -> gnn.dims[-1]. Ignored by the Laplace smodels.
    s["signnet_phi_dims"] = [64, 64]        # phi hidden+output widths (phi_out = last)
    s["signnet_rho_dims"] = [64]            # rho hidden widths (output = fusion width)
    return s


def _federated() -> Registry:
    f = Registry("federated")
    f["fl"] = True                          # FedLap's FL flag. False = local-only baseline: each client
                                            # trains on its own subgraph and evaluates on its own next
                                            # snapshot, with NO broadcast and NO aggregation (metrics
                                            # node-count-weight-averaged). The utility FLOOR that
                                            # federation must beat to be worth anything.
    f["aggregation"] = "fedavg"             # cross-client weight aggregation
    f["weighting"] = "node_count"           # 'node_count', 'uniform'
    f["sfv_share"] = "local"                # learnable spectral W: 'local' per-client (FedLap joint_train_w
                                            # state_dict semantics) | 'avg' joins the weight averaging
    return f


def _node2vec() -> Registry:
    # FedLap-kept (static path): node2vec/hop2vec structural features (Node2Vec.py).
    n = Registry("node2vec")
    n["epochs"] = 50
    n["walk_length"] = 20
    n["context_size"] = 10
    n["walks_per_node"] = 10
    n["lr"] = 0.01
    n["batch_size"] = 128
    n["num_negative_samples"] = 1
    n["p"] = 1
    n["q"] = 1
    n["show_bar"] = True
    return n


def _wandb() -> Registry:
    w = Registry("wandb")
    w["run_name"] = ""
    w["project"] = "dynamic-fedlap"
    w["job_type"] = "edge-prediction"
    w["group"] = ""
    w["group_suffix"] = ""                  # appended to the auto-built group (sweep-specific runs)
    w["extra_tags"] = []                    # appended to the auto-built tags, e.g. [depth-abl,L4]
    w["mode"] = "offline"                   # 'online', 'offline', 'disabled'
    return w


def overlay_config(config: Registry, data: dict) -> None:
    """Deep-merge a (possibly partial) dict onto a config Registry in place.

    Lets YAML files specify only the keys that differ from the defaults (unlike
    ``Registry.from_yaml``, which replaces the whole tree). Nested dicts recurse
    into sub-Registries; scalar/list values overwrite; new keys are added.
    """
    for key, val in data.items():
        if key in config and isinstance(config[key], Registry) and isinstance(val, dict):
            overlay_config(config[key], val)
        else:
            config[key] = val


def get_default_config() -> Registry:
    config = Registry("main")
    config["device"] = "auto"
    config["outdir"] = "results"
    config["seed"] = 1234
    config["num_workers"] = 0
    config["num_threads"] = 6
    config["print"] = "both"
    config["metric_best"] = "auto"
    config["remark"] = ""
    config["downstream_task"] = "edge-prediction"
    config["experiment"] = 0                 # FedLap-kept top-level keys
    config["num_runs"] = 1

    config["dataset"] = _dataset()
    config["subgraph"] = _subgraph()
    config["train"] = _train()
    config["model"] = _model()
    config["gnn"] = _gnn()
    config["optim"] = _optim()
    config["meta"] = _meta()
    config["experimental"] = _experimental()
    config["metric"] = _metric()
    config["feature_model"] = _feature_model()
    config["structure_model"] = _structure_model()
    config["spectral"] = _spectral()
    config["federated"] = _federated()
    config["node2vec"] = _node2vec()
    config["wandb"] = _wandb()
    return config
