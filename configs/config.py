from configs.registry import Registry

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
    t["auto_resume"] = False
    t["ckpt_clean"] = True
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
    m["data_type"] = "f+s"                  # 'feature', 'structure', 'f+s'
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
    return e


def _metric() -> Registry:
    m = Registry("metric")
    m["mrr_method"] = "max"                 # 'min', 'max', 'mean'
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
    s["L_type"] = "rw"                      # 'normal', 'rw', 'sym'
    s["method"] = "arnoldi"                 # 'arnoldi', 'lanczos'
    s["regularizer_coef"] = 0
    s["matrix"] = "lap"                     # 'adj', 'lap', 'inc'
    s["decompose"] = "eigh"                 # 'svd', 'eigh'
    s["update_mode"] = "update"             # 'keep', 'update', 'recompute'
    s["recompute_prob"] = 0.0               # update mode: per-snapshot Bernoulli full re-Lanczos (basis refresh)
    s["use_procrustes"] = True
    s["output_bn"] = True                   # BatchNorm on smodel output S (bounds spectral amplification; gamma zero-init)
    return s


def _federated() -> Registry:
    f = Registry("federated")
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
