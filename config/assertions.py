import logging

from config.registry import Registry

__all__ = ["assert_cfg"]


logger = logging.getLogger(__name__)


_TASKS = {"node", "edge", "graph", "link_pred"}
_TASK_TYPES = {"classification", "regression"}
_OPTIMIZERS = {"adam", "adamw", "sgd"}
_SCHEDULERS = {"none", "steps", "cos"}
_EDGE_DECODINGS = {"dot", "cosine_similarity", "concat"}
_EMBED_UPDATE_METHODS = {"moving_average", "masked_gru", "gru"}
_META_METHODS = {"none", "moving_average", "online_mean"}
_TRAIN_MODES = {"live_update"}
_MRR_METHODS = {"min", "max", "mean"}
_MRR_FILTERS = {"split", "snapshot", "both"}
_EVAL_SCOPES = {"auto", "local", "global"}
_WANDB_MODES = {"online", "offline", "disabled"}
_AGGREGATIONS = {"fedavg"}
_WEIGHTINGS = {"node_count", "uniform"}
_SPECTRAL_SMODELS = {"SpectralLaplace", "LanczosLaplace", "SpectralDGCN", "LanczosDGCN", "SignNet"}
_FUSIONS = {"add", "concat"}
_DATA_TYPES = {"feature", "f+s", "structure", "f+pe", "f+es"}
_UPDATE_MODES = {"keep", "update", "recompute"}
_SOLVERS = {"arnoldi", "exact", "chebyshev"}
_BASIS_SOURCES = {"laplacian", "random", "shuffled", "random_fixed", "shuffled_fixed"}
_CUM_DECAYS = {"none", "count", "harmonic", "exp", "window"}
_SFV_SHARES = {"local", "avg"}


def _require_in(value, allowed: set, path: str) -> None:
    if value not in allowed:
        raise ValueError(f"{path}={value!r} is not one of {sorted(allowed)}")


def assert_cfg(config: Registry) -> None:
    dataset = config["dataset"]
    train = config["train"]
    model = config["model"]
    gnn = config["gnn"]
    optim = config["optim"]
    meta = config["meta"]
    metric = config["metric"]
    subgraph = config["subgraph"]
    federated = config["federated"]
    spectral = config["spectral"]

    # ----- hard checks (ROLAND) ----- #
    _require_in(dataset["task"], _TASKS, "dataset.task")
    _require_in(dataset["task_type"], _TASK_TYPES, "dataset.task_type")
    _require_in(optim["optimizer"], _OPTIMIZERS, "optim.optimizer")
    _require_in(optim["scheduler"], _SCHEDULERS, "optim.scheduler")
    _require_in(train["mode"], _TRAIN_MODES, "train.mode")
    _require_in(metric["mrr_method"], _MRR_METHODS, "metric.mrr_method")
    if "mrr_filter" in metric:
        _require_in(metric["mrr_filter"], _MRR_FILTERS, "metric.mrr_filter")
    _require_in(
        gnn["embed_update_method"], _EMBED_UPDATE_METHODS, "gnn.embed_update_method"
    )

    if dataset["task"] == "link_pred":
        _require_in(model["edge_decoding"], _EDGE_DECODINGS, "model.edge_decoding")

    if meta["is_meta"]:
        _require_in(meta["method"], _META_METHODS, "meta.method")

    if "wandb" in config and "mode" in config["wandb"]:
        _require_in(config["wandb"]["mode"], _WANDB_MODES, "wandb.mode")

    es = train["early_stopping"]
    if not (isinstance(es, bool) or (isinstance(es, int) and es > 0)):
        raise ValueError(
            f"train.early_stopping must be bool or positive int, got {es!r}"
        )

    exp = config["experimental"]
    if "deterministic" in exp and not isinstance(exp["deterministic"], bool):
        raise ValueError(
            "experimental.deterministic must be a bool (true = "
            f"torch.use_deterministic_algorithms at startup), got {exp['deterministic']!r}"
        )

    if "cum_decay" in spectral:
        _require_in(spectral["cum_decay"], _CUM_DECAYS, "spectral.cum_decay")
        _cd, _cp = spectral["cum_decay"], spectral["cum_decay_param"]
        # Only exp and window read the parameter; the others take none, so an
        # out-of-range value there is inert rather than an error.
        if _cd == "exp" and not (isinstance(_cp, (int, float)) and 0.0 < _cp <= 1.0):
            raise ValueError(
                "spectral.cum_decay_param must be a decay factor in (0, 1] for "
                f"cum_decay='exp', got {_cp!r}"
            )
        if _cd == "window" and not (isinstance(_cp, int) and not isinstance(_cp, bool) and _cp >= 1):
            raise ValueError(
                "spectral.cum_decay_param must be a positive integer horizon for "
                f"cum_decay='window', got {_cp!r}"
            )

    _proc = spectral["use_procrustes"]
    if not (isinstance(_proc, bool) or _proc == "auto"):
        raise ValueError(
            "spectral.use_procrustes must be a bool or 'auto' ('auto' = on for "
            f"f+s/f+pe/structure, off for f+es), got {_proc!r}"
        )

    sm = config["structure_model"]
    for _k in ("sfv_reset_per_snapshot", "freeze_sfv"):
        if _k in sm and not isinstance(sm[_k], bool):
            raise ValueError(f"structure_model.{_k} must be a bool, got {sm[_k]!r}")

    # ----- hard checks (federated / spectral) ----- #
    if not isinstance(federated["fl"], bool):
        raise ValueError(
            "federated.fl must be a bool (true = federated, false = local-only floor), "
            f"got {federated['fl']!r}"
        )
    _require_in(metric["eval_scope"], _EVAL_SCOPES, "metric.eval_scope")
    if metric["eval_scope"] == "global" and not federated["fl"]:
        raise ValueError(
            "metric.eval_scope='global' requires federated.fl=true: the global eval decodes "
            "the stitched z with the SERVER model, which is never aggregated (so never "
            "trained) when fl=false. Use eval_scope='local' to compare fl=true vs fl=false."
        )
    _require_in(federated["aggregation"], _AGGREGATIONS, "federated.aggregation")
    _require_in(federated["weighting"], _WEIGHTINGS, "federated.weighting")

    if subgraph["num_subgraphs"] < 1:
        raise ValueError(
            f"subgraph.num_subgraphs must be >=1, got {subgraph['num_subgraphs']!r}"
        )

    # data_type + spectral.update_mode are left null in the dataset configs so a run
    # must choose the experiment mode explicitly (--set): feature needs only
    # data_type; f+s / structure also need spectral.update_mode.
    if model["data_type"] is None:
        raise ValueError(
            "model.data_type is None — choose the experiment mode explicitly, "
            "e.g. --set model.data_type=feature (or f+s / structure)"
        )
    _require_in(model["data_type"], _DATA_TYPES, "model.data_type")
    _require_in(model["fusion"], _FUSIONS, "model.fusion")
    _require_in(federated["sfv_share"], _SFV_SHARES, "federated.sfv_share")

    if model["data_type"] in {"structure", "f+s", "f+pe", "f+es"}:
        if spectral["update_mode"] is None:
            raise ValueError(
                "spectral.update_mode is None — for data_type=f+s/structure/f+pe/f+es choose it "
                "explicitly, e.g. --set spectral.update_mode=update (or keep / recompute)"
            )
        _require_in(spectral["update_mode"], _UPDATE_MODES, "spectral.update_mode")
        _require_in(spectral["basis_source"], _BASIS_SOURCES, "spectral.basis_source")
        # Dispatch is if/elif/else with the Krylov path as the fallback, so an
        # unrecognised name would silently select arnoldi and look like a real run.
        if "solver" in spectral:
            _require_in(spectral["solver"], _SOLVERS, "spectral.solver")
        if model["data_type"] in {"f+pe", "f+es"} and spectral["pe_dim"] <= 0:
            raise ValueError(
                f"spectral.pe_dim must be >0 for data_type={model['data_type']}, got {spectral['pe_dim']!r}"
            )
        if (model["data_type"] == "f+es" and "solver" in spectral
                and spectral["solver"] == "arnoldi"):
            # the Krylov estimate overlaps the true low subspace by only ~0.27-0.62
            # (results.md §10.12); an invariant readout over it reads noise
            raise ValueError(
                "data_type=f+es needs spectral.solver='chebyshev' or 'exact' — the arnoldi "
                "estimate does not resolve the clustered low spectrum (results.md §10.12)"
            )
        # cum_decay only reaches the operator through calc_eigs_chebyshev and
        # calc_eigs_exact_sym. The Krylov path (calc_eignvalues) and the tracking
        # path (update_eigpairs) both rebuild L via Graph.create_L -> create_adj,
        # which hard-codes edge_weight=ones (src/utils/utils.py:372) and has no
        # weight hook. On those paths a decayed run is bit-identical to none while
        # _run_id still stamps `cum-<kernel>` -- a null that is pure plumbing and
        # reads exactly like a real negative result. Refuse it instead.
        #   update_mode='keep' computes only t=0, where every kernel gives f(0)=1,
        #   so it is inert for all of them.
        #   solver='exact' + update_mode='update' is inert too: t=0 is f(0)=1 and
        #   every t>0 goes through the unweighted update_eigpairs.
        if "cum_decay" in spectral and spectral["cum_decay"] != "none":
            _eff = "exact" if model["data_type"] == "f+pe" else spectral["solver"]
            _mode = spectral["update_mode"]
            _treated = (
                (_eff == "chebyshev" and _mode in ("update", "recompute"))
                or (_eff == "exact" and _mode == "recompute")
            )
            if not _treated:
                raise ValueError(
                    f"spectral.cum_decay={spectral['cum_decay']!r} has NO EFFECT with "
                    f"solver={_eff!r} + update_mode={_mode!r}: that path builds the "
                    "Laplacian through create_adj, which ignores edge weights, so the "
                    "basis would be identical to cum_decay='none' while the run id "
                    "still records the kernel. Use solver='chebyshev' with "
                    "update_mode='update'|'recompute', or solver='exact' with "
                    "update_mode='recompute'."
                )
        if model["smodel_type"] in _SPECTRAL_SMODELS and spectral["spectral_len"] <= 0:
            raise ValueError(
                f"spectral.spectral_len must be >0 for smodel_type={model['smodel_type']!r}, "
                f"got {spectral['spectral_len']!r}"
            )
        if not 0.0 <= spectral["recompute_prob"] <= 1.0:
            raise ValueError(
                f"spectral.recompute_prob must be in [0, 1], got {spectral['recompute_prob']!r}"
            )

    # ----- soft fixes ----- #
    if dataset["task_type"] == "classification" and model["loss_fun"] == "mse":
        logger.warning(
            "model.loss_fun=mse incompatible with classification task; "
            "switching to bce_with_logits"
        )
        model["loss_fun"] = "bce_with_logits"

    if not gnn["dims"]:
        raise ValueError("gnn.dims must list at least one MP layer width")

    if "encoder_edge_drop" in gnn:
        p = gnn["encoder_edge_drop"]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(
                f"gnn.encoder_edge_drop must be a float in [0, 1), got {p!r}"
            )
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(
                f"gnn.encoder_edge_drop must be in [0, 1) (1.0 would leave the encoder "
                f"no message-passing edges at all), got {p!r}"
            )
