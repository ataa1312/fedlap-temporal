import logging

from configs.registry import Registry

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
_WANDB_MODES = {"online", "offline", "disabled"}
_AGGREGATIONS = {"fedavg"}
_WEIGHTINGS = {"node_count", "uniform"}
_SPECTRAL_SMODELS = {"SpectralLaplace", "LanczosLaplace", "SpectralDGCN", "LanczosDGCN"}


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

    # ----- hard checks (federated / spectral) ----- #
    _require_in(federated["aggregation"], _AGGREGATIONS, "federated.aggregation")
    _require_in(federated["weighting"], _WEIGHTINGS, "federated.weighting")

    if subgraph["num_subgraphs"] < 1:
        raise ValueError(
            f"subgraph.num_subgraphs must be >=1, got {subgraph['num_subgraphs']!r}"
        )

    if model["data_type"] in {"structure", "f+s"} and model["smodel_type"] in _SPECTRAL_SMODELS:
        if spectral["spectral_len"] <= 0:
            raise ValueError(
                f"spectral.spectral_len must be >0 for smodel_type={model['smodel_type']!r}, "
                f"got {spectral['spectral_len']!r}"
            )

    # ----- soft fixes ----- #
    if dataset["task_type"] == "classification" and model["loss_fun"] == "mse":
        logger.warning(
            "model.loss_fun=mse incompatible with classification task; "
            "switching to bce_with_logits"
        )
        model["loss_fun"] = "bce_with_logits"

    if gnn["layers_post_mp"] < 1:
        logger.warning("gnn.layers_post_mp must be >=1; clamping to 1")
        gnn["layers_post_mp"] = 1
