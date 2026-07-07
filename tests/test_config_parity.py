import os
import yaml
import pytest
from config.config import get_default_config, overlay_config

NAMES = [
    "uci_gru", "uci_ma",
    "bitcoin_alpha_gru", "bitcoin_alpha_ma",
    "bitcoin_otc_gru", "bitcoin_otc_ma",
    "as733_gru", "as733_ma",
    "reddit_body_gru", "reddit_body_ma",
    "reddit_title_gru", "reddit_title_ma"
]

APPLICABLE_PARAMS = {
    "dataset": [
        "name", "snapshot_freq", "split", "edge_dim", "edge_encoder",
        "edge_encoder_dim", "edge_encoder_bn", "node_encoder",
        "node_encoder_dim", "node_encoder_bn", "task", "task_type",
        "transductive", "shuffle", "split_method", "edge_negative_sampling_ratio",
        "negative_sample_weight"
    ],
    "model": [
        "type", "loss_fun", "edge_decoding", "match_upper", "edge_pred_shape"
    ],
    "gnn": [
        "layer_type", "embed_update_method", "dims_pre_mp", "dims",
        "dims_post_mp", "batchnorm", "act", "skip_connection", "l2norm",
        "agg", "dropout", "msg_direction", "normalize_adj", "skip_every",
        "only_update_top_state", "keep_edge", "keep_ratio_mode", "gru_kernel",
        "stage_type"
    ],
    "optim": [
        "optimizer", "base_lr", "weight_decay", "scheduler"
    ],
    "meta": [
        "is_meta", "method", "alpha"
    ],
    "train": [
        "num_epochs", "internal_validation_tolerance", "mode"
    ],
    "metric": [
        "mrr_method"
    ],
    "experimental": [
        "rank_eval_multiplier", "restrict_training_set"
    ]
}

EXCLUDED_ROLAND_KEYS = {
    "dataset": {
        "path", "format", "split_seed", "snapshot", "is_hetero", "load_cache",
        "premade_datasets", "include_node_features"
    },
    "model": {
        "data_type", "smodel_type", "fmodel_type", "fusion", "iterations",
        "local_epochs", "thresh"
    },
    "gnn": {
        "att_heads", "att_final_linear", "att_final_linear_bn", "mlp_update_layers"
    },
    "optim": {
        "momentum", "steps", "lr_decay"
    },
    "train": {
        "batch_size", "early_stopping", "eval_period", "ckpt_period",
        "ckpt_clean", "auto_resume", "stop_live_update_after"
    },
    "experimental": {
        "eval_seed"
    },
    "meta": set(),
    "metric": set()
}


@pytest.mark.parametrize("name", NAMES)
def test_config_parity(name):
    fedlap_path = f"config/{name}.yaml"
    centralized_path = f"tests/data/centralized_configs/{name}.yaml"
    
    assert os.path.exists(fedlap_path), f"FedLap config file {fedlap_path} does not exist"
    assert os.path.exists(centralized_path), f"Centralized config file {centralized_path} does not exist"
    
    fl = get_default_config()
    with open(fedlap_path) as f:
        overlay_config(fl, yaml.safe_load(f))
        
    with open(centralized_path) as f:
        cl = yaml.safe_load(f)
        
    mismatches = []
    for sec, keys in APPLICABLE_PARAMS.items():
        for k in keys:
            fl_val = fl[sec][k]
            cl_val = cl[sec][k]
            if isinstance(fl_val, list) and isinstance(cl_val, list):
                if fl_val != cl_val:
                    mismatches.append((f"{sec}.{k}", fl_val, cl_val))
            elif fl_val != cl_val:
                mismatches.append((f"{sec}.{k}", fl_val, cl_val))
                
    assert not mismatches, f"Config mismatches for {name}: {mismatches}"
    
    drifted_keys = []
    for sec in APPLICABLE_PARAMS.keys():
        if sec in fl and sec in cl and cl[sec] is not None:
            common = set(fl[sec]) & set(cl[sec].keys())
            for k in common:
                if k not in APPLICABLE_PARAMS[sec] and k not in EXCLUDED_ROLAND_KEYS[sec]:
                    drifted_keys.append(f"{sec}.{k}")
                    
    assert not drifted_keys, f"Undocumented common keys found (config drift) for {name}: {drifted_keys}"
