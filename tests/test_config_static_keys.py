from configs.config import get_default_config

def test_top_level_keys():
    config = get_default_config()
    keys = ["experiment", "num_runs", "seed", "downstream_task"]
    missing = [k for k in keys if k not in config]
    assert not missing, f"Missing top-level keys: {missing}"

def test_model_keys():
    config = get_default_config()
    keys = [
        "iterations", "lr", "gnn_layer_type", "dropout", "metric", "smodel_type",
        "fmodel_type", "data_type", "weight_decay", "num_samples", "batch",
        "batch_size", "local_epochs"
    ]
    section = config["model"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing model keys: {missing}"

def test_feature_model_keys():
    config = get_default_config()
    keys = [
        "gnn_layer_sizes", "mlp_layer_sizes", "heads", "dropout", "residual",
        "use_edge_features", "edge_dimension", "DGCN_layer_sizes", "DGCN_layers"
    ]
    section = config["feature_model"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing feature_model keys: {missing}"

def test_structure_model_keys():
    config = get_default_config()
    keys = [
        "GNN_structure_layers_sizes", "DGCN_structure_layers_sizes", "DGCN_layers",
        "structure_type", "num_structural_features", "estimate",
        "num_mp_vectors", "rw_len", "gnn_epochs", "mlp_epochs"
    ]
    section = config["structure_model"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing structure_model keys: {missing}"

def test_spectral_keys():
    config = get_default_config()
    keys = [
        "spectral_len", "lanczos_iter", "method", "L_type", "regularizer_coef",
        "matrix", "decompose", "update_mode", "use_procrustes"
    ]
    section = config["spectral"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing spectral keys: {missing}"

def test_subgraph_keys():
    config = get_default_config()
    keys = ["num_subgraphs", "partitioning", "delta", "train_ratio", "test_ratio", "prune"]
    section = config["subgraph"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing subgraph keys: {missing}"

def test_dataset_keys():
    config = get_default_config()
    keys = ["name", "multi_label", "normalize"]
    section = config["dataset"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing dataset keys: {missing}"

def test_node2vec_keys():
    config = get_default_config()
    keys = [
        "epochs", "walk_length", "context_size", "walks_per_node", "lr",
        "batch_size", "num_negative_samples", "p", "q", "show_bar"
    ]
    section = config["node2vec"]
    missing = [k for k in keys if k not in section]
    assert not missing, f"Missing node2vec keys: {missing}"
