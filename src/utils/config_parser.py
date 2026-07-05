import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)

        self.dataset = DatasetConfig(self.config["dataset"])
        self.subgraph = SubgraphConfig(self.config["subgraph"])
        self.model = ModelConfig(self.config["model"])
        self.feature_model = FeatureModelConfig(self.config["feature_model"])
        self.structure_model = StructureModelConfig(self.config["structure_model"])
        self.spectral = SpectralConfig(self.config["spectral"])
        self.node2vec = Node2VecConfig(self.config["node2vec"])
        self.fedsage = FedSAGEConfig(self.config["fedsage"])
        self.fedpub = PubMedConfig(self.config["fedpub"])
        self.fedgcn = FedGCNConfig(self.config["fedgcn"])

    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config


class DatasetConfig:
    def __init__(self, dataset):
        self.load_config(dataset)

    def load_config(self, dataset):
        self.dataset_name = dataset["dataset_name"]
        self.multi_label = dataset["multi_label"]


class SubgraphConfig:
    def __init__(self, subgraph):
        self.load_config(subgraph)

    def load_config(self, subgraph):
        self.num_subgraphs = subgraph["num_subgraphs"]
        self.delta = subgraph["delta"]
        self.train_ratio = subgraph["train_ratio"]
        self.test_ratio = subgraph["test_ratio"]
        self.partitioning = subgraph["partitioning"]
        self.prune = subgraph["prune"]
        self.pruning_th = subgraph["pruning_th"]


class ModelConfig:
    def __init__(self, model):
        self.load_config(model)

    def load_config(self, model):
        self.num_samples = model["num_samples"]
        self.batch = model["batch"]
        self.batch_size = model["batch_size"]
        self.local_epochs = model["local_epochs"]
        self.lr = model["lr"]
        self.weight_decay = model["weight_decay"]
        self.gnn_layer_type = model["gnn_layer_type"]
        self.smodel_type = model["smodel_type"]
        self.fmodel_type = model["fmodel_type"]
        self.dropout = model["dropout"]
        self.iterations = model["iterations"]
        self.metric = model["metric"]


class FeatureModelConfig:
    def __init__(self, feature_model):
        self.load_config(feature_model)

    def load_config(self, feature_model):
        self.gnn_layer_sizes = feature_model["gnn_layer_sizes"]
        self.mlp_layer_sizes = feature_model["mlp_layer_sizes"]
        self.DGCN_layer_sizes = feature_model["DGCN_layer_sizes"]
        self.DGCN_layers = feature_model["DGCN_layers"]


class StructureModelConfig:
    def __init__(self, structure_model):
        self.load_config(structure_model)

    def load_config(self, structure_model):
        self.GNN_structure_layers_sizes = structure_model["GNN_structure_layers_sizes"]
        self.DGCN_structure_layers_sizes = structure_model["DGCN_structure_layers_size"]
        self.DGCN_layers = structure_model["DGCN_layers"]
        self.structure_type = structure_model["structure_type"]
        self.num_structural_features = structure_model["num_structural_features"]
        self.estimate = structure_model["estimate"]
        self.num_mp_vectors = structure_model["num_mp_vectors"]
        self.rw_len = structure_model["rw_len"]
        self.gnn_epochs = structure_model["gnn_epochs"]
        self.mlp_epochs = structure_model["mlp_epochs"]


class SpectralConfig:
    def __init__(self, spectral_model):
        self.load_config(spectral_model)

    def load_config(self, spectral_model):
        self.spectral_len = spectral_model["spectral_len"]
        self.lanczos_iter = spectral_model["lanczos_iter"]
        self.method = spectral_model["method"]
        self.L_type = spectral_model["L_type"]
        self.regularizer_coef = spectral_model["regularizer_coef"]
        self.matrix = spectral_model["matrix"]
        self.decompose = spectral_model["decompose"]


class Node2VecConfig:
    def __init__(self, node2vec):
        self.load_config(node2vec)

    def load_config(self, node2vec):
        self.epochs = node2vec["epochs"]
        self.walk_length = node2vec["walk_length"]
        self.context_size = node2vec["context_size"]
        self.walks_per_node = node2vec["walks_per_node"]
        self.lr = node2vec["lr"]
        self.batch_size = node2vec["batch_size"]
        self.num_negative_samples = node2vec["num_negative_samples"]
        self.p = node2vec["p"]
        self.q = node2vec["q"]
        self.show_bar = node2vec["show_bar"]


class FedSAGEConfig:
    def __init__(self, fedsage):
        self.load_config(fedsage)

    def load_config(self, fedsage):
        self.neighgen_epochs = fedsage["neighgen_epochs"]
        self.neighgen_lr = fedsage["neighgen_lr"]
        self.neighen_feature_gen = fedsage["neighen_feature_gen"]
        self.num_pred = fedsage["num_pred"]
        self.latent_dim = fedsage["latent_dim"]
        self.hidden_layer_sizes = fedsage["hidden_layer_sizes"]
        self.impaired_train_nodes_ratio = fedsage["impaired_train_nodes_ratio"]
        self.impaired_test_nodes_ratio = fedsage["impaired_test_nodes_ratio"]
        self.hidden_portion = fedsage["hidden_portion"]
        self.use_inter_connections = fedsage["use_inter_connections"]
        self.a = fedsage["a"]
        self.b = fedsage["b"]
        self.c = fedsage["c"]


class PubMedConfig:
    def __init__(self, fedpub):
        self.load_config(fedpub)

    def load_config(self, fedpub):
        self.epochs = fedpub["epochs"]
        self.frac = fedpub["frac"]
        self.clsf_mask_one = fedpub["clsf_mask_one"]
        self.laye_mask_one = fedpub["laye_mask_one"]
        self.norm_scale = fedpub["norm_scale"]
        self.lr = fedpub["lr"]
        self.weight_decay = fedpub["weight_decay"]
        self.n_dims = fedpub["n_dims"]
        self.agg_norm = fedpub["agg_norm"]
        self.n_proxy = fedpub["n_proxy"]
        self.l1 = fedpub["l1"]
        self.loc_l2 = fedpub["loc_l2"]


class FedGCNConfig:
    def __init__(self, fedpub):
        self.load_config(fedpub)

    def load_config(self, feedgcn):
        self.num_hops = feedgcn["num_hops"]
        self.iid_beta = feedgcn["iid_beta"]
