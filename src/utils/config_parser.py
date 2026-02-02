import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)

        # Top-level attributes
        self.seed = self.config["seed"]
        self.experiment = self.config["experiment"]
        self.num_runs = self.config["num_runs"]
        self.downstream_task = self.config[
            "downstream_task"
        ]  # Note: preserving typo from original code "downstream_tasl" as key might be that in yaml

        # Existing configs
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

        # New configs from edge_prediction_config.py
        self.dynamic = DynamicConfig(self.config["dynamic"])
        self.wandb = WandbConfig(self.config["wandb"])

        self.dynamic.evaluation.data.is_directed = self.dataset.is_directed

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
        self.is_directed = dataset["is_directed"]
        self.shape = dataset["shape"]
        self.num_classes = dataset["num_classes"]
        self.normalize = dataset["normalize"]


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
        self.data_type = model["data_type"]
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
        self.heads = feature_model["heads"]
        self.dropout = feature_model["dropout"]
        self.residual = feature_model["residual"]
        self.use_edge_features = feature_model["use_edge_features"]
        if self.use_edge_features:
            self.edge_dimension = self.gnn_layer_sizes[-1]
        else:
            self.edge_dimension = 0
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
        self.update_mode = spectral_model["update_mode"]
        self.use_procrustes = spectral_model["use_procrustes"]


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


class ModelTemporalConfig:
    def __init__(self, temporal):
        self.load_config(temporal)

    def load_config(self, temporal):
        self.input_dimension = temporal["input_dimension"]
        self.num_attention_layers = temporal["num_attention_layers"]
        self.attention_heads = temporal["attention_heads"]
        self.dropout = temporal["dropout"]
        self.num_snapshots = temporal["num_snapshots"]
        self.use_feedforward = temporal["use_feedforward"]
        self.residual = temporal["residual"]
        self.layer_norm = temporal["layer_norm"]


class TrainModelConfig:
    def __init__(self, model):
        self.load_config(model)

    def load_config(self, model):
        self.temporal = ModelTemporalConfig(model["temporal"])


class RandomWalkConfig:
    def __init__(self, random_walk):
        self.load_config(random_walk)

    def load_config(self, random_walk):
        self.num = random_walk["num"]
        self.length = random_walk["length"]
        self.window_size = random_walk["window_size"]


class NegativeSamplingConfig:
    def __init__(self, neg_sampling):
        self.load_config(neg_sampling)

    def load_config(self, neg_sampling):
        self.size = neg_sampling["size"]
        self.weight = neg_sampling["weight"]
        self.retries = neg_sampling["retries"]
        self.distortion = neg_sampling["distortion"]
        self.tempoal_window = neg_sampling["tempoal_window"]


class UnsupervisedTrainingConfig:
    def __init__(self, unsupervised):
        self.load_config(unsupervised)

    def load_config(self, unsupervised):
        self.neg_sampling = NegativeSamplingConfig(unsupervised["neg_sampling"])
        self.batch_size = unsupervised["batch_size"]
        self.weight_decay = unsupervised["weight_decay"]


class DynamicConfig:
    def __init__(self, train):
        self.load_config(train)

    def load_config(self, train):
        self.model = TrainModelConfig(train["model"])
        self.loss_fn = train["loss_fn"]
        self.min_snapshot = train["min_snapshot"]
        self.max_snapshot = train["max_snapshot"]
        self.random_walk = RandomWalkConfig(train["random_walk"])
        self.unsupervised = UnsupervisedTrainingConfig(train["unsupervised"])

        self.max_gradient_norm = train["max_gradient_norm"]
        self.evaluation = EvaluationConfig(train["evaluation"])


class BaseTxConfig:
    def __init__(self, tx):
        self.load_config(tx)

    def load_config(self, tx):
        self.fn = tx["fn"]
        self.lr = tx["lr"]
        self.moment = tx["moment"]
        self.weight_decay = tx["weight_decay"]
        self.lr_decay_per_worker_epoch = tx["lr_decay_per_worker_epoch"]
        self.type_decay_per_worker_epoch = tx["type_decay_per_worker_epoch"]


class ClassifierConfig:
    def __init__(self, classifier):
        self.load_config(classifier)

    def load_config(self, classifier):
        self.num_epoch = classifier["num_epoch"]
        self.loss_fn = classifier["loss_fn"]
        self.implementation = classifier["implementation"]
        self.tx = BaseTxConfig(classifier["tx"])


class QueryConfig:
    def __init__(self, query):
        self.load_config(query)

    def load_config(self, query):
        self.num_pos_samples = query["num_pos_samples"]
        self.num_neg_samples_per_pos = query["num_neg_samples_per_pos"]
        self.num_retries = query["num_retries"]


class EvaluationDataConfig:
    def __init__(self, data):
        self.load_config(data)

    def load_config(self, data):
        self.num_val = data["num_val"]
        self.num_test = data["num_test"]
        self.is_directed = data["is_directed"]
        self.add_negative_train_samples = data["add_negative_train_samples"]
        self.neg_sampling_ratio = data["neg_sampling_ratio"]


class EvaluationConfig:
    def __init__(self, evaluation):
        self.load_config(evaluation)

    def load_config(self, evaluation):
        self.eval_freq = evaluation["eval_freq"]
        self.first_epoch = evaluation["first_epoch"]
        self.last_epoch = evaluation["last_epoch"]
        self.mode = evaluation["mode"]
        self.link_feature_operator = evaluation["link_feature_operator"]
        self.classifier = ClassifierConfig(evaluation["classifier"])
        self.data = EvaluationDataConfig(evaluation["data"])
        self.query = QueryConfig(evaluation["query"])


class WandbConfig:
    def __init__(self, wandb):
        self.load_config(wandb)

    def load_config(self, wandb):
        self.project = wandb["project"]
        self.name = wandb["name"]
        self.group = wandb["group"]
        self.job_type = wandb["job_type"]
        self.mode = wandb["mode"]
