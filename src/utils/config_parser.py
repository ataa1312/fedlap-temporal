import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)

        # Top-level attributes
        self.seed = self.config.get("seed", 1234)
        self.experiment = self.config.get("experiment", 0)
        self.num_runs = self.config.get("num_runs", 10)

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
        self.is_directed = dataset.get("is_directed", False)
        self.shape = dataset.get("shape", None)
        self.num_classes = dataset.get("num_classes", None)
        self.normalize = dataset.get("normalize", False)


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


class ModelStructuralConfig:
    def __init__(self, structural):
        self.load_config(structural)

    def load_config(self, structural):
        self.input_dimension = structural.get("input_dimension", None)
        self.output_dimension = structural.get("output_dimension", 128)
        self.layer_dimensions = structural.get("layer_dimensions", [])
        self.attention_heads = structural.get("attention_heads", [16])
        self.dropout = structural.get("dropout", 0.1)
        self.residual = structural.get("residual", False)


class ModelTemporalConfig:
    def __init__(self, temporal):
        self.load_config(temporal)

    def load_config(self, temporal):
        self.input_dimension = temporal.get("input_dimension", 128)
        self.num_attention_layers = temporal.get("num_attention_layers", 1)
        self.attention_heads = temporal.get("attention_heads", 16)
        self.dropout = temporal.get("dropout", 0.5)
        self.num_snapshots = temporal.get("num_snapshots", None)
        self.use_feedforward = temporal.get("use_feedforward", True)


class TrainModelConfig:
    def __init__(self, model):
        self.load_config(model)

    def load_config(self, model):
        self.structural = ModelStructuralConfig(model["structural"])
        self.temporal = ModelTemporalConfig(model["temporal"])


class RandomWalkConfig:
    def __init__(self, random_walk):
        self.load_config(random_walk)

    def load_config(self, random_walk):
        self.num = random_walk.get("num", 10)
        self.length = random_walk.get("length", 40)
        self.window_size = random_walk.get("window_size", 10)


class NegativeSamplingConfig:
    def __init__(self, neg_sampling):
        self.load_config(neg_sampling)

    def load_config(self, neg_sampling):
        self.size = neg_sampling.get("size", 10)
        self.weight = neg_sampling.get("weight", 1.0)
        self.retries = neg_sampling.get("retries", 5)
        self.distortion = neg_sampling.get("distortion", 0.75)
        self.tempoal_window = neg_sampling.get("tempoal_window", -1)


class UnsupervisedTrainingConfig:
    def __init__(self, unsupervised):
        self.load_config(unsupervised)

    def load_config(self, unsupervised):
        self.neg_sampling = NegativeSamplingConfig(unsupervised["neg_sampling"])
        self.batch_size = unsupervised.get("batch_size", 512)
        self.weight_decay = unsupervised.get("weight_decay", 5e-4)


class DynamicConfig:
    def __init__(self, train):
        self.load_config(train)

    def load_config(self, train):
        self.model = TrainModelConfig(train["model"])
        self.loss_fn = train.get("loss_fn", "BCEWithLogitsLoss")
        self.min_snapshot = train.get("min_snapshot", 2)
        self.max_snapshot = train.get("max_snapshot", 12)
        self.random_walk = RandomWalkConfig(train["random_walk"])
        self.unsupervised = UnsupervisedTrainingConfig(train["unsupervised"])
        self.max_gradient_norm = train.get("max_gradient_norm", 1.0)
        self.evaluation = EvaluationConfig(train["evaluation"])


class BaseTxConfig:
    def __init__(self, tx):
        self.load_config(tx)

    def load_config(self, tx):
        self.fn = tx.get("fn", None)
        self.lr = tx.get("lr", 1.0)
        self.moment = tx.get("moment", 0.0)
        self.weight_decay = tx.get("weight_decay", 0.0)
        self.lr_decay_per_worker_epoch = tx.get("lr_decay_per_worker_epoch", 1.0)
        self.type_decay_per_worker_epoch = tx.get(
            "type_decay_per_worker_epoch", "geometric"
        )


class ClassifierConfig:
    def __init__(self, classifier):
        self.load_config(classifier)

    def load_config(self, classifier):
        self.num_epoch = classifier.get("num_epoch", 100)
        self.loss_fn = classifier.get("loss_fn", "BCEWithLogitsLoss")
        self.tx = BaseTxConfig(classifier["tx"])


class EvaluationDataConfig:
    def __init__(self, data):
        self.load_config(data)

    def load_config(self, data):
        self.num_val = data.get("num_val", 0.2)
        self.num_test = data.get("num_test", 0.6)
        self.is_directed = data.get("is_directed", False)
        self.add_negative_train_samples = data.get("add_negative_train_samples", True)
        self.neg_sampling_ratio = data.get("neg_sampling_ratio", 1.0)


class EvaluationConfig:
    def __init__(self, evaluation):
        self.load_config(evaluation)

    def load_config(self, evaluation):
        self.eval_freq = evaluation.get("eval_freq", 1)
        self.first_epoch = evaluation.get("first_epoch", True)
        self.last_epoch = evaluation.get("last_epoch", True)
        self.mode = evaluation.get("mode", "roland")
        self.link_feature_operator = evaluation.get("link_feature_operator", "hadamard")
        self.classifier = ClassifierConfig(evaluation["classifier"])
        self.data = EvaluationDataConfig(evaluation["data"])


class WandbConfig:
    def __init__(self, wandb):
        self.load_config(wandb)

    def load_config(self, wandb):
        self.project = wandb.get("project", "dynamic-centralized")
        self.name = wandb.get("name", None)
        self.group = wandb.get("group", None)
        self.job_type = wandb.get("job_type", "dynamic-edge-prediction")
        self.mode = wandb.get("mode", "online")
