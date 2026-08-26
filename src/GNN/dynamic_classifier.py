import torch
import torch.nn.functional as F

from src import *
from src.classifier import Classifier
from src.models.model_binders import ModelSpecs, ModelBinder
from src.models.weight_init import init_weights
from src.train.federated_orchestrator import _mp_graph
from registries import heads

# D1 sentinel: distinguishes an omitted arg (use the stateful self.graph/self.hs)
# from an explicit None (e.g. hs=None = no prior state on the first snapshot).
_UNSET = object()


class DynamicClassifier(Classifier):
    """Per-client ROLAND recurrent link-prediction model in the FedLap idiom.

    Encoder = a ModelBinder assembled from ModelSpecs (the assembly that was
    RecurrentGNN.__init__); decoder = a task-keyed head (heads[task]). Pure
    feature model: spectral quantities live in subclasses that add an smodel
    (FedLap's fmodel/smodel composition). The federated protocol covers
    encoder + head.
    """

    def __init__(self, graph=None):
        super().__init__(graph)
        self.head = None
        self.hs = None        # prior per-layer hidden state (threaded by the orchestrator)
        self.last_hs = None   # most recent new_hs (for the orchestrator to carry forward)
        self.l2norm = False
        self.create_model()

    def create_model(self):
        gnn, mcfg, ds = config["gnn"], config["model"], config["dataset"]
        task = ds["task"]
        specs = []

        # edge encoder transforms edge_attr; place first so the recurrent layers
        # see the encoded edge features.
        raw_edge_dim = ds["edge_dim"]
        if ds["edge_encoder"] and raw_edge_dim > 0:
            specs.append(ModelSpecs(
                type="edge_encoder",
                layer_sizes=[raw_edge_dim, ds["edge_encoder_dim"]],
                block_kwargs={"batchnorm": ds["edge_encoder_bn"]},
            ))
            effective_edge_dim = ds["edge_encoder_dim"]
        else:
            effective_edge_dim = raw_edge_dim

        d = self.input_dim()
        if ds["node_encoder"]:
            specs.append(ModelSpecs(
                type="node_encoder",
                layer_sizes=[d, ds["node_encoder_dim"]],
                block_kwargs={"batchnorm": ds["node_encoder_bn"]},
            ))
            d = ds["node_encoder_dim"]

        dims_pre_mp = gnn["dims_pre_mp"]
        if dims_pre_mp:
            specs.append(ModelSpecs(
                type="roland_mlp",
                layer_sizes=[d] + dims_pre_mp,
                block_kwargs={"batchnorm": gnn["batchnorm"], "dropout": gnn["dropout"],
                              "act": gnn["act"], "final_act": True},
            ))
            d = dims_pre_mp[-1]

        layer_type = gnn["layer_type"]
        layer_kwargs = {}
        if layer_type == "residual_edge_conv":
            layer_kwargs = {"edge_dim": effective_edge_dim, "msg_direction": gnn["msg_direction"],
                            "normalize": gnn["normalize_adj"], "agg": gnn["agg"]}
        updater_kwargs = {}
        if gnn["embed_update_method"] == "moving_average":
            updater_kwargs["alpha"] = config["meta"]["alpha"]
        rec = {"layer_type": layer_type, "updater_name": gnn["embed_update_method"],
               "batchnorm": gnn["batchnorm"], "dropout": gnn["dropout"], "act": gnn["act"],
               "skip_connection": gnn["skip_connection"], "layer_kwargs": layer_kwargs,
               "updater_kwargs": updater_kwargs}
        for w in gnn["dims"]:
            specs.append(ModelSpecs(
                type="recurrent_layer",
                layer_sizes=[d, w],
                block_kwargs=dict(rec),
            ))
            d = w

        self.model = ModelBinder(specs).to(device)
        self.l2norm = gnn["l2norm"]

        head_cls = heads[task]
        decoding = mcfg["edge_decoding"]
        is_edge = task in ("edge", "link_pred")
        head_kwargs = {"dims_inner": gnn["dims_post_mp"],
                       "batchnorm": gnn["batchnorm"], "act": gnn["act"]}
        if is_edge:
            head_kwargs["decoding"] = decoding
        dim_out = ds["num_classes"] if task == "node" and ds["num_classes"] else 1
        self.head = head_cls(self.head_dim_in(d), dim_out, **head_kwargs).to(device)
        self.model.apply(init_weights)
        self.head.apply(init_weights)

    # ---- encode / decode ---- #

    def input_dim(self):
        return self.graph.num_features

    def node_input(self, g):
        return g.x

    def encode(self, data=_UNSET, hs=_UNSET):
        g = self.graph if data is _UNSET else data
        h = self.hs if hs is _UNSET else hs
        active = getattr(g, "node_degree_new", None)
        if active is not None:
            active = active > 0
        # The ONLY place the encoder's message-passing edge set is chosen.
        # gnn.encoder_edge_drop attaches a fixed subset here and nowhere else,
        # so labels/negatives/splits/spectral basis keep the full edge set.
        mp_edge_index, mp_edge_attr = _mp_graph(g)
        z, new_hs = self.model.encode(
            self.node_input(g), mp_edge_index, h,
            edge_attr=mp_edge_attr,
            keep_ratio=getattr(g, "keep_ratio", None),
            active_mask=active,
        )
        if self.l2norm:
            z = F.normalize(z, p=2, dim=-1)
        self.last_hs = new_hs
        return z, new_hs

    def get_embeddings(self):
        z, _ = self.encode()
        return z

    def decode(self, z=None, data=_UNSET):
        g = self.graph if data is _UNSET else data
        if z is None:
            z = self.get_embeddings()
        return self.head(z, g)

    def forward(self, data=_UNSET, hs=_UNSET):
        z, new_hs = self.encode(data, hs)
        g = self.graph if data is _UNSET else data
        pred, label = self.head(z, g)
        return pred, label, new_hs

    def __call__(self, data=_UNSET, hs=_UNSET):
        # Route calls to forward. The base Classifier.__call__ returns embeddings
        # and takes no args; the ported ROLAND helpers call model(snap, hs), which
        # for a recurrent model must hit forward -> (pred, label, new_hs).
        return self.forward(data, hs)

    def head_dim_in(self, d):
        # encoder-output width the head consumes; spectral subclasses widen it
        return d

    def refresh_hs(self):
        # advance the carried state at the snapshot boundary (TBPTT-1, detached)
        self.hs = None if self.last_hs is None else [h.detach() for h in self.last_hs]

    def train_step(self, eval_=True):
        # the parent's mask-based node step does not apply to the live-update task
        raise NotImplementedError(
            "DynamicClassifier trains through the live-update loop "
            "(DynamicServer.joint_train_w), not train_step"
        )

    # ---- federated protocol: base covers self.model; extend for the head ---- #

    def state_dict(self):
        weights = super().state_dict()
        if self.head is not None:
            weights["head"] = self.head.state_dict()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        if self.head is not None and "head" in weights:
            self.head.load_state_dict(weights["head"])

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if not just_SFV and self.head is not None:
            grads["head"] = [p.grad for p in self.head.parameters()]
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if "head" in grads.keys() and self.head is not None:
            for p, g in zip(self.head.parameters(), grads["head"]):
                p.grad = g

    def parameters(self):
        params = super().parameters()
        if self.head is not None:
            params += list(self.head.parameters())
        return params

    def train(self, mode: bool = True):
        super().train(mode)
        if self.head is not None:
            self.head.train(mode)

    def eval(self):
        super().eval()
        if self.head is not None:
            self.head.eval()

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        if self.head is not None:
            self.head.zero_grad(set_to_none=set_to_none)
