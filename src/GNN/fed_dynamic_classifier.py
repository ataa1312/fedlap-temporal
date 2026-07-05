import torch
import torch.nn as nn

from src import *
from src.utils.graph import Graph
from src.GNN.sGNN import SClassifier
from src.GNN.laplace import SpectralLaplaceMixin
from src.models.model_binders import ModelSpecs, ModelBinder
from src.GNN.dynamic_classifier import DynamicClassifier, _UNSET


class DynamicSLaplace(SpectralLaplaceMixin, SClassifier):
    """Spectral structure model for the dynamic path: S = MLP(relu(Q @ W)) with
    W = sgraph.x learnable (SFVMixin) and Q, D per-client buffers via set_QD.
    Same construction as SClassifier except the MLP output is sized for fusion
    with the recurrent z instead of num_classes. federated.sfv_share='avg' adds
    W to state_dict so it joins the weight averaging (FedLap's joint_train_w
    syncs it through the just_SFV gradient path instead)."""

    def __init__(self, graph: Graph, out_dim, hidden_layer_size=None):
        self.out_dim = out_dim
        if hidden_layer_size is None:
            hidden_layer_size = config["structure_model"]["DGCN_structure_layers_sizes"]
        super().__init__(graph, hidden_layer_size)

    def create_smodel(self, hidden_layer_size=[]):
        layer_sizes = [self.graph.num_features] + hidden_layer_size + [self.out_dim]

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

        # Output BatchNorm bounds the spectral amplification (S otherwise grows to
        # dominate the l2-normed z). gamma is zero-initialized so S starts at 0 ->
        # f+s begins exactly at the feature-only baseline and grows only as training
        # earns it. track_running_stats=False (FedLap MLP idiom): batch stats, no
        # cross-snapshot running-stat drift, and no int buffers to corrupt in FedAvg.
        # Owned here (not folded into the MLP, which drops norms from its state_dict)
        # so it joins parameters / state_dict / FedAvg / early-stop restore-best.
        if config["spectral"]["output_bn"]:
            self.bn = nn.BatchNorm1d(
                self.out_dim, affine=True, track_running_stats=False
            ).to(device)
            nn.init.zeros_(self.bn.weight)
        else:
            self.bn = nn.Identity()

    def get_embeddings(self):
        return self.bn(super().get_embeddings())

    def state_dict(self):
        weights = super().state_dict()
        weights["bn"] = self.bn.state_dict()
        if config["federated"]["sfv_share"] == "avg":
            weights["SFV"] = self.graph.x.detach().clone()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        if "bn" in weights:
            self.bn.load_state_dict(weights["bn"])
        if "SFV" in weights:
            with torch.no_grad():
                self.graph.x.copy_(weights["SFV"])

    def parameters(self):
        return super().parameters() + list(self.bn.parameters())

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if not just_SFV:
            grads["bn"] = [p.grad for p in self.bn.parameters()]
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if "bn" in grads:
            for p, g in zip(self.bn.parameters(), grads["bn"]):
                p.grad = g

    def train(self, mode: bool = True):
        super().train(mode)
        self.bn.train(mode)

    def eval(self):
        super().eval()
        self.bn.eval()

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        self.bn.zero_grad(set_to_none=set_to_none)


class FedDynamicClassifier(DynamicClassifier):
    """DynamicClassifier (the fmodel role) + a spectral smodel, FedLap's f/s
    composition on the dynamic path. encode fuses S = smodel.get_embeddings()
    with the recurrent output z per model.fusion: 'add' (smodel MLP output
    matches z width) or 'concat' (head widened to 2d). Q rows follow the owning
    subgraph's node order, so S is row-aligned with z."""

    def __init__(self, fgraph, sgraph: Graph):
        self.smodel = None
        self.fusion = config["model"]["fusion"]
        super().__init__(fgraph)
        self.create_smodel(sgraph)

    def create_smodel(self, sgraph: Graph):
        raise NotImplementedError

    def head_dim_in(self, d):
        return 2 * d if self.fusion == "concat" else d

    def encode(self, data=_UNSET, hs=_UNSET):
        z, new_hs = super().encode(data, hs)
        S = self.smodel.get_embeddings()
        z = torch.cat([z, S], dim=-1) if self.fusion == "concat" else z + S
        return z, new_hs

    # ---- spectral delegation (FedMixin surface) ---- #

    def set_QD(self, U, D):
        self.smodel.set_QD(U, D)

    def get_SFV(self):
        return self.smodel.get_SFV()

    def get_D(self):
        return self.smodel.get_D()

    def intrinsic_regularizer(self):
        return self.smodel.intrinsic_regularizer()

    # ---- federated protocol: base covers model+head; extend for the smodel ---- #

    def state_dict(self):
        weights = super().state_dict()
        if self.smodel is not None:
            weights["smodel"] = self.smodel.state_dict()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        if self.smodel is not None and "smodel" in weights:
            self.smodel.load_state_dict(weights["smodel"])

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if self.smodel is not None:
            grads["smodel"] = self.smodel.get_grads(just_SFV)
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if self.smodel is not None and "smodel" in grads:
            self.smodel.set_grads(grads["smodel"])

    def parameters(self):
        params = super().parameters()
        if self.smodel is not None:
            params += self.smodel.parameters()
        return params

    def train(self, mode: bool = True):
        super().train(mode)
        if self.smodel is not None:
            self.smodel.train(mode)

    def eval(self):
        super().eval()
        if self.smodel is not None:
            self.smodel.eval()

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        if self.smodel is not None:
            self.smodel.zero_grad(set_to_none=set_to_none)


class FedDynamicSpectralLaplaceClassifier(FedDynamicClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = DynamicSLaplace(sgraph, out_dim=config["gnn"]["dims"][-1])


class FedDynamicLanczosLaplaceClassifier(FedDynamicClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = DynamicSLaplace(sgraph, out_dim=config["gnn"]["dims"][-1])


FED_DYNAMIC_CLASSIFIERS = {
    "SpectralLaplace": FedDynamicSpectralLaplaceClassifier,
    "LanczosLaplace": FedDynamicLanczosLaplaceClassifier,
}


def make_sgraph(SFV) -> Graph:
    # fresh leaf per owner (FedLap's create_SGNN_data re-wrap): every client
    # starts from the same W init but owns an independent parameter
    x = torch.tensor(
        SFV.detach().cpu().numpy(),
        requires_grad=SFV.requires_grad,
        device=device,
    )
    return Graph(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        node_ids=torch.arange(x.shape[0]),
    )


def make_fed_dynamic_classifier(smodel_type, fgraph, SFV):
    if smodel_type not in FED_DYNAMIC_CLASSIFIERS:
        raise NotImplementedError(
            f"smodel_type={smodel_type!r} has no dynamic classifier; "
            f"known: {sorted(FED_DYNAMIC_CLASSIFIERS)}"
        )
    return FED_DYNAMIC_CLASSIFIERS[smodel_type](fgraph, make_sgraph(SFV))
