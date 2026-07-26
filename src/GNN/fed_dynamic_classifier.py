import torch
import torch.nn as nn

from src import *
from src.classifier import Classifier
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


class DynamicSSignNet(Classifier):
    """Sign-invariant spectral structure model (SignNet, Lim et al. 2022) for the
    dynamic path. S = rho( sum_i [ phi(u_i) + phi(-u_i) ] ): a shared phi maps each
    eigenvector entry and its sign-flip, the two are summed (invariant to the
    per-eigenvector sign gauge), summed again over the spectral_len columns
    (DeepSets over eigenvectors), then rho maps the per-node aggregate to the
    fusion width. This REPLACES DynamicSLaplace's sign-fix + procrustes gauge
    handling at the source. There is no Q@W filter, so there is no SFV to federate
    (federated.sfv_share is moot); phi/rho join FedAvg through state_dict like the
    fmodel. Q, D are per-client buffers set via set_QD each snapshot; only Q is
    consumed. out_dim sizes rho for fusion (add/concat) with the recurrent z."""

    def __init__(self, graph: Graph, out_dim):
        super().__init__(graph)
        self.out_dim = out_dim
        self.Q = None
        self.D = None
        if graph.x is not None:
            graph.x.requires_grad_(False)  # SFV leaf is unused by SignNet
        self.create_smodel()

    def create_smodel(self):
        phi_dims = config["spectral"]["signnet_phi_dims"]
        rho_dims = config["spectral"]["signnet_rho_dims"]
        # phi: shared per-entry map R^1 -> R^phi_out. NO input normalization -- the
        # MLP normalizes its INPUT layer, and Layer/BatchNorm over a width-1 input
        # would erase the eigenvector value.
        self.phi = ModelBinder(
            [
                ModelSpecs(
                    type="MLP",
                    layer_sizes=[1] + list(phi_dims),
                    final_activation_function="linear",
                    normalization=None,
                )
            ]
        )
        # rho: aggregated node encoding R^phi_out -> R^out_dim (fusion width).
        self.rho = ModelBinder(
            [
                ModelSpecs(
                    type="MLP",
                    layer_sizes=[phi_dims[-1]] + list(rho_dims) + [self.out_dim],
                    final_activation_function="linear",
                    normalization="layer",
                )
            ]
        )
        self.phi.to(device)
        self.rho.to(device)
        # Zero-init output BatchNorm (same role as DynamicSLaplace): bounds the
        # spectral amplification and starts S at 0, so f+s begins exactly at the
        # feature-only baseline and grows only as training earns it. Owned here so
        # it joins parameters / state_dict / FedAvg / early-stop restore-best.
        if config["spectral"]["output_bn"]:
            self.bn = nn.BatchNorm1d(
                self.out_dim, affine=True, track_running_stats=False
            ).to(device)
            nn.init.zeros_(self.bn.weight)
        else:
            self.bn = nn.Identity()

    def set_QD(self, Q, D):
        self.Q = Q
        self.D = D

    def get_D(self):
        return self.D

    def get_SFV(self):
        return None

    def get_embeddings(self):
        Q = self.Q
        N, k = Q.shape
        x = Q.reshape(N * k, 1)
        h = self.phi(x) + self.phi(-x)  # sign-invariant per (node, eigenvector)
        h = h.reshape(N, k, -1).sum(dim=1)  # DeepSets sum over the k eigenvectors
        S = self.rho(h)
        return self.bn(S)

    # ---- FedLap Classifier protocol (phi + rho + output bn) ---- #

    def parameters(self):
        return (
            list(self.phi.parameters())
            + list(self.rho.parameters())
            + list(self.bn.parameters())
        )

    def state_dict(self):
        return {
            "phi": self.phi.state_dict(),
            "rho": self.rho.state_dict(),
            "bn": self.bn.state_dict(),
        }

    def load_state_dict(self, weights):
        self.phi.load_state_dict(weights["phi"])
        self.rho.load_state_dict(weights["rho"])
        if "bn" in weights:
            self.bn.load_state_dict(weights["bn"])

    def get_grads(self, just_SFV=False):
        if just_SFV:
            return {}
        return {
            "phi": self.phi.get_grads(),
            "rho": self.rho.get_grads(),
            "bn": [p.grad for p in self.bn.parameters()],
        }

    def set_grads(self, grads):
        if "phi" in grads:
            self.phi.set_grads(grads["phi"])
        if "rho" in grads:
            self.rho.set_grads(grads["rho"])
        if "bn" in grads:
            for p, g in zip(self.bn.parameters(), grads["bn"]):
                p.grad = g

    def train(self, mode: bool = True):
        self.phi.train(mode)
        self.rho.train(mode)
        self.bn.train(mode)

    def eval(self):
        self.phi.eval()
        self.rho.eval()
        self.bn.eval()

    def zero_grad(self, set_to_none=False):
        self.phi.zero_grad(set_to_none=set_to_none)
        self.rho.zero_grad(set_to_none=set_to_none)
        self.bn.zero_grad(set_to_none=set_to_none)


class DynamicSInvariant:
    """Spectral structure model whose output is an EDGE score built from
    ROTATION-INVARIANT features of the eigenbasis.

    Why invariants (results.md §10.12/§10.12b): the low spectrum of these graphs
    is clustered (~1e-3 gaps), so individual eigenvectors are not identifiable —
    between snapshots they rotate 50-80 degrees while the SUBSPACE stays fixed
    (overlap 0.96-1.00). A per-coordinate MLP over rows of U therefore reads a
    quantity the data does not determine, which is what the Laplace smodel did.
    Every readout that worked was instead a function of the projector:

        phi_j(u, v) = sum_i f_j(lambda_i) U_ui U_vi  =  [U f_j(Lambda) U^T]_uv

    which is invariant to U -> UR for any rotation R within an eigenspace, and
    only O(|f_j(lambda_i) - f_j(lambda_j)|) sensitive to mixing between two
    nearly-equal eigenvalues — so a SMOOTH f is stable exactly where the basis
    is ambiguous. Filters are heat kernels exp(-tau_j lambda) with tau learnable
    (plus the unfiltered projector entry and its cosine form), so the fixed
    affinity used in the probes is the tau -> 0 special case and training can
    only sharpen it. The MLP over those few invariants is the learnable part,
    federated through state_dict like any FedLap smodel."""

    def __init__(self, n_filters=4, hidden=None, features=None):
        self.Q = None
        self.D = None
        self.keys = None      # sorted pair keys of the cumulative graph (persistence)
        self.n_nodes = None
        self.features = features or config["spectral"]["es_features"]
        if self.features not in ("spec", "persist", "both"):
            raise ValueError(f"spectral.es_features must be spec|persist|both, got {self.features!r}")
        self.n_filters = n_filters if self.features != "persist" else 0
        n_spec = (self.n_filters + 2) if self.features != "persist" else 0
        n_persist = 1 if self.features in ("persist", "both") else 0
        hidden = hidden or config["structure_model"]["DGCN_structure_layers_sizes"]
        # log-tau so tau stays positive; spread the initial scales over the
        # decades that matter for lambda in [0, 2]
        self.log_tau = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_filters, device=device)
        )
        dims = [n_spec + n_persist] + list(hidden) + [1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.model = nn.Sequential(*layers[:-1]).to(device)
        # zero-init the last layer: the edge-score term starts at exactly 0, so
        # training begins at the feature-only baseline and earns any deviation
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)

    def set_QD(self, Q, D):
        self.Q = Q
        self.D = D

    def set_adj(self, edge_index, num_nodes):
        """Serve the cumulative graph itself, for the PERSISTENCE control: a
        1-bit 'this pair is already an edge' feature. §10.11 measured that
        trivial feature ABOVE the spectral affinity on every dataset (as733
        +0.50/+0.56 vs spectral's smaller margins), so any spectral claim has to
        be made against it, not only against the shuffled placebo — the placebo
        removes structure AND history together and cannot separate them."""
        self.n_nodes = num_nodes
        if edge_index is None or edge_index.numel() == 0:
            self.keys = torch.empty(0, dtype=torch.long, device=device)
            return
        a = torch.minimum(edge_index[0], edge_index[1]).to(torch.long)
        b = torch.maximum(edge_index[0], edge_index[1]).to(torch.long)
        self.keys = torch.unique(a * num_nodes + b).to(device)

    def _persistence(self, edge_label_index):
        if self.keys is None or self.n_nodes is None:
            return None
        a = torch.minimum(edge_label_index[0], edge_label_index[1]).to(torch.long)
        b = torch.maximum(edge_label_index[0], edge_label_index[1]).to(torch.long)
        k = a * self.n_nodes + b
        return torch.isin(k, self.keys).to(torch.float32).unsqueeze(-1)

    def get_SFV(self):
        return None

    def get_D(self):
        return self.D

    def intrinsic_regularizer(self):
        return torch.zeros((), device=device)

    def edge_score(self, edge_label_index):
        """Scalar per candidate pair, added to the decoder logit."""
        if self.features == "persist":
            ex = self._persistence(edge_label_index)
            return None if ex is None else self.model(ex).squeeze(-1)
        if self.Q is None or self.D is None:
            return None
        u = self.Q[edge_label_index[0]]
        v = self.Q[edge_label_index[1]]
        # Row-normalise first: eigenvector entries are O(1/sqrt(N)), so raw
        # projector entries are O(1/N) (~1e-4 here) and would reach the MLP as
        # dead inputs. Normalising makes every filter an O(1) cosine-style
        # affinity and makes the unfiltered feature exactly the quantity the
        # probes measured (§10.11), so training starts from a known-good signal.
        nu = u.norm(dim=-1, keepdim=True)
        nv = v.norm(dim=-1, keepdim=True)
        un, vn = u / (nu + 1e-12), v / (nv + 1e-12)
        prod = un * vn                                      # (P, k)
        lam = self.D.to(prod.dtype).unsqueeze(0)            # (1, k)
        taus = torch.exp(self.log_tau).unsqueeze(1)         # (J, 1)
        gains = torch.exp(-taus * lam)                      # (J, k)
        phi = prod @ gains.t()                              # (P, J) heat-kernel affinities
        cos = prod.sum(-1, keepdim=True)                    # unfiltered affinity (the probe's)
        # leverage: ||U_u|| is sqrt of a projector diagonal, so it is invariant too
        lev = torch.log1p(nu * nv * float(self.Q.shape[0]))
        feats = [phi, cos, lev]
        if self.features == "both":
            ex = self._persistence(edge_label_index)
            if ex is None:
                return None
            feats.append(ex)
        return self.model(torch.cat(feats, dim=-1)).squeeze(-1)

    # ---- FedLap smodel protocol ---- #

    def state_dict(self):
        return {"model": self.model.state_dict(), "log_tau": self.log_tau.detach().clone()}

    def load_state_dict(self, weights):
        if "model" in weights:
            self.model.load_state_dict(weights["model"])
        if "log_tau" in weights:
            with torch.no_grad():
                self.log_tau.copy_(weights["log_tau"])

    def parameters(self):
        return list(self.model.parameters()) + [self.log_tau]

    def get_grads(self, just_SFV=False):
        return {"model": [p.grad for p in self.parameters()]}

    def set_grads(self, grads):
        if "model" in grads:
            for p, g in zip(self.parameters(), grads["model"]):
                p.grad = g

    def train(self, mode: bool = True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def zero_grad(self, set_to_none=False):
        self.model.zero_grad(set_to_none=set_to_none)
        if self.log_tau.grad is not None:
            self.log_tau.grad = None if set_to_none else torch.zeros_like(self.log_tau)


class FedDynamicEdgeScoreClassifier(DynamicClassifier):
    """fmodel + an smodel that contributes at the DECISION level
    (model.data_type=f+es): the spectral term is added to the decoder logit
    rather than to the node embedding.

    §10.11 measured why: embedding-level injection is absorbed by training,
    while decode-time fusion of a spectral affinity moves the reported MRR
    (+0.37..+0.75 on as733's new pairs, +0.02..+0.03 on reddit_body, both
    growing with sharding, placebo null). This is that fusion made learnable and
    federated, with the invariant features of DynamicSInvariant as its input."""

    def __init__(self, fgraph):
        self.smodel = None
        super().__init__(fgraph)
        self.smodel = DynamicSInvariant()

    def _edge_term(self, g, pred):
        if self.smodel is None:
            return pred
        s = self.smodel.edge_score(g.edge_label_index)
        return pred if s is None else pred + s

    def decode(self, z=None, data=_UNSET):
        g = self.graph if data is _UNSET else data
        pred, label = super().decode(z, data)
        return self._edge_term(g, pred), label

    def forward(self, data=_UNSET, hs=_UNSET):
        g = self.graph if data is _UNSET else data
        pred, label, new_hs = super().forward(data, hs)
        return self._edge_term(g, pred), label, new_hs

    # ---- spectral delegation + federated protocol (FedMixin surface) ---- #

    def set_QD(self, U, D):
        self.smodel.set_QD(U, D)

    def set_adj(self, edge_index, num_nodes):
        self.smodel.set_adj(edge_index, num_nodes)

    def get_SFV(self):
        return self.smodel.get_SFV()

    def get_D(self):
        return self.smodel.get_D()

    def intrinsic_regularizer(self):
        return self.smodel.intrinsic_regularizer()

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


class FedDynamicPEClassifier(DynamicClassifier):
    """Input-side spectral positional encoding (model.data_type=f+pe): the
    server-computed exact low-k eigenbasis slice is CONCATENATED to the node
    features before the encoder, LapPE-style, so message passing can use the
    coordinates relationally — unlike the smodel path, which fuses at the
    output where S can only shift each node's final embedding. There is no
    smodel and nothing spectral is learned or federated; the federated
    protocol is the plain fmodel one. PE rows follow the owning subgraph's
    node order (set_QD each snapshot, same serving contract as Q)."""

    def __init__(self, fgraph):
        self.pe_dim = config["spectral"]["pe_dim"]
        self.PE = None
        self.D = None
        super().__init__(fgraph)

    def input_dim(self):
        return self.graph.num_features + self.pe_dim

    def node_input(self, g):
        pe = self.PE
        assert pe is not None, "f+pe encode before set_QD: no PE served yet"
        assert pe.shape[0] == g.x.shape[0], (
            f"PE rows {pe.shape[0]} != nodes {g.x.shape[0]} (slice/order mismatch)"
        )
        return torch.cat([g.x, pe], dim=-1)

    # ---- spectral serving surface (FedMixin protocol, nothing learned) ---- #

    def set_QD(self, Q, D):
        self.PE = Q[:, : self.pe_dim].to(device)
        self.D = D

    def get_SFV(self):
        return None

    def get_D(self):
        return self.D

    def intrinsic_regularizer(self):
        return torch.zeros((), device=device)


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


class FedDynamicSignNetClassifier(FedDynamicClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = DynamicSSignNet(sgraph, out_dim=config["gnn"]["dims"][-1])


FED_DYNAMIC_CLASSIFIERS = {
    "SpectralLaplace": FedDynamicSpectralLaplaceClassifier,
    "LanczosLaplace": FedDynamicLanczosLaplaceClassifier,
    "SignNet": FedDynamicSignNetClassifier,
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
