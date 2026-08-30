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

    def get_SFV(self):
        # The learnable SFV lives in the graph here; the SignNet and Invariant
        # smodels own none and return None. Checkpointing goes through this
        # protocol rather than reaching into smodel internals.
        return self.graph.x

    def set_SFV(self, w):
        if w is None or self.graph.x is None:
            return
        with torch.no_grad():
            self.graph.x.copy_(w.to(self.graph.x.device))

    def state_dict(self):
        weights = super().state_dict()
        weights["bn"] = self.bn.state_dict()
        # A frozen W never changes, so averaging it is a numerical no-op -- but
        # keep it out of the payload anyway, so "frozen" means one thing and the
        # arm cannot be confused with a trained-and-shared one.
        sm = config["structure_model"]
        frozen = bool(sm["freeze_sfv"]) if "freeze_sfv" in sm else False
        if config["federated"]["sfv_share"] == "avg" and not frozen:
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

    def set_SFV(self, w):
        # owns no SFV; its learnables are in state_dict already
        return

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
        self.n_scale = None   # GLOBAL node count; see set_scale
        self.adj = None       # CSR cumulative adjacency, built only for 'cn'
        self.features = features or config["spectral"]["es_features"]
        if self.features not in ("spec", "persist", "both", "cn"):
            raise ValueError(
                f"spectral.es_features must be spec|persist|both|cn, got {self.features!r}"
            )
        # attribution ablation: which of the three invariant blocks reach the MLP
        self.parts = tuple(p for p in config["spectral"]["es_spec_parts"].split("+") if p)
        unknown = set(self.parts) - {"phi", "cos", "lev"}
        if unknown or not self.parts:
            raise ValueError(
                f"spectral.es_spec_parts must be a '+'-joined non-empty subset of phi|cos|lev, "
                f"got {config['spectral']['es_spec_parts']!r}"
            )
        _trivial = self.features in ("persist", "cn")   # 1-feature baseline arms
        self.n_filters = n_filters if not _trivial and "phi" in self.parts else 0
        n_spec = 0 if _trivial else (
            self.n_filters + ("cos" in self.parts) + ("lev" in self.parts)
        )
        n_persist = 1 if self.features in ("persist", "both") else 0
        n_cn = 1 if self.features == "cn" else 0
        hidden = hidden or config["structure_model"]["DGCN_structure_layers_sizes"]
        # log-tau so tau stays positive; spread the initial scales over the
        # decades that matter for lambda in [0, 2]
        self.log_tau = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_filters, device=device)
        )
        dims = [n_spec + n_persist + n_cn] + list(hidden) + [1]
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

    def set_scale(self, n_global):
        """Global node count, served identically to server and clients.

        The leverage feature is scaled by a node count. Using the SERVED block's
        row count instead makes the same pair yield different features on the
        server (all N rows) and on a client (its own rows only), while the MLP
        that consumes them is FedAvg-averaged across both -- a train/serve
        inconsistency. The scale must therefore come from the server."""
        self.n_scale = float(n_global)

    def set_adj(self, edge_index, num_nodes):
        """Serve the cumulative graph itself, for the PERSISTENCE control: a
        1-bit 'this pair is already an edge' feature. §10.11 measured that
        trivial feature ABOVE the spectral affinity on every dataset (as733
        +0.50/+0.56 vs spectral's smaller margins), so any spectral claim has to
        be made against it, not only against the shuffled placebo — the placebo
        removes structure AND history together and cannot separate them."""
        self.n_nodes = num_nodes
        self.adj = None
        if edge_index is None or edge_index.numel() == 0:
            self.keys = torch.empty(0, dtype=torch.long, device=device)
            if self.features == "cn":
                # An EMPTY adjacency, not None. Leaving it None makes
                # _common_neighbours return None, edge_score return None, and the
                # whole arm fall back to the plain decoder with no error and no log
                # -- a cn run that is silently feature-only while _run_id still
                # stamps esf-cn. Reachable for any owner whose induced cumulative
                # union is empty (a client at small t under heavy sharding). persist
                # has no such hole: its empty case leaves an empty keys tensor and
                # still feeds a real 0 to the head, so this keeps the two baselines
                # behaving alike, which is the whole point of the comparison.
                from scipy import sparse as _sp
                self.adj = _sp.csr_matrix((num_nodes, num_nodes), dtype="float32")
            return
        a = torch.minimum(edge_index[0], edge_index[1]).to(torch.long)
        b = torch.maximum(edge_index[0], edge_index[1]).to(torch.long)
        self.keys = torch.unique(a * num_nodes + b).to(device)
        if self.features == "cn":
            # Symmetric unweighted CSR of the SAME cumulative graph persist reads,
            # so cn and persist differ only in what they compute from it. Built only
            # for this arm: it costs memory and no other arm reads it.
            from scipy import sparse as _sp
            import numpy as _np
            au = a.detach().cpu().numpy()
            bu = b.detach().cpu().numpy()
            r = _np.concatenate([au, bu])
            c = _np.concatenate([bu, au])
            m = _sp.csr_matrix(
                (_np.ones(r.size, dtype=_np.float32), (r, c)),
                shape=(num_nodes, num_nodes),
            )
            m.data[:] = 1.0        # collapse multi-edges: neighbour SET, not counts
            m.setdiag(0)           # a node is not its own common neighbour
            m.eliminate_zeros()
            self.adj = m

    def _common_neighbours(self, edge_label_index):
        """log1p(|N(u) & N(v)|) on the cumulative graph — the pre-registered
        structural baseline of results.md §10.11, which required every new-pair
        claim to be tested against `exists` AND `cn`. An offline probe put cn
        ABOVE the spectral affinity on both reddit graphs, so without this arm a
        spectral new-pair result cannot be attributed to the spectrum.

        log1p keeps the input O(1) like the leverage feature; it is monotone, so
        it cannot change any ranking the raw count would induce."""
        if self.adj is None or self.n_nodes is None:
            return None
        import numpy as _np
        u = edge_label_index[0].detach().cpu().numpy()
        v = edge_label_index[1].detach().cpu().numpy()
        out = _np.empty(u.size, dtype=_np.float32)
        # Chunked: adj[u] materialises a (P, N) sparse block whose nnz is sum(deg),
        # which is large for a full 1000-negatives-per-source eval batch.
        step = 65536
        for i in range(0, u.size, step):
            uu, vv = u[i:i + step], v[i:i + step]
            out[i:i + step] = _np.asarray(
                self.adj[uu].multiply(self.adj[vv]).sum(axis=1)
            ).ravel()
        t = torch.from_numpy(out).to(edge_label_index.device)
        return torch.log1p(t).unsqueeze(-1)

    def _persistence(self, edge_label_index):
        if self.keys is None or self.n_nodes is None:
            return None
        a = torch.minimum(edge_label_index[0], edge_label_index[1]).to(torch.long)
        b = torch.maximum(edge_label_index[0], edge_label_index[1]).to(torch.long)
        k = a * self.n_nodes + b
        return torch.isin(k, self.keys).to(torch.float32).unsqueeze(-1)

    def get_SFV(self):
        return None

    def set_SFV(self, w):
        # owns no SFV; the MLP and log_tau are already in state_dict
        return

    def get_D(self):
        return self.D

    def intrinsic_regularizer(self):
        return torch.zeros((), device=device)

    def edge_score(self, edge_label_index):
        """Scalar per candidate pair, added to the decoder logit."""
        if self.features == "persist":
            ex = self._persistence(edge_label_index)
            return None if ex is None else self.model(ex).squeeze(-1)
        if self.features == "cn":
            c = self._common_neighbours(edge_label_index)
            return None if c is None else self.model(c).squeeze(-1)
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
        feats = []
        if "phi" in self.parts:
            lam = self.D.to(prod.dtype).unsqueeze(0)        # (1, k)
            taus = torch.exp(self.log_tau).unsqueeze(1)     # (J, 1)
            gains = torch.exp(-taus * lam)                  # (J, k)
            feats.append(prod @ gains.t())                  # (P, J) heat-kernel affinities
        if "cos" in self.parts:
            feats.append(prod.sum(-1, keepdim=True))        # unfiltered affinity (the probe's)
        if "lev" in self.parts:
            # leverage: ||U_u|| is sqrt of a projector diagonal, so it is invariant too
            scale = self.n_scale if self.n_scale else float(self.Q.shape[0])
            feats.append(torch.log1p(nu * nv * scale))
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

    def set_scale(self, n_global):
        self.smodel.set_scale(n_global)

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

    def encode(self, data=_UNSET, hs=_UNSET, inject=None):
        # inject_at decides WHERE S enters. At 'output' (default) it is fused with the
        # final z, after every layer and every state update, so new_hs never sees it.
        # At 'last_mp' it is handed to the encoder and added to the last recurrent
        # layer's MP output before the state update -- and the output fusion is
        # SUPPRESSED, so S enters exactly once. Injecting at both points would put the
        # same signal through two different downstream paths and neither arm would be
        # interpretable.
        sp = config["spectral"]
        where = sp["inject_at"] if "inject_at" in sp else "output"
        if where == "last_mp":
            S = self.smodel.get_embeddings()
            z, new_hs = super().encode(data, hs, inject=S)
            self._record_inject_scale(S, new_hs)
            return z, new_hs
        z, new_hs = super().encode(data, hs, inject=inject)
        S = self.smodel.get_embeddings()
        self._record_inject_scale(S, new_hs)
        z = torch.cat([z, S], dim=-1) if self.fusion == "concat" else z + S
        return z, new_hs

    def _record_inject_scale(self, S, new_hs):
        """Relative magnitude of the structural term against the state it joins.

        This is a GATE on interpreting any injection result, not a diagnostic.
        `spectral.output_bn` zero-inits the smodel's BatchNorm gamma and its bias, so
        S is EXACTLY zero at initialisation -- deliberately, so training starts at the
        feature-only baseline. If it never leaves zero, a `last_mp` null means the term
        was never injected rather than that the injection point does not matter, and
        the two are indistinguishable without this number. The opposite failure is on
        record too: W7 measured the spectral branch growing to ~10x the recurrent
        representation when unconstrained.
        """
        try:
            if S is None or not new_hs:
                self.inject_scale = float("nan"); return
            h = new_hs[-1]
            hn = float(h.norm())
            self.inject_scale = float(S.norm()) / hn if hn > 0 else float("nan")
        except Exception:
            self.inject_scale = float("nan")

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
