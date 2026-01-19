from src import *
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster, SpectralDGCN
from src.GNN.fGNN import FGNN, NewFGNN, FEdgeGNN
from src.GNN.sGNN import SGNNSlave, SGNNMaster, SClassifier, SEdgeClassifier
from src.classifier import Classifier, EdgeClassifier
from src.utils.data import Data
from src.GNN.laplace import (
    SLaplace,
    SEdgeLaplace,
    LanczosLaplace,
    SpectralLaplace,
    LanczosLaplaceNew,
    LanczosEdgeLaplace,
    SpectralEdgeLaplace,
)
from src.utils.graph import Graph, AGraph
from src.GNN.temporal import TemporalBlock
from src.utils.define_graph import get_degrees
from src.utils.config_parser import TrainModelConfig


class FedMixin:
    def state_dict(self):
        weights = super().state_dict()
        weights["fmodel"] = self.fmodel.state_dict()
        weights["smodel"] = self.smodel.state_dict()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        self.fmodel.load_state_dict(weights["fmodel"])
        self.smodel.load_state_dict(weights["smodel"])

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        grads["fmodel"] = self.fmodel.get_grads(just_SFV)
        grads["smodel"] = self.smodel.get_grads(just_SFV)
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        self.fmodel.set_grads(grads["fmodel"])
        self.smodel.set_grads(grads["smodel"])

    def reset_parameters(self):
        super().reset_parameters()
        self.fmodel.reset_parameters()
        self.smodel.reset_parameters()

    def parameters(self):
        parameters = super().parameters()
        parameters += self.fmodel.parameters()
        parameters += self.smodel.parameters()
        return parameters

    def train(self, mode: bool = True):
        self.fmodel.train(mode)
        self.smodel.train(mode)

    def eval(self):
        self.fmodel.eval()
        self.smodel.eval()

    def zero_grad(self, set_to_none=False):
        self.fmodel.zero_grad(set_to_none=set_to_none)
        self.smodel.zero_grad(set_to_none=set_to_none)

    def restart(self):
        super().restart()
        self.fmodel = None
        self.smodel = None

    def reset(self):
        super().reset()
        self.fmodel.reset()
        self.smodel.reset()

    def get_UD(self):
        return self.smodel.get_UD()

    def set_QD(self, U, D):
        self.smodel.set_QD(U, D)

    def get_SFV(self):
        return self.smodel.get_SFV()

    def get_x(self):
        return self.smodel.get_x()

    def get_D(self):
        return self.smodel.get_D()

    def get_embeddings(self):
        H = self.fmodel.get_embeddings()
        S = self.smodel.get_embeddings()
        O = H + S
        return O

    def __call__(self):
        return self.get_embeddings()

    def intrinsic_regularizer(self):
        return self.smodel.intrinsic_regularizer()

    def ambient_regularizer(self):
        return self.smodel.ambient_regularizer()

    def calc_mask_metric(self, mask="test", metric=""):
        res = super().calc_mask_metric(mask=mask, metric=metric)
        f_res = self.fmodel.calc_mask_metric(mask=mask, metric=metric)
        s_res = self.smodel.calc_mask_metric(mask=mask, metric=metric)

        return res + f_res + s_res


class FedClassifier(FedMixin, Classifier):
    def __init__(self, fgraph: Graph, sgraph: Graph):
        super().__init__(fgraph)
        self.fmodel: Classifier = None
        self.create_fmodel(fgraph)
        self.smodel: Classifier = None
        self.create_smodel(sgraph)

    def create_smodel(self, sgraph: Graph):
        raise NotImplementedError

    def create_fmodel(self, fgraph) -> Classifier:
        if isinstance(fgraph, AGraph):
            self.fmodel = DGCN(fgraph)
        if isinstance(fgraph, Graph):
            self.fmodel = FGNN(fgraph)

    def get_prediction(self):
        H = self.get_embeddings()
        if config.dataset.multi_label:
            y_pred = torch.nn.functional.sigmoid(H)
        else:
            y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def train_step(self, eval_=True):
        res = super().train_step(eval_=eval_)

        if eval_:
            f_res = self.fmodel.calc_mask_metric(mask="val", metric="acc")
            s_res = self.smodel.calc_mask_metric(mask="val", metric="acc")
            f_test = self.fmodel.calc_mask_metric(mask="test", metric="acc")
            s_test = self.smodel.calc_mask_metric(mask="test", metric="acc")
            return res + f_res + s_res + f_test + s_test
        else:
            return res


class FedEdgeClassifier(FedMixin, EdgeClassifier):
    def __init__(
        self,
        fgraph: Graph,
        sgraph: Graph,
        link_feature_operator: LinkFeatureOperator = "hadamard",
    ):
        super().__init__(fgraph, link_feature_operator)
        self.fmodel: EdgeClassifier = None
        self.create_fmodel(fgraph, link_feature_operator)
        self.smodel: EdgeClassifier = None
        self.create_smodel(sgraph, link_feature_operator)
        # FIXME: This is a hack to get have logistic regression model for combined edge prediction as well
        self.logistic_regression_model = self.fmodel.logistic_regression_model
        self.smodel.logistic_regression_model = self.fmodel.logistic_regression_model

    def create_smodel(self, sgraph: Graph, link_feature_operator: LinkFeatureOperator):
        raise NotImplementedError

    def create_fmodel(
        self, fgraph: Graph, link_feature_operator: LinkFeatureOperator
    ) -> EdgeClassifier:
        if isinstance(fgraph, Graph):
            self.fmodel = FEdgeGNN(fgraph, link_feature_operator)
        # Note: Edge prediction for AGraph/DGCN not yet implemented
        # if isinstance(fgraph, AGraph):
        #     self.fmodel = EdgeDGCN(fgraph, link_feature_operator)

    def get_edge_embeddings(self, edge_index: torch.Tensor) -> torch.Tensor:
        H: torch.Tensor = self.get_embeddings()  # Combined H + S
        src_emb = H[edge_index[0]]  # [num_edges, embed_dim]
        tgt_emb = H[edge_index[1]]  # [num_edges, embed_dim]

        match self.link_feature_operator:
            case "dot-product":  # Inner product: returns [num_edges] scalar scores
                return (src_emb * tgt_emb).sum(dim=-1)
            case "hadamard":  # Element-wise product: returns [num_edges, embed_dim]
                return src_emb * tgt_emb
            case "concat":  # Concatenation: returns [num_edges, 2*embed_dim]
                return torch.cat([src_emb, tgt_emb], dim=-1)
            case _:
                raise NotImplementedError(
                    f"Operator {self.link_feature_operator} not implemented"
                )

    def train_step(self, eval_=True):
        res = super().train_step(eval_=eval_)

        if eval_:
            # Use AUC for edge prediction instead of accuracy
            f_res = self.fmodel.calc_mask_metric(mask="val", metric="auc")
            s_res = self.smodel.calc_mask_metric(mask="val", metric="auc")
            f_test = self.fmodel.calc_mask_metric(mask="test", metric="auc")
            s_test = self.smodel.calc_mask_metric(mask="test", metric="auc")
            return res + f_res + s_res + f_test + s_test
        else:
            return res


class FedSlave(FedClassifier):
    def __init__(self, graph: Data, server_embedding_func):
        Classifier.__init__(self, graph)
        self.create_fmodel(graph)
        self.create_smodel(graph, server_embedding_func)

    def create_smodel(self, graph: Data, server_embedding_func):
        self.smodel = SGNNSlave(graph, server_embedding_func)

    def state_dict(self):
        weights = {}
        weights["fmodel"] = self.fmodel.state_dict()
        return weights

    def load_state_dict(self, weights):
        self.fmodel.load_state_dict(weights["fmodel"])


class FedGNNMaster(FedClassifier):
    def __init__(self, fgraph: Graph, sgraph: Graph):
        FedClassifier.__init__(self, fgraph, sgraph)

    def create_smodel(self, sgraph: Graph):
        self.smodel = SGNNMaster(sgraph)

    def get_embeddings(self, node_ids=None):
        H = self.fmodel()
        S = self.smodel(node_ids)
        O = H + S
        return O

    def get_embeddings_func(self):
        return self.smodel.get_embeddings

    def state_dict(self):
        weights = {}
        weights["fmodel"] = self.fmodel.state_dict()
        return weights

    def load_state_dict(self, weights):
        self.fmodel.load_state_dict(weights["fmodel"])


class FedDGCN(FedClassifier):
    def create_smodel(self, sgraph: AGraph):
        self.smodel = SDGCN(sgraph)


class FedSpectralDGCN(FedClassifier):
    def create_smodel(self, sgraph: AGraph):
        self.smodel = SpectralDGCN(sgraph)


class FedDGCNMaster(FedGNNMaster):
    def create_smodel(self, sgraph: AGraph):
        self.smodel = SDGCNMaster(sgraph)


class FedLaplaceClassifier(FedClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = SLaplace(sgraph)


class FedSpectralLaplaceClassifier(FedClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = SpectralLaplace(sgraph)


class FedLanczosLaplaceClassifier(FedClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = LanczosLaplace(sgraph)


class FedMLPClassifier(FedClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = SClassifier(sgraph)


# Edge prediction analogues
class FedLaplaceEdgeClassifier(FedEdgeClassifier):
    def create_smodel(self, sgraph: Graph, link_feature_operator: LinkFeatureOperator):
        self.smodel = SEdgeLaplace(sgraph, link_feature_operator)


class FedSpectralLaplaceEdgeClassifier(FedEdgeClassifier):
    def create_smodel(self, sgraph: Graph, link_feature_operator: LinkFeatureOperator):
        self.smodel = SpectralEdgeLaplace(sgraph, link_feature_operator)


class FedLanczosLaplaceEdgeClassifier(FedEdgeClassifier):
    def create_smodel(self, sgraph: Graph, link_feature_operator: LinkFeatureOperator):
        self.smodel = LanczosEdgeLaplace(sgraph, link_feature_operator)


# Dynamic/Temporal Classifiers with temporal encoder support
class FedDynamicClassifier(FedClassifier):
    def __init__(
        self, fgraph: Graph, sgraph: Graph, num_ss: int, model_config: TrainModelConfig
    ):
        super().__init__(fgraph, sgraph)
        embed_dim = config.feature_model.gnn_layer_sizes[-1]
        assert num_ss is not None
        self.model_config = model_config
        self.tmodel: TemporalBlock | None = None
        self.create_tmodel(edim=embed_dim, num_snapshots=num_ss)
        self.stored_sfvs_grad: dict[int, torch.Tensor] = {
            ss_idx: torch.zeros(
                config.spectral.spectral_len,
                config.structure_model.num_structural_features,
            ).to(device=device)
            for ss_idx in range(num_ss)
        }

    def create_fmodel(self, fgraph) -> None:
        if isinstance(fgraph, AGraph):
            self.fmodel = DGCN(fgraph)
        if isinstance(fgraph, Graph):
            self.fmodel = NewFGNN(fgraph)

    def create_tmodel(self, edim: int, num_snapshots: int):
        assert edim == self.model_config.temporal.input_dimension
        self.tmodel = TemporalBlock(
            edim=edim,
            num_layers=self.model_config.temporal.num_attention_layers,
            heads=self.model_config.temporal.attention_heads,
            dropout=self.model_config.temporal.dropout,
            num_ss=num_snapshots,
        ).to(device)

    def state_dict(self):
        weights = super().state_dict()
        if self.tmodel is not None:
            weights["tmodel"] = self.tmodel.state_dict()
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        if "tmodel" in weights and self.tmodel is not None:
            self.tmodel.load_state_dict(weights["tmodel"])

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        _ = grads["smodel"].pop("SFV", None)  # Remove SFV if present
        for key, val in self.stored_sfvs_grad.items():
            if val.requires_grad and val.grad is not None:
                grads["smodel"][f"SFVs-ss{key}"] = val.grad
        if self.tmodel is not None:
            grads["tmodel"] = self.tmodel.get_grads()
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if "smodel" in grads:
            for key in self.stored_sfvs_grad.keys():
                sfv_key = f"SFVs-ss{key}"
                if sfv_key in grads["smodel"]:
                    self.stored_sfvs_grad[key].grad = grads["smodel"][sfv_key]
        if "tmodel" in grads and self.tmodel is not None:
            self.tmodel.set_grads(grads["tmodel"])

    def reset_parameters(self):
        super().reset_parameters()
        if self.tmodel is not None:
            if hasattr(self.tmodel, "reset_parameters"):
                self.tmodel.reset_parameters()
            else:
                for module in self.tmodel.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)

    def parameters(self):
        parameters = super().parameters()
        if self.tmodel is not None:
            parameters += list(self.tmodel.parameters())
        for sfv in self.stored_sfvs_grad.values():
            if sfv is not None and sfv.requires_grad:
                parameters.append(sfv)
        return parameters

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tmodel is not None:
            self.tmodel.train(mode)

    def eval(self):
        super().eval()
        if self.tmodel is not None:
            self.tmodel.eval()

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none=set_to_none)
        if self.tmodel is not None:
            self.tmodel.zero_grad(set_to_none=set_to_none)
        for sfv in self.stored_sfvs_grad.values():
            if sfv.requires_grad:
                if set_to_none:
                    sfv.grad = None
                else:
                    if sfv.grad is not None:
                        sfv.grad.zero_()

    def restart(self):
        super().restart()
        self.tmodel = None

    def reset(self):
        super().reset()
        if self.tmodel is not None:
            self.tmodel.zero_grad(set_to_none=True)
        # Reset stored_sfvs_grad to zero tensors for new epoch
        num_ss = len(self.stored_sfvs_grad)
        self.stored_sfvs_grad = {
            ss_idx: torch.zeros(
                config.spectral.spectral_len,
                config.structure_model.num_structural_features,
            ).to(device=device)
            for ss_idx in range(num_ss)
        }

    def get_temporal_embeddings(self, stored_embeddings: dict[int, torch.Tensor]):
        if self.tmodel is None:
            raise ValueError(
                "Temporal model (tmodel) is not initialized. Call create_tmodel first."
            )
        if len(stored_embeddings) == 0:
            raise ValueError("No stored embeddings provided.")
        stored_embeddings_list = [
            stored_embeddings[ss_idx]
            for ss_idx in sorted(stored_embeddings.keys())
        ]
        max_num_nodes = max(map(lambda x: len(x), stored_embeddings_list))
        embeddings = []
        for emb in stored_embeddings_list:
            num_nodes = emb.shape[0]
            embed_dim = emb.shape[1]
            zero_pad = torch.zeros(
                max_num_nodes - num_nodes, embed_dim, device=emb.device, dtype=emb.dtype
            )
            emb = torch.cat([emb, zero_pad], dim=0)
            emb = emb.unsqueeze(0)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.transpose(0, 1)
        temporal_embeddings = self.tmodel(embeddings)
        temporal_embeddings = temporal_embeddings.transpose(0, 1)
        return temporal_embeddings

    def get_edge_embeddings(
        self,
        edge_index: torch.Tensor,
        embeddings: torch.Tensor,
        operator: LinkFeatureOperator = "hadamard",
    ) -> np.ndarray:
        match operator:
            case "hadamard":
                # Hadamard product: element-wise multiplication
                src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
                tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
                return (src_emb * tgt_emb).cpu().numpy()
            case "dot-product":
                # Dot product: inner product (sum of element-wise multiplication)
                src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
                tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
                return (src_emb * tgt_emb).sum(dim=-1).cpu().numpy()
            case _:
                raise NotImplementedError(f"Operator {operator} not implemented")

    def compute_random_walk_loss(
        self,
        temporal_embeddings: torch.Tensor,
        context_pairs: dict[int, list[int]],
        snapshot_idx: int,
        edge_index: torch.Tensor,
        neg_sample_size: int = 10,
        neg_weight: float = 1.0,
        batch_size: int = 512,
        neg_sampling_distortion: float = 0.75,
        neg_sampling_retries: int = 5,
    ):
        z = temporal_embeddings
        context_dict = context_pairs

        nodes_with_pairs = list(context_dict.keys())
        if len(nodes_with_pairs) == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        criterion = torch.nn.BCEWithLogitsLoss()

        # Compute degree-based unigram distribution for negative sampling
        num_nodes_total = z.shape[0]
        degrees = get_degrees(edge_index, None, num_nodes_total, device, "both")
        unigram_probs = torch.pow(degrees + 1e-8, neg_sampling_distortion)
        unigram_probs = unigram_probs / unigram_probs.sum()

        num_nodes = len(nodes_with_pairs)
        num_batches = (num_nodes + batch_size - 1) // batch_size
        step_losses = []
        sample_size_tensor = torch.tensor(neg_sample_size, device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_nodes)
            batch_nodes = nodes_with_pairs[start_idx:end_idx]

            context_lengths = torch.tensor(
                [len(context_dict[node]) for node in batch_nodes],
                dtype=torch.long,
                device=device,
            )
            actual_sample_sizes = torch.minimum(context_lengths, sample_size_tensor)
            total_pairs = int(actual_sample_sizes.sum().item())

            if total_pairs == 0:
                continue

            src_nodes = torch.zeros(total_pairs, dtype=torch.long, device=device)
            tgt_nodes = torch.zeros(total_pairs, dtype=torch.long, device=device)

            offset = 0
            for i, node in enumerate(batch_nodes):
                contexts = context_dict[node]
                num_samples = actual_sample_sizes[i].item()
                if len(contexts) > num_samples:
                    context_tensor = torch.tensor(
                        contexts, dtype=torch.long, device=device
                    )
                    sampled_indices = torch.randperm(len(contexts), device=device)[
                        :num_samples
                    ]
                    sampled_contexts = context_tensor[sampled_indices]
                else:
                    sampled_contexts = torch.tensor(
                        contexts, dtype=torch.long, device=device
                    )
                src_nodes[offset : offset + num_samples] = node
                tgt_nodes[offset : offset + num_samples] = sampled_contexts
                offset += num_samples

            pos_src_emb = z[src_nodes]
            pos_tgt_emb = z[tgt_nodes]
            pos_scores = (pos_src_emb * pos_tgt_emb).sum(dim=-1)

            pos_labels = torch.ones_like(pos_scores)
            pos_loss = criterion(pos_scores, pos_labels)

            num_pos = len(src_nodes)
            neg_nodes = torch.multinomial(
                unigram_probs, num_pos * neg_sample_size, replacement=True
            ).view(num_pos, neg_sample_size)
            tgt_expanded = tgt_nodes.view(-1, 1)
            for _ in range(neg_sampling_retries):
                mask = neg_nodes == tgt_expanded
                if not mask.any():
                    break
                resample = torch.multinomial(
                    unigram_probs, int(mask.sum().item()), replacement=True
                )
                neg_nodes[mask] = resample
            mask = neg_nodes == tgt_expanded
            if mask.any():
                fallback = (tgt_expanded + 1) % z.shape[0]
                neg_nodes = torch.where(mask, fallback, neg_nodes)

            neg_emb = z[neg_nodes]
            neg_scores = (-1.0) * (pos_src_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
            neg_labels = torch.ones_like(neg_scores)
            neg_loss = criterion(neg_scores, neg_labels)
            batch_loss = pos_loss.mean() + neg_weight * neg_loss.mean()

            step_losses.append(batch_loss)

        if len(step_losses) == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        return torch.stack(step_losses).mean()


class FedDynamicLanczosLaplaceClassifier(FedDynamicClassifier):
    def create_smodel(self, sgraph: Graph):
        self.smodel = LanczosLaplaceNew(sgraph)
