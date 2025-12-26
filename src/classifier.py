import torch
from src import *
from src.utils.data import Data
from src.utils.graph import Graph
from src.models.model_binders import ModelBinder


class Classifier:
    def __init__(self, graph=None):
        self.graph: Graph | Data | None = graph
        self.model: ModelBinder | None = None
        self.optimizer = None

    def create_smodel(self):
        raise NotImplementedError

    def create_optimizer(self):
        parameters = self.parameters()
        if len(parameters) == 0:
            return
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
        )

    def state_dict(self):
        weights = {}
        if self.model is not None:
            weights["model"] = self.model.state_dict()

        return weights

    def load_state_dict(self, weights):
        if self.model is not None:
            self.model.load_state_dict(weights["model"])

    def get_grads(self, just_SFV=False):
        if just_SFV:
            return {}
        grads = {}
        if self.model is not None:
            grads["model"] = self.model.get_grads()

        return grads

    def set_grads(self, grads):
        if "model" in grads.keys():
            self.model.set_grads(grads["model"])

    def reset_parameters(self):
        if self.model is not None:
            self.model.reset_parameters()

    def parameters(self):
        parameters = []
        if self.model is not None:
            parameters += self.model.parameters()

        return parameters

    def train(self, mode: bool = True):
        if self.model is not None:
            self.model.train(mode)

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def zero_grad(self, set_to_none=False):
        if self.model is not None:
            self.model.zero_grad(set_to_none=set_to_none)

    def update_model(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def reset(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def restart(self):
        self.graph = None
        self.model = None
        self.optimizer = None

    def get_UD(self):
        return None, None

    def set_QD(self, U, D):
        pass

    def get_embeddings(self):
        raise NotImplementedError

    def get_embeddings_func(self):
        return self.get_embeddings

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        H = self.get_embeddings()
        if config.dataset.multi_label:
            y_pred = torch.nn.functional.sigmoid(H)
        else:
            y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def get_SFV(self):
        # return self()
        return self.graph.x

    def get_x(self):
        return self.graph.x

    def get_D(self):
        return None

    def intrinsic_regularizer(self):
        return 0

    def ambient_regularizer(self):
        return 0

    def train_step(self, eval_=True) -> tuple[float, ...]:
        label_loss, train_acc = Classifier.calc_mask_metric(self, mask="train")
        intrinsic_loss = self.intrinsic_regularizer()
        # ambient_loss = self.ambient_regularizer()
        train_loss = (
            1 * label_loss + config.spectral.regularizer_coef * intrinsic_loss
            # + 0 * ambient_loss
        )

        train_loss.backward(retain_graph=True)

        if eval_:
            (test_acc,) = Classifier.calc_mask_metric(self, mask="test", metric="acc")
            if self.graph.val_mask is not None:
                val_loss, val_acc = Classifier.calc_mask_metric(self, mask="val")
                return train_loss.item(), train_acc, val_loss.item(), val_acc, test_acc
            else:
                return train_loss.item(), train_acc, 0, 0, test_acc
        else:
            return train_loss.item(), train_acc

    def calc_mask_metric(self, mask="test", metric="", loss_function="cross_entropy"):
        if mask == "train":
            self.train()
            metric_mask = self.graph.train_mask
        elif mask == "val":
            self.eval()
            metric_mask = self.graph.val_mask
        elif mask == "test":
            self.eval()
            metric_mask = self.graph.test_mask
        return Classifier.calc_metrics(
            self, self.graph.y, metric_mask, metric, loss_function=loss_function
        )

    # @torch.no_grad()
    @staticmethod
    def calc_metrics(
        model, y, mask, metric="", loss_function="cross_entropy"
    ) -> tuple[float | torch.Tensor, ...]:
        # model.eval()
        y_pred = model.get_prediction()

        loss, acc, f1_score, precision, recall, auc, ap = calc_metrics(
            y,
            y_pred,
            mask,
            loss_function=loss_function,
        )

        if metric == "acc":
            return (acc.item(),)
        elif metric == "f1":
            return (f1_score.item(),)
        elif metric == "precision":
            return (precision.item(),)
        elif metric == "recall":
            return (recall.item(),)
        elif metric == "ap":
            return (ap.item(),)
        elif metric == "auc":
            return (auc.item(),)
        elif metric == "loss":
            return (loss.item(),)
        else:
            return loss, acc.item()


class EdgeClassifier(Classifier):
    """
    Edge prediction classifier that extends the base Classifier.
    Works with edge_index/edge_label instead of node masks.
    """

    def __init__(
        self, graph=None, link_feature_operator: LinkFeatureOperator = "dot-product"
    ):
        super().__init__(graph)
        self.link_feature_operator = link_feature_operator
        self.logistic_regression_model: ModelBinder | None = None

    def state_dict(self):
        weights = super().state_dict()
        if self.logistic_regression_model is not None:
            weights["logistic_regression_model"] = (
                self.logistic_regression_model.state_dict()
            )
        return weights

    def load_state_dict(self, weights):
        super().load_state_dict(weights)
        if (
            "logistic_regression_model" in weights
            and self.logistic_regression_model is not None
        ):
            self.logistic_regression_model.load_state_dict(
                weights["logistic_regression_model"]
            )

    def get_grads(self, just_SFV=False):
        grads = super().get_grads(just_SFV)
        if not just_SFV and self.logistic_regression_model is not None:
            grads["logistic_regression_model"] = (
                self.logistic_regression_model.get_grads()
            )
        return grads

    def set_grads(self, grads):
        super().set_grads(grads)
        if (
            "logistic_regression_model" in grads.keys()
            and self.logistic_regression_model is not None
        ):
            self.logistic_regression_model.set_grads(grads["logistic_regression_model"])

    def reset_parameters(self):
        super().reset_parameters()
        if self.logistic_regression_model is not None:
            self.logistic_regression_model.reset_parameters()

    def parameters(self):
        parameters = super().parameters()
        if self.logistic_regression_model is not None:
            parameters += list(self.logistic_regression_model.parameters())
        return parameters

    def train(self, mode: bool = True):
        super().train(mode)
        if self.logistic_regression_model is not None:
            self.logistic_regression_model.train(mode)

    def eval(self):
        super().eval()
        if self.logistic_regression_model is not None:
            self.logistic_regression_model.eval()

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none)
        if self.logistic_regression_model is not None:
            self.logistic_regression_model.zero_grad(set_to_none=set_to_none)

    def restart(self):
        super().restart()
        self.link_feature_operator = None

    def _get_edge_features(self, edge_index: torch.Tensor) -> torch.Tensor:
        H: torch.Tensor = self.get_embeddings()
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

    def get_prediction(self, edge_index: torch.Tensor | None = None):
        if edge_index is None:
            raise ValueError("edge_index cannot be None for edge prediction")
        else:
            edge_features = self._get_edge_features(edge_index)

            if self.link_feature_operator == "dot-product":
                return edge_features
            else:
                if self.logistic_regression_model is None:
                    raise RuntimeError(
                        f"Current operator: {self.link_feature_operator}."
                        "logistic_regression_model must be initialized first!"
                    )

                edge_logits = self.logistic_regression_model(edge_features)
                return edge_logits.squeeze(-1)  # [num_edges, 1] -> [num_edges]

    @staticmethod
    def calc_metrics(
        model,
        edge_index: torch.Tensor,
        edge_label: torch.Tensor,
        metric="",
        loss_function="BCELoss",
    ) -> tuple[float | torch.Tensor, ...]:
        edge_logits = model.get_prediction(edge_index)  # [num_edges]

        loss, acc, f1_score, precision, recall, auc, ap = calc_metrics(
            edge_label,
            edge_logits,
            mask=None,
            loss_function=loss_function,
        )

        if metric == "acc":
            return (acc.item(),)
        elif metric == "f1":
            return (f1_score.item(),)
        elif metric == "precision":
            return (precision.item(),)
        elif metric == "recall":
            return (recall.item(),)
        elif metric == "auc":
            return (auc.item(),)
        elif metric == "ap":
            return (ap.item(),)
        elif metric == "loss":
            return (loss.item(),)
        else:
            return loss, auc.item()

    def calc_mask_metric(
        self, mask="test", metric="", loss_function="BCELoss"
    ) -> tuple[float | torch.Tensor, ...]:
        """
        Calculate metrics for edge prediction using edge_index splits.
        """
        if mask == "train":
            self.train()
            edge_index = self.graph.train_edge_index
            edge_label = self.graph.train_edge_label
        elif mask == "val":
            self.eval()
            edge_index = self.graph.val_edge_index
            edge_label = self.graph.val_edge_label
        elif mask == "test":
            self.eval()
            edge_index = self.graph.test_edge_index
            edge_label = self.graph.test_edge_label
        else:
            raise ValueError(f"Unknown mask: {mask}")

        if edge_index is None or edge_label is None:
            raise ValueError(f"Edge data not available for mask: {mask}")

        return EdgeClassifier.calc_metrics(
            self, edge_index, edge_label, metric, loss_function
        )

    def train_step(self, eval_=True) -> tuple[float, ...]:
        train_loss, train_auc = EdgeClassifier.calc_mask_metric(
            self, mask="train", loss_function="BCELoss"
        )
        intrinsic_loss = self.intrinsic_regularizer()
        train_loss = 1 * train_loss + config.spectral.regularizer_coef * intrinsic_loss

        train_loss.backward(retain_graph=True)

        if eval_:
            (test_auc,) = EdgeClassifier.calc_mask_metric(
                self, mask="test", metric="auc", loss_function="BCELoss"
            )
            if (
                hasattr(self.graph, "val_edge_index")
                and self.graph.val_edge_index is not None
            ):
                val_loss, val_auc = EdgeClassifier.calc_mask_metric(
                    self, mask="val", loss_function="BCELoss"
                )
                return train_loss.item(), train_auc, val_loss.item(), val_auc, test_auc
            else:
                return train_loss.item(), train_auc, 0.0, 0.0, test_auc

        else:
            return train_loss.item(), train_auc
