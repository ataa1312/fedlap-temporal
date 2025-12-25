from typing import Literal

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
