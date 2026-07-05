from torch_geometric.loader import NeighborLoader

from src import *
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster, SpectralDGCN
from src.GNN.fGNN import FGNN
from src.GNN.laplace import LanczosLaplace, SLaplace, SpectralLaplace
from src.GNN.sGNN import SClassifier, SGNNMaster, SGNNSlave
from src.utils.graph import AGraph, Graph
from src.utils.data import Data
from src.classifier import Classifier


class FedClassifier(Classifier):
    def __init__(self, fgraph: Graph, sgraph: Graph):
        super().__init__(fgraph)
        self.fmodel: Classifier = None
        self.create_fmodel(fgraph)
        self.smodel: Classifier = None
        self.create_smodel(sgraph)

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

    def create_smodel(self):
        raise NotImplementedError

    def create_fmodel(self, fgraph) -> Classifier:
        if isinstance(fgraph, AGraph):
            self.fmodel = DGCN(fgraph)
        if isinstance(fgraph, Graph):
            self.fmodel = FGNN(fgraph)

    def get_SFV(self):
        # return self.smodel()
        return self.smodel.get_SFV()

    def get_x(self):
        return self.smodel.get_x()

    def get_D(self):
        return self.smodel.get_D()

    def get_embeddings(self):
        H = self.fmodel()
        S = self.smodel()
        O = H + S
        return O

    def __call__(self):
        return self.get_embeddings()

    def get_prediction(self):
        H = self.get_embeddings()
        if config.dataset.multi_label:
            y_pred = torch.nn.functional.sigmoid(H)
        else:
            y_pred = torch.nn.functional.softmax(H, dim=1)
        return y_pred

    def intrinsic_regularizer(self):
        return self.smodel.intrinsic_regularizer()

    def ambient_regularizer(self):
        return self.smodel.ambient_regularizer()

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

    def calc_mask_metric(self, mask="test", metric=""):
        res = super().calc_mask_metric(mask=mask, metric=metric)
        f_res = self.fmodel.calc_mask_metric(mask=mask, metric=metric)
        s_res = self.smodel.calc_mask_metric(mask=mask, metric=metric)

        return res + f_res + s_res


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
