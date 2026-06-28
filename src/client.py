from tqdm import tqdm

from src import *
from src.utils.graph import Graph
from src.classifier import Classifier


class Client:
    def __init__(
        self,
        graph: Graph,
        id: int = 0,
        classifier_type="GNN",
    ):
        self.id = id
        self.graph = graph
        self.classifier_type = classifier_type

        # LOGGER.info(f"Client{self.id} statistics:")
        # LOGGER.info(f"Number of nodes: {self.graph.num_nodes}")

        self.classifier: Classifier = None

    def get_nodes(self):
        return self.graph.node_ids

    def num_nodes(self) -> int:
        return len(self.graph.node_ids)

    def parameters(self):
        return self.classifier.parameters()

    def zero_grad(self):
        self.classifier.zero_grad()

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def state_dict(self):
        return self.classifier.state_dict()

    def load_state_dict(self, weights):
        self.classifier.load_state_dict(weights)

    def train(self, mode: bool = True):
        self.classifier.train(mode)

    def eval(self):
        self.classifier.eval()

    def test_classifier(self, metric=config["model"]["metric"]):
        return self.classifier.calc_mask_metric(mask="test", metric=metric)

    def get_train_results(self, eval_=True):
        res = self.train_step(eval_=eval_)

        metrics = [
            "Train Loss",
            "Train Acc",
            "Val Loss",
            "Val Acc",
            "Test Acc",
            "Val F Acc",
            "Val S Acc",
            "Test F Acc",
            "Test S Acc",
        ]
        result = {}
        for metric, val in zip(metrics, res):
            result[metric] = round(val, 4)

        return result

    def get_test_results(self):
        res = self.test_classifier()
        metrics = [
            "Test Acc",
            "Test Acc F",
            "Test Acc S",
        ]
        result = {}
        for metric, val in zip(metrics, res):
            result[metric] = val

        return result

    def report_result(self, result, framework=""):
        LOGGER.info(f"{framework} results for client{self.id}:")
        LOGGER.info(f"{result}")

    def train_local_model(
        self,
        epochs=config["model"]["iterations"],
        log=True,
        plot=True,
        model_type="GNN",
    ):
        if log:
            LOGGER.info("local training starts!")

        if log:
            bar = tqdm(total=epochs, position=0)

        results = []
        for epoch in range(epochs):
            self.reset_model()

            self.train()
            result = self.get_train_results(eval_=log)
            result["Epoch"] = epoch + 1
            results.append(result)

            self.update_model()

            if log:
                bar.set_postfix(result)
                bar.update()

                if epoch == epochs - 1:
                    self.report_result(result, "Local Training")

        if plot:
            title = f"{model_type}"
            plot_path = f"{save_path}/plots/{now}/"
            plot_metrics(results, title=title, save_path=plot_path)

        test_results = self.get_test_results()
        if log:
            for key, val in test_results.items():
                LOGGER.info(f"Client {self.id} {key}: {val:0.4f}")

        return test_results

    def get_grads(self, just_SFV=False):
        return self.classifier.get_grads(just_SFV)

    def set_grads(self, grads):
        self.classifier.set_grads(grads)

    def update_model(self):
        self.classifier.update_model()

    def reset_model(self):
        self.classifier.reset()

    def train_step(self, eval_=True):
        return self.classifier.train_step(eval_=eval_)
