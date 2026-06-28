import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src import *


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        last_layer="softmax",
        dropout=0.5,
        normalization=None,
    ):
        super().__init__()
        self.type_ = "MLP"
        self.num_layers = len(layer_sizes) - 1
        self.dropout = dropout
        self.last_layer = last_layer
        self.normalization = normalization

        self.layers = self.create_models(layer_sizes)
        # self.net = nn.Sequential(*self.layers)

        # self.default_weights = self.state_dict()
        # self.default_weights = deepcopy(self.state_dict())

    def __getitem__(self, item):
        return self.layers[item]

    def create_models(self, layer_sizes):
        layers = nn.ParameterList()

        if self.normalization == "batch":
            norm_layer = nn.BatchNorm1d(
                layer_sizes[0], affine=True, track_running_stats=False
            )
            layers.append(norm_layer)
        elif self.normalization == "layer":
            norm_layer = nn.LayerNorm(layer_sizes[0])
            layers.append(norm_layer)
        elif self.normalization == "instance":
            norm_layer = nn.InstanceNorm1d(layer_sizes[0])
            layers.append(norm_layer)

        for layer_num in range(self.num_layers):
            layer = nn.Linear(layer_sizes[layer_num], layer_sizes[layer_num + 1])
            layers.append(layer)
            if layer_num < self.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))

        if self.last_layer == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif self.last_layer == "relu":
            layers.append(nn.ReLU())
        elif self.last_layer == "tanh":
            layers.append(nn.Tanh())
        elif self.last_layer == "sigmoid":
            layers.append(nn.Sigmoid())

        return layers

    def reset_parameters(self) -> None:
        # self.load_state_dict(self.default_weights)
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except:
                pass

    def state_dict(self):
        weights = {}
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            weights[f"layer{id}"] = layer.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, layer in enumerate(self.layers):
            if (
                isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.BatchNorm1d)
                or isinstance(layer, nn.InstanceNorm1d)
            ):
                continue
            layer.load_state_dict(weights[f"layer{id}"])

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]

        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.layers:
            layer.train(mode)

    def val(self):
        super().val()
        for layer in self.layers:
            layer.val()

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)

        return h
        # return self.net(x)

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=1,
        verbose=False,
        plot=False,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config["model"]["lr"],
            weight_decay=config["model"]["weight_decay"],
        )

        self.train()
        train_acc_list = []
        val_acc_list = []
        for epoch in range(epochs + 1):
            # Training
            optimizer.zero_grad()
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Print metrics every 10 epochs
            if verbose and x_val is not None:
                acc = calc_accuracy(y_pred.argmax(dim=1), y_train)
                f1_score = calc_f1_score(y_pred.argmax(dim=1), y_train)
                # Validation
                y_pred_val = self(x_val)
                val_loss = criterion(y_pred_val, y_val)
                val_acc = calc_accuracy(y_pred_val.argmax(dim=1), y_val)
                val_f1_score = calc_f1_score(y_pred_val.argmax(dim=1), y_val)

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:"
                        f" {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | "
                        f"F1 Score: {f1_score*100:.2f}% | "
                        f"Val Acc: {val_acc*100:.2f}% | "
                        f"Val F1 Score: {val_f1_score*100:.2f}%",
                        end="\r",
                    )

                train_acc_list.append(acc)
                val_acc_list.append(val_acc)

        if verbose:
            print("\n")

        if plot:
            plt.figure()
            plt.plot(train_acc_list)
            plt.plot(val_acc_list)

        self.eval()
        y_pred_val = self(x_val)
        val_loss = criterion(y_pred_val, y_val)
        val_acc = calc_accuracy(y_pred_val.argmax(dim=1), y_val)

        return val_acc, val_loss
        # return loss, val_loss, acc, val_acc, TP, val_TP

    @torch.no_grad()
    def test(self, x, y):
        self.eval()
        out = self(x)
        test_accuracy = calc_accuracy(out.argmax(dim=1), y)
        return test_accuracy
