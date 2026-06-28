from src import *

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn.models import Node2Vec


def train(model: Node2Vec, loader, optimizer: torch.optim.SparseAdam):
    model.train()
    total_loss = 0
    count = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / len(loader)


@torch.no_grad()
def test(model, graph):
    model.eval()
    z = model()
    acc = model.test(
        z[graph.train_mask],
        graph.y[graph.train_mask],
        z[graph.test_mask],
        graph.y[graph.test_mask],
        max_iter=100,
    )
    return acc


def find_node2vect_embedings(
    edge_index,
    epochs=config["node2vec"]["epochs"],
    embedding_dim=64,
    show_bar=config["node2vec"]["show_bar"],
    plot=False,
):
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=config["node2vec"]["walk_length"],
        context_size=config["node2vec"]["context_size"],
        walks_per_node=config["node2vec"]["walks_per_node"],
        num_negative_samples=config["node2vec"]["num_negative_samples"],
        p=config["node2vec"]["p"],
        q=config["node2vec"]["q"],
        sparse=True,
    )
    # model.to(device=dev)

    loader = model.loader(batch_size=config["node2vec"]["batch_size"], shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config["node2vec"]["lr"])

    if show_bar:
        bar = tqdm(total=epochs)
    res = []
    for epoch in range(epochs):
        loss = train(model, loader, optimizer)
        if show_bar:
            bar.set_description(f"Epoch: {epoch:02d}")
            bar.set_postfix({"Loss": loss})
            bar.update()

        res.append(loss)

    # print(f"test accuracy: {test(model, )}")
    if plot:
        plt.plot(res)

    model.eval()
    z = model()
    z = z.detach()

    return z
