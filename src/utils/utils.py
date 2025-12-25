import os
import random
import itertools
from datetime import datetime
from typing import Literal

import imageio
import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_sparse import SparseTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.utils import degree, add_self_loops, scatter, remove_self_loops
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.cm as cm

from src.utils.config_parser import Config
from src.utils.logger import getLOGGER
from src.utils.plot_graph import plot_graph

LinkFeatureOperator = Literal["hadamard", "dot-product", "concat"]

load_dotenv()
config_path = os.environ.get("CONFIGPATH", "")
config = Config(config_path)

plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 15

if torch.cuda.is_available():
    dev = "cuda:0"
# elif torch.backends.mps.is_available():
#     dev = "mps"
else:
    dev = "cpu"
device = torch.device(dev)

if dev == "mps":
    local_dev = "cpu"
else:
    local_dev = dev


seed = int(os.environ.get("seed", 1))
seed = seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

now = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = os.environ.get("RESULTPATH")
save_path = (
    f"{result_path}"
    f"{config.dataset.dataset_name}/"
    f"{config.structure_model.structure_type}/"
    f"{config.subgraph.partitioning}/"
    f"{config.model.smodel_type}/"
    f"{config.subgraph.num_subgraphs}/all/"
)

os.makedirs(save_path, exist_ok=True)
LOGGER = getLOGGER(
    name=f"{now}_{config.dataset.dataset_name}",
    log_on_file=True,
    save_path=save_path,
)


@torch.no_grad()
def calc_accuracy(pred_y, y) -> torch.Tensor:
    """Calculate accuracy."""
    return (pred_y == y).sum() / len(y)


@torch.no_grad()
def calc_precision_and_racall(pred_y, y):
    """Calculate precision."""
    mask1 = pred_y == 1
    mask2 = pred_y == y
    positive_pred = np.sum(mask1)
    positive_case = np.sum(y)
    tp = np.sum(mask2 & mask1)
    fp = positive_pred - tp
    fn = positive_case - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    # fp = ((not mask2) & mask1).sum()
    return precision, recall


@torch.no_grad()
def calc_precision(pred_y, y) -> torch.Tensor:
    """Calculate precision."""
    mask1 = pred_y == 1
    mask2 = pred_y == y
    positive_pred = mask1.sum()
    tp = (mask2 & mask1).sum()
    fp = positive_pred - tp
    precision = tp / (tp + fp + 1e-6)
    return precision


@torch.no_grad()
def calc_recall(pred_y, y) -> torch.Tensor:
    """Calculate recall."""
    tp = ((pred_y == y) & (pred_y == 1)).sum()
    fn = ((pred_y != y) & (pred_y == 0)).sum()
    return tp / (tp + fn + 1e-6)


@torch.no_grad()
def calc_f1_score(pred_y, y) -> torch.Tensor:
    """Calculate F1 score."""
    precision, recall = calc_precision_and_racall(pred_y, y)
    # recall = calc_recall(pred_y, y)
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def calculate_average_precision(y_pred, y_true) -> torch.Tensor:
    """
    Calculate the average precision for each label, then take the mean.
    Works for both binary (1D) and multi-label (2D) classification.

    """
    # Ensure the inputs are numpy arrays for compatibility with sklearn
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # Calculate average precision (works for both binary and multi-label)
    mean_ap = torch.tensor(average_precision_score(y_true_np, y_pred_np))
    return mean_ap


@torch.no_grad()
def calc_roc_auc(y_true, y_score) -> float:
    """
    Calculate ROC AUC score for binary classification.
    """
    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.detach().cpu().numpy()
    auc = roc_auc_score(y_true_np, y_score_np)
    return float(auc)


# FIXME: The function in the directed case just returns a repeated tensor of
# the node itself.
def find_neighbors_(
    node_id: int,
    edge_index: torch.Tensor,
    include_node=False,
    flow="undirected",  # could be "undirected", "source_to_target" or "target_to_source"
):
    if flow == "undirected":
        edge_mask = edge_index.unsqueeze(2).eq(node_id).any(2).any(0)
        edges = edge_index[:, edge_mask]
    elif flow == "source_to_target":
        edge_mask = edge_index[0].unsqueeze(1).eq(node_id).any(1)
        edges = edge_index[0, edge_mask]
    elif flow == "target_to_source":
        edge_mask = edge_index[1].unsqueeze(1).eq(node_id).any(1)
        edges = edge_index[1, edge_mask]

    all_neighbors = torch.unique(edges)
    if not include_node:
        mask = all_neighbors != node_id
        all_neighbors = all_neighbors[mask]

    return all_neighbors


def plot_metrics(
    res,
    title="",
    save_path="./",
):
    dataset = pd.DataFrame.from_dict(res)
    dataset.set_index("Epoch", inplace=True)

    os.makedirs(save_path, exist_ok=True)

    loss_columns = list(filter(lambda x: x.endswith("Loss"), dataset.columns))
    dataset[loss_columns].plot()
    plot_title = f"loss {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")

    acc_columns = list(filter(lambda x: x.endswith("Acc"), dataset.columns))
    dataset[acc_columns].plot()
    plot_title = f"accuracy {title}"
    plt.title(plot_title)
    plt.savefig(f"{save_path}{plot_title}.png")


def plot_spectral_hist(x, D, path_file="./"):
    if D is None:
        D = np.arange(x.shape[0])
    else:
        D = D.numpy()
    hist_x = np.sum(abs(x), axis=1)
    normalized_hist_x = hist_x / sum(hist_x)
    # sorted_x = np.sort(abs(x), axis=0)
    # ax = fig.add_subplot()
    fig, ax = plt.subplots()
    ax.plot(D, x, "*")
    fig.savefig(f"{path_file}x1.png")

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    X = range(x.shape[0])
    Y = range(x.shape[1])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(
        X,
        Y,
        x.T,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    fig.savefig(f"{path_file}x2.png")
    fig, ax = plt.subplots()
    ax.bar(D, normalized_hist_x)
    fig.savefig(f"{path_file}x3.png")
    fig, ax = plt.subplots()
    ax.bar(range(len(normalized_hist_x)), hist_x)
    fig.savefig(f"{path_file}x4.png")


def plot_TSNE(
    file_path, SFVs, edge_index, labels, num_classes, correctly_classified_list=None
):
    image_path = f"{file_path}points/"
    os.makedirs(image_path, exist_ok=True)
    graph_path = f"{file_path}graph/"
    os.makedirs(graph_path, exist_ok=True)
    cmap = plt.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(1.0 * i / num_classes) for i in range(num_classes)]
    tsne = TSNE(
        random_state=seed,
        perplexity=15,
        max_iter=500,
        n_iter_without_progress=100,
        n_jobs=7,
    )

    reference_tsne = None
    bar = tqdm(total=len(SFVs))
    for epoch, SFV in reversed(list(enumerate(SFVs))):
        points = tsne_and_align(tsne, SFV, reference_tsne)
        reference_tsne = points

        fig, ax = plt.subplots()
        for i in range(num_classes):
            mask = labels == i
            class_points = points[mask]
            ax.scatter(
                class_points[:, 0],
                class_points[:, 1],
                color=colors[i],
                marker=f"${i}$",
                label=i,
            )

        plot_title = f"epoch {epoch}"

        file_name = f"{image_path}{plot_title}.png"
        ax.legend(loc="lower right")
        plt.title(plot_title)
        fig.savefig(file_name)

        fig2, ax = plt.subplots()
        plot_graph(
            edge_index,
            SFV.shape[0],
            num_classes,
            labels,
            points,
            correctly_classified_list[epoch],
        )
        plt.title(plot_title)
        file_name2 = f"{graph_path}{plot_title}.png"
        fig2.savefig(file_name2)
        # plt.show()
        bar.update()

    generate_gif(file_path, name="points")
    generate_gif(file_path, name="graph")


def tsne_and_align(tsne: TSNE, new_matrix, reference_tsne):
    if reference_tsne is not None:
        tsne.set_params(init=reference_tsne)
    tsne_result = tsne.fit_transform(new_matrix)
    return tsne_result


def generate_gif(root_path, name="graph"):
    image_path = f"{root_path}{name}/"
    filenames = os.listdir(image_path)
    filenames = [filename for filename in filenames if filename.endswith(".png")]
    filenames = list(sorted(filenames, key=lambda elem: int(elem.split(".")[0][6:])))

    images = []
    for filename in filenames:
        file_path = f"{image_path}{filename}"
        images.append(imageio.imread(file_path))
    gif_path = f"{root_path}gif/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave(f"{gif_path}/{name}.gif", images, format="GIF", fps=5)


def create_adj(
    edge_index: torch.Tensor,
    normalization: str = "normal",
    self_loop: bool = False,
    num_nodes: int | None = None,
    nodes=None,
):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    if nodes is None:
        nodes = torch.arange(num_nodes, device=edge_index.device)
    if self_loop:
        undirected_edges = add_self_loops(edge_index)[0]
    else:
        undirected_edges = edge_index

    edge_mask = undirected_edges[0].unsqueeze(1).eq(nodes).any(1)
    directed_edges = undirected_edges[:, edge_mask]
    # directed_edges, _ = remove_self_loops(directed_edges)

    edge_weight = torch.ones(
        directed_edges.size(1), dtype=torch.float32, device=edge_index.device
    )
    row, col = directed_edges[0], directed_edges[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    if normalization == "rw":
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv = deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif normalization == "sym":
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        deg2 = scatter(edge_weight, col, 0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt2 = deg2.pow_(-0.5)
        deg_inv_sqrt2 = deg_inv_sqrt2.masked_fill_(deg_inv_sqrt2 == float("inf"), 0)
        edge_weight *= deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt2[edge_index[1]]
    if dev != "mps":
        adj = torch.sparse_coo_tensor(
            directed_edges,
            edge_weight,
            (num_nodes, num_nodes),
            dtype=torch.float32,
            device=edge_index.device,
        )

    return adj


def create_inc(
    edge_index: torch.Tensor,
    normalization: str = "normal",
    self_loop: bool = False,
    num_nodes: int | None = None,
    nodes=None,
):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    if nodes is None:
        nodes = torch.arange(num_nodes)
    if self_loop:
        undirected_edges = add_self_loops(edge_index)[0]
    else:
        undirected_edges = edge_index

    edge_mask = undirected_edges[0].unsqueeze(1).eq(nodes).any(1)
    directed_edges = undirected_edges[:, edge_mask]
    # directed_edges, _ = remove_self_loops(directed_edges)

    edge_weight1 = torch.ones(
        directed_edges.size(1), dtype=torch.float32, device=edge_index.device
    )
    edge_weight2 = -torch.ones(
        directed_edges.size(1), dtype=torch.float32, device=edge_index.device
    )
    row, col = directed_edges[0], directed_edges[1]
    deg1 = scatter(edge_weight1, row, 0, dim_size=num_nodes, reduce="sum")
    # deg2 = scatter(edge_weight1, col, 0, dim_size=num_nodes, reduce="sum")
    ind1 = torch.arange(row.shape[0], dtype=torch.int64, device=edge_index.device)
    ind2 = torch.arange(row.shape[0], dtype=torch.int64, device=edge_index.device)
    ind = torch.vstack((torch.hstack((row, col)), torch.hstack((ind1, ind2))))

    # if normalization == "rw":
    #     # Compute A_norm = -D^{-1} A.
    #     deg_inv = deg1.pow_(-0.5)
    #     deg_inv = deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    #     edge_weight1 *= deg_inv[row]
    #     edge_weight2 *= deg_inv[row]
    # elif normalization == "sym":
    #     # Compute A_norm = -D^{-1/2} A D^{-1/2}.
    #     deg_inv_sqrt1 = deg1.pow_(-0.5)
    #     deg_inv_sqrt1 = deg_inv_sqrt1.masked_fill_(deg_inv_sqrt1 == float("inf"), 0)
    #     deg_inv_sqrt2 = deg2.pow_(-0.5)
    #     deg_inv_sqrt2 = deg_inv_sqrt2.masked_fill_(deg_inv_sqrt2 == float("inf"), 0)
    #     edge_weight1 *= deg_inv_sqrt1[row]
    #     edge_weight2 *= deg_inv_sqrt2[row]

    edge_weight = torch.hstack((edge_weight1, edge_weight2))

    inc = torch.sparse_coo_tensor(
        ind,
        edge_weight,
        (num_nodes, row.shape[0]),
        dtype=torch.float32,
        device=edge_index.device,
    )

    if normalization == "sym":
        deg_inv = deg1.pow_(-0.5)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        D = sparse_eye(num_nodes, deg_inv, dev_=edge_index.device)
        inc = D @ inc

    return inc


def sparse_eye(n, vals=None, dev_="cpu"):
    if vals is None:
        vals = torch.ones(n, dtype=torch.float32, device=dev_)
    eye = torch.Tensor.repeat(torch.arange(n, device=dev_), [2, 1])
    sparse_matrix = torch.sparse_coo_tensor(eye, vals, (n, n), device=dev_)

    return sparse_matrix


def create_rw(edge_index, num_nodes, num_layers):
    local_dev = "cpu"

    abar = sparse_eye(num_nodes, dev_=local_dev)

    adj_hat = create_adj(
        edge_index, normalization="rw", self_loop=True, num_nodes=num_nodes
    ).to(local_dev)

    SE = []
    for i in range(num_layers):
        abar = torch.matmul(abar, adj_hat)  # Sparse-dense matrix multiplication
        SE.append(torch.diag(abar.to_dense()))

    SE_rw = torch.stack(SE, dim=-1)

    # if dev != "mps":
    SE_rw = SE_rw.to(dev)
    return SE_rw


def calc_abar(adj, num_layers, pruning=False):
    local_dev = "cpu"
    num_nodes = adj.shape[1]

    abar = sparse_eye(num_nodes, dev_=local_dev)
    adj = adj.to(local_dev)

    th = config.subgraph.pruning_th
    for i in range(num_layers):
        abar = torch.matmul(adj, abar)  # Sparse-dense matrix multiplication
        if pruning:
            abar = prune(abar, th)
        # print(f"{i}: {abar.values().shape[0]/(num_nodes*num_nodes)}")

    if dev != "mps":
        abar = abar.to(dev)
    return abar


def prune(abar, degree):
    num_nodes = abar.shape[0]
    num_vals = num_nodes * degree
    vals = abar.values()
    if num_vals >= vals.shape[0]:
        return abar
    sorted_vals_idx = torch.argsort(vals, descending=True)
    chosen_vals_idx = sorted_vals_idx[:num_vals]
    # chosen_vals_idx = np.random.choice(vals.shape[0], num_vals, replace=False)

    # mask = abar.values() > th
    idx = abar.indices()[:, chosen_vals_idx]
    val = abar.values()[chosen_vals_idx]
    # val = torch.masked_select(abar.values(), mask)

    abar = torch.sparse_coo_tensor(idx, val, abar.shape, device=abar.device)
    abar = abar.coalesce()

    return abar


def split_abar(abar: SparseTensor, nodes):
    num_nodes = abar.size()[0]
    # nodes = self.get_nodes().to(local_dev)
    num_nodes_i = len(nodes)
    indices = torch.arange(num_nodes_i, dtype=torch.long, device=local_dev)
    vals = torch.ones(num_nodes_i, dtype=torch.float32, device=local_dev)
    P = torch.sparse_coo_tensor(
        torch.vstack([indices, nodes]),
        vals,
        (num_nodes_i, num_nodes),
        device=local_dev,
    )
    abar_i = torch.matmul(P, abar)
    if dev != "cuda:0":
        abar_i = abar_i.to_dense().to(dev)
    return abar_i


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_abar(abar, edge_index, name="graph", save=False, power=1000, cmap="plasma"):
    dense_abar = abar.to_dense().numpy()
    dense_abar = (dense_abar - np.min(dense_abar)) / (
        np.max(dense_abar) - np.min(dense_abar)
    )
    dense_abar = sigmoid(power * (dense_abar))

    # dense_abar[dense_abar < 0.0001] = 0
    # dense_abar[dense_abar > 0.0001] = 1
    # dense_abar = np.log10(dense_abar)
    # dense_abar = np.power(dense_abar, power)
    dense_abar = (dense_abar - np.min(dense_abar)) / (
        np.max(dense_abar) - np.min(dense_abar)
    )

    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G)
    comminity_label = np.zeros(abar.shape[0], dtype=np.int32)
    for id, c in enumerate(community):
        for node in c:
            comminity_label[node] = id
    idx = np.argsort(comminity_label)

    sorted_community_groups = sorted(
        community, key=lambda item: len(item), reverse=True
    )
    community_based_node_order = list(
        itertools.chain.from_iterable(sorted_community_groups)
    )

    # dense_abar = dense_abar[:, idx]
    # dense_abar = dense_abar[idx, :]
    dense_abar = dense_abar[:, community_based_node_order]
    dense_abar = dense_abar[community_based_node_order, :]
    plt.imshow(dense_abar, cmap="gray", interpolation="none")
    # plt.imshow(dense_abar, cmap="gray", interpolation="nearest")
    if save:
        plt.imsave(f"./models/{name}.png", dense_abar, cmap=cmap)
    plt.grid(True, which="both", color="black", linestyle="-", linewidth=0.25)

    # Optionally, customize the grid to match the image resolution
    # ax = plt.gca()
    # plt.xticks(np.arange(0, abar.shape[1], 1))
    # plt.yticks(np.arange(0, abar.shape[0], 1))
    # plt.xticklabels([])
    # plt.set_yticklabels([])
    # plt.show()


def plot_abar2(abar, labels, name="graph", save=False):
    dense_abar = abar.to_dense().numpy()
    dense_abar = (dense_abar - np.min(dense_abar)) / (
        np.max(dense_abar) - np.min(dense_abar)
    )
    # dense_abar = np.log10(dense_abar)
    dense_abar = np.power(dense_abar, 0.25)
    dense_abar = (dense_abar - np.min(dense_abar)) / (
        np.max(dense_abar) - np.min(dense_abar)
    )
    idx = np.argsort(labels)

    dense_abar = dense_abar[:, idx]
    dense_abar = dense_abar[idx, :]

    plt.imshow(dense_abar, cmap="gray", interpolation="none")
    # plt.imshow(dense_abar, cmap="gray", interpolation="nearest")
    if save:
        plt.imsave(f"./models/{name}.png", dense_abar)
    plt.grid(True, which="both", color="black", linestyle="-", linewidth=0.25)

    # Optionally, customize the grid to match the image resolution
    # ax = plt.gca()
    # plt.xticks(np.arange(0, abar.shape[1], 1))
    # plt.yticks(np.arange(0, abar.shape[0], 1))
    # plt.xticklabels([])
    # plt.set_yticklabels([])
    plt.show()


def estimate_abar(edge_index, num_nodes, num_layers, num_expriments=100):
    neighbors = [
        find_neighbors_(node, edge_index, include_node=True)
        for node in range(num_nodes)
    ]
    abar = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for _ in tqdm(range(num_expriments)):
        for node in range(num_nodes):
            chosen_node = node
            for _ in range(num_layers):
                idx = np.random.randint(neighbors[chosen_node].shape[0])
                chosen_node = neighbors[chosen_node][idx]
            abar[node, chosen_node] += 1

    abar /= num_expriments
    abar = abar.to_sparse()

    return abar


def calc_metrics(
    y, y_pred, mask=None, loss_function="cross_entropy"
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if mask is None:
        y_masked = y
        y_pred_masked = y_pred
    else:
        y_masked = y[mask]
        y_pred_masked = y_pred[mask]

    if loss_function == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_function == "log_likelihood":
        criterion = torch.nn.NLLLoss()
    elif loss_function == "BCELoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_pred_masked, y_masked)
    if config.dataset.multi_label:
        acc = calculate_average_precision(y_pred_masked, y_masked)
    else:
        acc = calc_accuracy(y_pred_masked.argmax(dim=1), y_masked)
    try:
        f1_score_ = calc_f1_score(y_pred_masked.argmax(dim=1), y_masked)
    except:
        f1_score_ = torch.tensor(0.0)

    try:
        precission = calc_precision(y_pred_masked.argmax(dim=1), y_masked)
    except:
        precission = torch.tensor(0.0)

    try:
        recall = calc_recall(y_pred_masked.argmax(dim=1), y_masked)
    except:
        recall = torch.tensor(0.0)

    if loss_function == "BCELoss" and y_pred_masked.dim() == 1:
        # Binary classification with logits - convert to probabilities
        y_probs = torch.sigmoid(y_pred_masked)
    elif y_pred_masked.dim() == 1:
        # Already probabilities or single output
        y_probs = y_pred_masked
    else:
        # Multi-class or multi-label - use softmax for probabilities
        y_probs = torch.softmax(y_pred_masked, dim=-1)
        if y_probs.shape[1] == 2:
            # Binary classification with 2D output - take positive class
            y_probs = y_probs[:, 1]
    try:
        auc = torch.tensor(calc_roc_auc(y_masked, y_probs))
    except:
        auc = torch.tensor(0.0)

    try:
        ap = calculate_average_precision(y_probs, y_masked)
    except:
        ap = torch.tensor(0.0)

    return loss, acc, f1_score_, precission, recall, auc, ap


def lod2dol(list_of_dicts):
    # convert list of dicts to dict of lists
    if len(list_of_dicts) == 0:
        return {}
    keys = list_of_dicts[0].keys()
    dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}

    return dict_of_lists


def lol2lol(list_of_lists):
    # convert list of lists to list of lists
    if len(list_of_lists) == 0:
        return []
    keys = range(len(list_of_lists[0]))
    res = [[d[key] for d in list_of_lists] for key in keys]

    return res


def sum_lod(x: list, coef=None):
    if len(x) == 0:
        return x
    if coef is None:
        coef = len(x) * [1 / len(x)]
    if isinstance(x[0], dict):
        z = lod2dol(x)
        for key, val in z.items():
            z[key] = sum_lod(val, coef)
        return z
    elif isinstance(x[0], list):
        z = lol2lol(x)
        for key, val in enumerate(z):
            z[key] = sum_lod(val, coef)
        return z
    else:
        return sum([weight * val for weight, val in zip(coef, x)])
