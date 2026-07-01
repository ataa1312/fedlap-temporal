import math
import torch
import pytest
from src.metrics.classification import binary_classification_metrics
from src.metrics.mrr import compute_mrr_from_z
from src.utils.graph import Graph
from src.GNN.dynamic_classifier import DynamicClassifier

def test_binary_classification_metrics_perfect():
    logits = torch.tensor([2.0, -1.0, 3.0, -2.0])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["ap"] == 1.0

def test_binary_classification_metrics_mixed():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    
    assert math.isclose(metrics["roc_auc"], 0.75, rel_tol=1e-5)
    assert math.isclose(metrics["ap"], 5/6, rel_tol=1e-5)

def test_binary_classification_metrics_single_class():
    logits1 = torch.tensor([1.0, 2.0, 3.0])
    labels1 = torch.tensor([0, 0, 0])
    metrics1 = binary_classification_metrics(logits1, labels1)
    assert math.isnan(metrics1["roc_auc"])
    assert math.isnan(metrics1["ap"])
    
    logits2 = torch.tensor([1.0, 2.0, 3.0])
    labels2 = torch.tensor([1, 1, 1])
    metrics2 = binary_classification_metrics(logits2, labels2)
    assert math.isnan(metrics2["roc_auc"])
    assert metrics2["ap"] == 1.0

def test_compute_mrr_from_z(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 0
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    
    N = 5
    edge_label_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    edge_label = torch.tensor([1.0, 1.0])
    
    snap = Graph(x=torch.ones(N, 1),
                 edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
                 node_ids=torch.arange(N))
    snap.edge_label_index = edge_label_index
    snap.edge_label = edge_label
    
    dyn = DynamicClassifier(snap)
    dyn.eval()
    
    z = torch.randn(N, 16)
    torch.manual_seed(42)
    mrr = compute_mrr_from_z(z, snap, K=2, method="min", device="cpu", model=dyn)
    assert 0.0 <= mrr <= 1.0
    
    torch.manual_seed(42)
    mrr_dup = compute_mrr_from_z(z, snap, K=2, method="min", device="cpu", model=dyn)
    assert mrr == mrr_dup
