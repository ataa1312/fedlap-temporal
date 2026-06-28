import pytest
import importlib

def test_import_src():
    import src

@pytest.mark.parametrize("module_name", [
    "src.utils.graph",
    "src.utils.graph_partitioning",
    "src.GNN.GNN_models",
    "src.MLP.MLP_model",
    "src.models.model_binders",
    "src.GNN.laplace",
    "src.GNN.sGNN",
    "src.classifier",
])
def test_import_module(module_name):
    import src
    importlib.import_module(module_name)
