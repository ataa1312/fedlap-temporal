"""wandb run tagging (commit 680688a) -- pure-function tests on main._wandb_meta.

The group string is what collapses multi-seed runs into one averaged cell in
the wandb UI, so a missing condition axis silently merges a treatment with its
placebo. These pin the f+pe axes and the sweep-tagging knobs.
"""

import copy

import pytest

import src
import main


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def set_pe_run(config, basis_source="laplacian", update_mode="keep", pe_dim=50):
    config["dataset"]["name"] = "uci"
    config["gnn"]["dims"] = [64, 64]
    config["model"]["data_type"] = "f+pe"
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["basis_source"] = basis_source
    config["spectral"]["pe_dim"] = pe_dim


def test_fpe_group_carries_the_condition_axes(config):
    set_pe_run(config, basis_source="laplacian", update_mode="keep", pe_dim=50)

    group, _, tags = main._wandb_meta()

    assert "um-keep" in group.split("_")
    assert "pe50" in group.split("_")
    assert "basis-laplacian" in group.split("_")
    assert "um-keep" in tags
    assert "basis-laplacian" in tags


def test_basis_source_separates_the_treatment_from_its_placebo(config):
    set_pe_run(config, basis_source="laplacian")
    real, _, _ = main._wandb_meta()

    set_pe_run(config, basis_source="random_fixed")
    placebo, _, _ = main._wandb_meta()

    assert real != placebo


def test_group_suffix_and_extra_tags_default_to_no_ops(config):
    set_pe_run(config)
    assert config["wandb"]["group_suffix"] == ""
    assert config["wandb"]["extra_tags"] == []

    base_group, _, base_tags = main._wandb_meta()

    config["wandb"]["group_suffix"] = "abl1"
    config["wandb"]["extra_tags"] = ["depth-abl", "L4"]
    group, _, tags = main._wandb_meta()

    assert group == f"{base_group}_abl1"
    assert tags == base_tags + ["depth-abl", "L4"]


def test_cfg_payload_carries_dims_and_pe_axes(config):
    set_pe_run(config, basis_source="shuffled", pe_dim=32)
    config["gnn"]["dims"] = [128, 64]

    _, cfg, _ = main._wandb_meta()

    assert cfg["gnn_dims"] == [128, 64]
    assert cfg["pe_dim"] == 32
    assert cfg["basis_source"] == "shuffled"
