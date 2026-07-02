import math
import statistics

from src import *
from src.server import Server
from src.utils.graph import Graph
from src.dynamic_client import DynamicClient
from src.GNN.dynamic_classifier import DynamicClassifier
from src.metrics.mrr import compute_mrr_from_z
from src.train.federated_orchestrator import (
    _partition_edges_per_snapshot,
    _precompute_keep_ratio,
    _pos_for_split,
    _attach_future_link_pred_labels,
    _step_eval_with_mrr_pair,
    _stitch_global_z,
    _clone_state,
)
from registries import losses


class DynamicServer(Server):
    """Federated ROLAND live-update over client subgraphs, reusing FedLap's Server
    FedAvg primitives (share_weights / get_weights / sum_lod / node-count coef).

    Mirrors the GNNServer surface: add_client / initialize / initialize_FL /
    joint_train_w. Per global snapshot pair (t, t+1): reported eval, then `epochs`
    FedAvg rounds of `model.local_epochs` local ROLAND steps, weight-averaged via
    sum_lod with round-level early stopping on the aggregated client val loss,
    then a per-client hidden-state refresh. Inactive clients (a subgraph too small
    to form a train/val split) abstain from the aggregation. FL=False = local-only
    baseline: no broadcast, no aggregation; the reported eval becomes the
    node-count-weighted mean of per-client local MRRs.
    """

    def __init__(self, global_snaps):
        gsnap0 = global_snaps[0]
        global_graph = Graph(
            x=gsnap0.x,
            edge_index=gsnap0.edge_index,
            edge_attr=gsnap0.edge_attr,
            node_ids=torch.arange(gsnap0.num_nodes, device=device),
        )
        super().__init__(graph=global_graph)
        self.global_snaps = global_snaps
        self.clients = list[DynamicClient]()

    def add_client(self, snaps):
        client = DynamicClient(snaps, id=self.num_clients)
        self.clients.append(client)
        self.num_clients += 1

    def initialize(
        self,
        smodel_type=None,
        fmodel_type=None,
        data_type=None,
        **kwargs,
    ):
        data_type = config["model"]["data_type"] if data_type is None else data_type
        if data_type == "feature":
            self.classifier = DynamicClassifier(self.graph)  # global model (decode in eval)
        else:
            raise NotImplementedError(
                f"data_type={data_type!r} needs the spectral smodel subclasses (W7)"
            )
        self.classifier.eval()
        share = {}
        return share

    def initialize_FL(
        self,
        smodel_type=None,
        fmodel_type=None,
        data_type=None,
        **kwargs,
    ) -> None:
        share = self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        kwargs.update(share)
        for client in self.clients:
            client.initialize(
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                data_type=data_type,
                **kwargs,
            )
        self.share_weights()  # sync all clients to the server init

    def _coef(self, clients):
        total = sum(c.num_nodes() for c in clients)
        return [c.num_nodes() / total for c in clients]

    def joint_train_g(self, *args, **kwargs):
        raise NotImplementedError(
            "gradient averaging is unused in the dynamic path; use joint_train_w"
        )

    def joint_train_w(
        self,
        epochs=None,
        smodel_type=None,
        fmodel_type=None,
        FL=True,
        data_type=None,
        log=True,
        plot=False,
        model_type="",
        **kwargs,
    ):
        # epochs = FedAvg rounds per snapshot; None -> config at call time (the
        # YAML overlay lands after import, so no def-time default binding).
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )
        rounds = config["model"]["iterations"] if epochs is None else epochs
        local_epochs = config["model"]["local_epochs"]
        tol = config["train"]["internal_validation_tolerance"]
        embed = config["gnn"]["embed_update_method"]
        split = config["dataset"]["split"]
        ds = config["dataset"]
        seed = (
            ds["split_seed"]
            if ("split_seed" in ds and ds["split_seed"] is not None)
            else config["seed"]
        )
        loss_fn = losses[config["model"]["loss_fun"]]
        mrr_k = config["experimental"]["rank_eval_multiplier"]
        mrr_method = config["metric"]["mrr_method"]
        if log:
            tag = model_type or ("FL WA" if FL else "Local WA")
            LOGGER.info(f"{tag} live-update starts!")

        # Global edge split -> pos_test for the reported eval; per-client splits ->
        # pos_train/pos_val for local fine-tune (seed offset per client).
        _partition_edges_per_snapshot(self.global_snaps, split, seed)
        for c, cl in enumerate(self.clients):
            _partition_edges_per_snapshot(cl.snaps, split, seed + 1000 * (c + 1))
        if embed in {"moving_average", "masked_gru"}:
            km = config["gnn"]["keep_ratio_mode"]
            _precompute_keep_ratio(self.global_snaps, km)
            for cl in self.clients:
                _precompute_keep_ratio(cl.snaps, km)

        n_tasks = len(self.global_snaps) - 1
        if n_tasks < 1:
            raise ValueError(
                f"federated live_update needs >= 2 snapshots; got {len(self.global_snaps)}"
            )

        mrr_history: list[float] = []
        for t in range(n_tasks):
            # 1. Reported eval. FL: clients hold the global weights (share_weights
            #    at initialize_FL / end of the previous snapshot) -> global stitch.
            #    Local-only: weighted mean of per-client local MRRs.
            if FL:
                mrr = self._eval_mrr(t, mrr_k, mrr_method)
            else:
                mrr = self._eval_mrr_local(t, loss_fn, mrr_k, mrr_method)
            if mrr is not None and not math.isnan(mrr):
                mrr_history.append(mrr)
                if log:
                    LOGGER.info(f"t={t} mrr={mrr:.4f}")

            # 2. Training. FL: `rounds` communication rounds of `local_epochs`
            #    local steps, weight-averaged via sum_lod, with round-level early
            #    stopping on the aggregated (node-count-weighted) client val loss.
            #    Local-only: the same step budget without broadcast/aggregation.
            if FL:
                best = {"val": float("inf"), "state": None}
                stale = 0
                for _ in range(rounds):
                    self.share_weights()  # broadcast global -> clients
                    parts = [cl for cl in self.clients if cl.can_train(t)]  # abstention
                    if not parts:
                        break
                    for cl in parts:
                        cl.local_finetune(t, local_epochs, loss_fn)
                    coef = self._coef(parts)
                    self.load_state_dict(sum_lod([cl.state_dict() for cl in parts], coef))
                    self.share_weights()  # push the aggregate so clients report val on it
                    vloss = sum_lod([cl.val_loss(t, loss_fn) for cl in parts], coef)
                    if vloss < best["val"]:
                        best = {"val": vloss, "state": _clone_state(self.state_dict())}
                        stale = 0
                    else:
                        stale += 1
                    if stale >= tol:
                        break
                if best["state"] is not None:
                    self.load_state_dict(best["state"])
                self.share_weights()  # sync clients to the best global weights
            else:
                for _ in range(rounds):
                    for cl in self.clients:
                        if cl.can_train(t):
                            cl.local_finetune(t, local_epochs, loss_fn)

            # 3. Refresh each client's carried hidden state (from the aggregate
            #    when FL, from its own weights when local-only).
            for cl in self.clients:
                cl.refresh(t)

        mean = statistics.fmean(mrr_history) if mrr_history else None
        std = statistics.pstdev(mrr_history) if len(mrr_history) > 1 else 0.0
        if log:
            LOGGER.info(
                f"live-update done: mean_mrr={mean} std={std:.4f} "
                f"over {len(mrr_history)} snapshots"
            )
        return {"mean_mrr": mean, "std_mrr": std, "mrr_history": mrr_history}

    def _eval_mrr(self, t, mrr_k, mrr_method):
        zs, ids = [], []
        for cl in self.clients:
            z, nid = cl.encode(t)
            zs.append(z)
            ids.append(nid)
        dim = zs[0].shape[1]
        gz = _stitch_global_z(zs, ids, self.global_snaps[0].num_nodes, dim, device)
        pos_test = _pos_for_split(self.global_snaps[t + 1], "test").to(device)
        if pos_test.size(1) == 0:
            return None
        eval_snap = _attach_future_link_pred_labels(
            self.global_snaps[t].to(device), self.global_snaps[t + 1].to(device), pos_test
        )
        return compute_mrr_from_z(gz, eval_snap, mrr_k, mrr_method, device, self.classifier)

    def _eval_mrr_local(self, t, loss_fn, mrr_k, mrr_method):
        vals, weights = [], []
        for cl in self.clients:
            if _pos_for_split(cl.snaps[t + 1], "test").size(1) == 0:
                continue
            _, mrr, _ = _step_eval_with_mrr_pair(
                cl.classifier, cl.snaps[t], cl.snaps[t + 1], cl._hs_in(),
                loss_fn, device, True, mrr_k, mrr_method,
            )
            if not math.isnan(mrr):
                vals.append(mrr)
                weights.append(cl.num_nodes())
        if not vals:
            return None
        return sum(v * w for v, w in zip(vals, weights)) / sum(weights)
