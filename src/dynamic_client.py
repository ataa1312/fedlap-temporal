from src import *
from src.client import Client
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import make_fed_dynamic_classifier
from src.train.federated_orchestrator import (
    _make_optimizer,
    _make_scheduler,
    _step_train_pair,
    _step_eval_loss_pair,
    _refresh_hs,
    _pos_for_split,
    _attach_future_link_pred_labels,
    _clone_state,
)


class DynamicClient(Client):
    """Federated client holding a per-client ROLAND DynamicClassifier + its
    subgraph snapshot sequence + carried hidden state. Reuses the base Client
    federated protocol (state_dict/load_state_dict/num_nodes via the Classifier)
    and adds the temporal local-training / encode / refresh steps."""

    def __init__(self, snaps, id: int = 0):
        super().__init__(graph=snaps[0], id=id)
        self.snaps = snaps
        self.hs = None
        self._val_cache = None
        self._val_cache_t = None

    def initialize(
        self,
        smodel_type=None,
        fmodel_type=None,
        data_type=None,
        **kwargs,
    ) -> None:
        data_type = config["model"]["data_type"] if data_type is None else data_type
        smodel_type = config["model"]["smodel_type"] if smodel_type is None else smodel_type
        if data_type == "feature":
            self.classifier = DynamicClassifier(self.snaps[0])
        elif data_type == "f+s":
            SFV = kwargs.get("SFV")
            if SFV is None:
                raise ValueError("f+s initialization needs the server-shared SFV")
            self.classifier = make_fed_dynamic_classifier(smodel_type, self.snaps[0], SFV)
        else:
            raise NotImplementedError(
                f"data_type={data_type!r}: structure-only needs an smodel-only subclass (deferred)"
            )

    def _hs_in(self):
        return [h.detach() for h in self.hs] if self.hs is not None else None

    def _val_batch(self, t: int):
        # Frozen val batch for snapshot t: built once (fixed positives + negatives)
        # and reused across all local epochs AND FedAvg rounds, so both the local
        # (local_finetune) and round-level (val_loss) early stops compare val loss
        # under a fixed batch — patience reacts to weights, not resampled negatives.
        if self._val_cache_t != t:
            today, tomorrow = self.snaps[t].to(device), self.snaps[t + 1].to(device)
            self._val_cache = _attach_future_link_pred_labels(
                today, tomorrow, _pos_for_split(tomorrow, "val").to(device)
            )
            self._val_cache_t = t
        return self._val_cache

    def can_train(self, t: int) -> bool:
        # Can fine-tune at t only if snap_{t+1} has both train and val positives
        # after the per-client edge split (tiny subgraphs otherwise abstain).
        tomorrow = self.snaps[t + 1]
        return (
            _pos_for_split(tomorrow, "train").size(1) > 0
            and _pos_for_split(tomorrow, "val").size(1) > 0
        )

    def encode(self, t: int):
        # Encode snap_t with the current (global) weights + carried hs; returns
        # (z_local, global_node_ids) for the server's global-z stitch.
        self.classifier.eval()
        snap = self.snaps[t].to(device)
        with torch.no_grad():
            z, _ = self.classifier.encode(snap, self._hs_in())
        return z, snap.node_ids.to(device)

    def local_finetune(self, t: int, local_epochs: int, loss_fn):
        # up to local_epochs local ROLAND train steps on (snap_t, snap_{t+1}) from the
        # currently-loaded (global) weights, with ROLAND internal-validation early
        # stopping: patience (internal_validation_tolerance) on a FROZEN val batch,
        # restore best. Train negatives resample each step (ROLAND train_step); the
        # val batch is fixed so patience reacts to weights, not sampling noise. This
        # makes local_epochs a MAX (not a fixed count) so large values no longer
        # overfit. fresh optimizer + scheduler per call (ROLAND drops both per snapshot).
        today, tomorrow = self.snaps[t].to(device), self.snaps[t + 1].to(device)
        hs_in = self._hs_in()
        optimizer = _make_optimizer(self.classifier)
        scheduler = _make_scheduler(optimizer)
        tol = config["train"]["internal_validation_tolerance"]
        val_snap = self._val_batch(t)
        best = {"val": float("inf"), "state": None}
        stale = 0
        for _ in range(local_epochs):
            _step_train_pair(
                self.classifier, today, tomorrow, hs_in, loss_fn, optimizer, device, True
            )
            if scheduler is not None:
                scheduler.step()
            vloss = _step_eval_loss_pair(
                self.classifier, today, tomorrow, hs_in, loss_fn, device, True,
                prepared_snap=val_snap,
            )
            if vloss < best["val"]:
                best = {"val": vloss, "state": _clone_state(self.classifier.state_dict())}
                stale = 0
            else:
                stale += 1
            if stale >= tol:
                break
        if best["state"] is not None:
            self.classifier.load_state_dict(best["state"])

    def val_loss(self, t: int, loss_fn) -> float:
        return _step_eval_loss_pair(
            self.classifier, self.snaps[t], self.snaps[t + 1], self._hs_in(),
            loss_fn, device, True, prepared_snap=self._val_batch(t),
        )

    def refresh(self, t: int):
        self.hs = _refresh_hs(self.classifier, self.snaps[t], self._hs_in(), device, True)
