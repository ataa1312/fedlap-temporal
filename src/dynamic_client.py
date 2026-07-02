from src import *
from src.client import Client
from src.GNN.dynamic_classifier import DynamicClassifier
from src.train.federated_orchestrator import (
    _make_optimizer,
    _step_train_pair,
    _step_eval_loss_pair,
    _refresh_hs,
    _pos_for_split,
)


class DynamicClient(Client):
    """Federated client holding a per-client ROLAND DynamicClassifier + its
    subgraph snapshot sequence + carried hidden state. Reuses the base Client
    federated protocol (state_dict/load_state_dict/num_nodes via the Classifier)
    and adds the temporal local-training / encode / refresh steps."""

    def __init__(self, snaps, id: int = 0):
        super().__init__(graph=snaps[0], id=id)
        self.snaps = snaps
        self.classifier = DynamicClassifier(snaps[0])
        self.hs = None

    def _hs_in(self):
        return [h.detach() for h in self.hs] if self.hs is not None else None

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
        # local_epochs local ROLAND train steps on (snap_t, snap_{t+1}) from the
        # currently-loaded (global) weights; fresh optimizer per call.
        today, tomorrow = self.snaps[t].to(device), self.snaps[t + 1].to(device)
        hs_in = self._hs_in()
        optimizer = _make_optimizer(self.classifier)
        for _ in range(local_epochs):
            _step_train_pair(
                self.classifier, today, tomorrow, hs_in, loss_fn, optimizer, device, True
            )

    def val_loss(self, t: int, loss_fn) -> float:
        return _step_eval_loss_pair(
            self.classifier, self.snaps[t], self.snaps[t + 1], self._hs_in(),
            loss_fn, device, True, pos_split="val",
        )

    def refresh(self, t: int):
        self.hs = _refresh_hs(self.classifier, self.snaps[t], self._hs_in(), device, True)
