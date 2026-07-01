import torch
from sklearn import model_selection
import torch.utils
import torch.utils.data

from src import *


class Data:
    def __init__(
        self,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        node_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.x = x
        self.y = y
        self.node_ids = node_ids
        self.num_nodes = node_ids.shape[0]
        self.num_features = x.shape[1]

        self.train_mask = kwargs.get("train_mask", None)
        self.test_mask = kwargs.get("test_mask", None)
        self.val_mask = kwargs.get("val_mask", None)
        self.num_classes = kwargs.get("num_classes", None)

    def clone(self):
        # Shallow-copy the instance, deep-copying only tensor attributes so
        # callers can rebind extra attrs (edge_label_index, ...) without
        # touching the original. Non-tensors (node_map, ints) are shared by
        # reference — the orchestrator helpers never mutate them in place.
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in self.__dict__.items()
        }
        return new

    def to(self, device):
        # Move dense tensor attributes to `device` in place; return self
        # (matches torch_geometric's Data.to semantics the helpers expect).
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                self.__dict__[k] = v.to(device)
        return self

    def get_masks(self):
        return (self.train_mask, self.val_mask, self.test_mask)

    def set_masks(self, masks):
        self.train_mask = masks[0]
        self.val_mask = masks[1]
        self.test_mask = masks[2]

    def add_masks(self, train_ratio=0.5, test_ratio=0.2):
        num_nodes = self.num_nodes

        self.train_mask, self.val_mask, self.test_mask = Data.split_indices(
            num_nodes, train_ratio, test_ratio
        )

    def split_indices(n, train_ratio=0.5, test_ratio=0.2):
        indices = torch.arange(n, device=dev)
        train_size = int(train_ratio * n)
        test_size = int(test_ratio * n)
        val_size = n - train_size - test_size

        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            indices,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

        train_mask = indices.unsqueeze(1).eq(torch.tensor(train_indices).to(dev)).any(1)
        val_mask = indices.unsqueeze(1).eq(torch.tensor(val_indices).to(dev)).any(1)
        test_mask = indices.unsqueeze(1).eq(torch.tensor(test_indices).to(dev)).any(1)
        # self.val_mask = ~(self.test_mask | self.train_mask)

        return train_mask, val_mask, test_mask
