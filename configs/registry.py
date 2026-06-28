import os
import json
from typing import Any

import yaml


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._registry = dict()

    def register(self, key: str, *, eager: bool = True):
        def decorator(cls_or_func):
            if key in self._registry:
                raise KeyError(f"Duplicate key in {self.name}: {key}")
            self._registry[key] = cls_or_func() if eager else cls_or_func
            return cls_or_func

        return decorator

    def unregister(self, key: str):
        if key not in self._registry:
            raise KeyError(f"Key not found in {self.name}: {key}")
        self._registry.pop(key)

    def __getitem__(self, key: str):
        if key not in self._registry:
            raise KeyError(f"Key not found in {self.name}: {key}")
        return self._registry[key]

    def __setitem__(self, key: str, val: Any) -> None:
        self._registry[key] = val

    def __delitem__(self, key: str) -> None:
        if key not in self._registry:
            raise KeyError(f"Key not found in {self.name}: {key}")
        del self._registry[key]

    def __contains__(self, key: str):
        return key in self._registry

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self):
        return f"Registry({self.name!r}, {self._registry!r})"

    def set_path(self, dot_path: str, value: Any) -> None:
        parts = dot_path.split(".")
        node = self
        for part in parts[:-1]:
            node = node[part]  # raises KeyError on typo

        leaf = parts[-1]
        if leaf not in node:
            raise KeyError(
                f"Unknown config key: {dot_path!r} (leaf {leaf!r} not in {node.name!r})"
            )
        node[leaf] = value

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in self._registry.items():
            if isinstance(val, Registry):
                out[key] = val.to_dict()
            elif isinstance(val, (tuple, set)):
                out[key] = list(val)
            else:
                out[key] = val

        return out

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Registry":
        reg = cls(name)
        for key, val in data.items():
            reg[key] = cls.from_dict(key, val) if isinstance(val, dict) else val
        return reg

    def to_yaml(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, name: str, path: str) -> "Registry":
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            raise yaml.YAMLError(f"Empty file: {path}")
        return cls.from_dict(name, data)

    def to_json(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, name: str, path: str) -> "Registry":
        with open(path) as f:
            data = json.load(f)
        if data is None:
            raise ValueError(f"Empty/null JSON: {path}")
        return cls.from_dict(name, data)
