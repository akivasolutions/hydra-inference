"""Cluster configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cluster.yaml"


@dataclass
class GPU:
    name: str
    vram_gb: int
    rpc_port: int | None = None  # None for local coordinator GPUs


@dataclass
class Worker:
    host: str
    gpus: list[GPU] = field(default_factory=list)

    @property
    def rpc_addresses(self) -> list[str]:
        return [f"{self.host}:{gpu.rpc_port}" for gpu in self.gpus]


@dataclass
class ModelConfig:
    name: str
    path: str
    ctx_size: int = 8192
    predict: int = 4096
    flash_attn: bool = True
    default: bool = False


@dataclass
class ClusterConfig:
    coordinator_host: str
    coordinator_port: int
    coordinator_backend: str
    coordinator_gpus: list[GPU]
    workers: list[Worker]
    models: dict[str, ModelConfig]
    coordinator_binary: str
    rpc_server_binary: str

    @property
    def all_gpus(self) -> list[GPU]:
        gpus = list(self.coordinator_gpus)
        for w in self.workers:
            gpus.extend(w.gpus)
        return gpus

    @property
    def total_vram_gb(self) -> int:
        return sum(g.vram_gb for g in self.all_gpus)

    @property
    def rpc_addresses(self) -> list[str]:
        addrs: list[str] = []
        for w in self.workers:
            addrs.extend(w.rpc_addresses)
        return addrs

    def tensor_split(self) -> list[float]:
        """Calculate --tensor-split proportions from VRAM sizes."""
        gpus = self.all_gpus
        total = sum(g.vram_gb for g in gpus)
        return [round(g.vram_gb / total, 2) for g in gpus]

    def default_model(self) -> ModelConfig | None:
        for m in self.models.values():
            if m.default:
                return m
        # Return first model if none marked default
        return next(iter(self.models.values()), None)


def load_config(path: str | Path | None = None) -> ClusterConfig:
    """Load cluster config from YAML file."""
    config_path = Path(path) if path else Path(
        os.environ.get("HYDRA_CONFIG", DEFAULT_CONFIG)
    )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    coord = raw["coordinator"]
    coordinator_gpus = [
        GPU(name=g["name"], vram_gb=g["vram_gb"])
        for g in coord.get("gpus", [])
    ]

    workers = []
    for w in raw.get("workers", []):
        gpus = [
            GPU(name=g["name"], vram_gb=g["vram_gb"], rpc_port=g["rpc_port"])
            for g in w.get("gpus", [])
        ]
        workers.append(Worker(host=w["host"], gpus=gpus))

    models = {}
    for name, m in raw.get("models", {}).items():
        models[name] = ModelConfig(
            name=name,
            path=m["path"],
            ctx_size=m.get("ctx_size", 8192),
            predict=m.get("predict", 4096),
            flash_attn=m.get("flash_attn", True),
            default=m.get("default", False),
        )

    binaries = raw.get("binaries", {})

    return ClusterConfig(
        coordinator_host=coord.get("host", "0.0.0.0"),
        coordinator_port=coord.get("port", 8080),
        coordinator_backend=coord.get("backend", "hip"),
        coordinator_gpus=coordinator_gpus,
        workers=workers,
        models=models,
        coordinator_binary=binaries.get("coordinator", "llama-server"),
        rpc_server_binary=binaries.get("rpc_server", "rpc-server"),
    )
