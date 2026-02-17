"""Tests for cluster config loading."""

import textwrap
from pathlib import Path

import pytest
import yaml

from hydra.config import ClusterConfig, load_config


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [
                {"name": "XTX 0", "vram_gb": 24},
                {"name": "XTX 1", "vram_gb": 24},
            ],
        },
        "workers": [
            {
                "host": "192.168.1.100",
                "gpus": [
                    {"name": "4070", "vram_gb": 16, "rpc_port": 50052},
                    {"name": "3060", "vram_gb": 12, "rpc_port": 50053},
                ],
            }
        ],
        "models": {
            "test-model": {
                "path": "/models/test.gguf",
                "ctx_size": 4096,
                "default": True,
            }
        },
        "binaries": {
            "coordinator": "/usr/local/bin/llama-server",
            "rpc_server": "rpc-server.exe",
        },
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_config(config_file):
    config = load_config(config_file)
    assert config.coordinator_port == 8080
    assert config.coordinator_backend == "hip"
    assert len(config.coordinator_gpus) == 2
    assert len(config.workers) == 1
    assert len(config.workers[0].gpus) == 2


def test_total_vram(config_file):
    config = load_config(config_file)
    assert config.total_vram_gb == 76  # 24+24+16+12


def test_tensor_split(config_file):
    config = load_config(config_file)
    split = config.tensor_split()
    assert len(split) == 4
    assert sum(split) == pytest.approx(1.0, abs=0.05)
    # Largest GPUs get largest share
    assert split[0] == split[1]  # Two XTXs equal
    assert split[0] > split[2]  # XTX > 4070
    assert split[2] > split[3]  # 4070 > 3060


def test_rpc_addresses(config_file):
    config = load_config(config_file)
    addrs = config.rpc_addresses
    assert addrs == ["192.168.1.100:50052", "192.168.1.100:50053"]


def test_default_model(config_file):
    config = load_config(config_file)
    model = config.default_model()
    assert model is not None
    assert model.name == "test-model"
    assert model.default is True
