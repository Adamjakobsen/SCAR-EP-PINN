# utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import random
import os
import h5py
import numpy as np
import torch
import yaml  # pip install pyyaml


# ---- YAML-only IO -----------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(f"Config must be YAML (.yml/.yaml). Got: {p.name}")
    cfg = yaml.safe_load(p.read_text())
    if not isinstance(cfg, dict):
        raise TypeError("Top-level YAML must be a mapping (dict).")
    return cfg


def save_config_yaml(cfg: Dict[str, Any], exp_dir: str | Path, filename: str = "config.yaml") -> Path:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    out = exp_dir / filename
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# ---- Repro & device helpers -------------------------------------------------

def set_seed(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("PROJECT", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict[str, Any]) -> torch.device:
    order = cfg.get("DEVICE", {}).get("prefer", ["cuda", "mps", "cpu"])
    for name in order:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def make_experiment_dir(cfg: Dict[str, Any]) -> Path:
    """
    Creates a unique run directory under PROJECT.path_directory (e.g., ./experiments/refactored_run).
    If PROJECT.path_directory already points to a specific run folder you manage elsewhere,
    this function will simply ensure it exists.
    """
    base = Path(cfg["PROJECT"]["path_directory"])
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_mesh_data():
    # Provide your custom HDF5 read logic or geometry load logic here.
    # This example is just a placeholder.
    # Return (coordinates, triangles, vm):
    with h5py.File("./meshes/scarmesh_1.h5", "r") as scar_file:
        coordinates = scar_file["coordinates"][:]
        triangles = scar_file["topology"][:]
    vm = np.zeros((1, len(coordinates)))  # dummy
    return coordinates, triangles, vm

def get_boundary():
    # Provide logic to load your boundary mesh
    with h5py.File("./meshes/boundary_mesh.h5", "r") as boundary_file:
        group1 = boundary_file["Mesh"]
        group2 = group1["mesh"]
        coordinates = group2["geometry"][:]
        triangles = group2["topology"][:]
    return coordinates, triangles