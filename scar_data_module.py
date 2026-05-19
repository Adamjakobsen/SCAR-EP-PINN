from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_mesh_data, get_boundary
from utils import get_mesh_data

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Flat-index helpers
# -----------------------------
def unravel_flat_index(flat_idx: int, n_space: int, n_time: int) -> tuple[int, int, int]:
    block = n_space * n_time
    exp_idx = flat_idx // block
    rem = flat_idx % block
    t_idx = rem // n_space
    x_idx = rem % n_space
    return x_idx, t_idx, exp_idx


def unravel_flat_indices(flat_idx: np.ndarray, n_space: int, n_time: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    block = n_space * n_time
    exp_idx = flat_idx // block
    rem = flat_idx % block
    t_idx = rem // n_space
    x_idx = rem % n_space
    return x_idx, t_idx, exp_idx


class ExcludingFlatIndexMap:
    """
    Maps dataset indices [0, ..., total_size - n_excluded) to flat indices
    [0, ..., total_size), skipping excluded indices without materializing
    the full training-index array.
    """
    def __init__(self, total_size: int, excluded_flat_indices: Optional[np.ndarray] = None):
        excluded = np.array([], dtype=np.int64)
        if excluded_flat_indices is not None and len(excluded_flat_indices) > 0:
            excluded = np.sort(np.asarray(excluded_flat_indices, dtype=np.int64))

        self.total_size = int(total_size)
        self.excluded = excluded
        self.adjusted = self.excluded - np.arange(self.excluded.size, dtype=np.int64)
        self.size = self.total_size - self.excluded.size

    def __len__(self) -> int:
        return self.size

    def to_flat(self, idx: int) -> int:
        offset = np.searchsorted(self.adjusted, idx, side="right")
        return idx + int(offset)


def build_feature_block(
    flat_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    g_values: np.ndarray,
    vmark: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_idx, t_idx, exp_idx = unravel_flat_indices(flat_idx, n_space=x.shape[0], n_time=t.shape[0])

    X = np.empty((flat_idx.shape[0], 5), dtype=np.float32)
    X[:, 0] = x[x_idx]
    X[:, 1] = y[x_idx]
    X[:, 2] = t[t_idx]
    X[:, 3] = g_values[exp_idx]
    X[:, 4] = vmark[x_idx]
    return X, x_idx, t_idx, exp_idx


def build_dense_batch(
    flat_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    g_values: np.ndarray,
    vmark: np.ndarray,
    vm: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, x_idx, t_idx, exp_idx = build_feature_block(flat_idx, x, y, t, g_values, vmark)
    V = vm[t_idx, x_idx, exp_idx].astype(np.float32, copy=False).reshape(-1, 1)
    Wb = W[t_idx, x_idx, exp_idx].astype(np.float32, copy=False).reshape(-1, 1)
    return X, V, Wb


def build_boundary_batch(
    flat_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    g_values: np.ndarray,
    vmark: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X, x_idx, _, _ = build_feature_block(flat_idx, x, y, t, g_values, vmark)
    X_normals = normals[x_idx].astype(np.float32, copy=False)
    return X, X_normals


# -----------------------------
# Dataset classes
# -----------------------------
class CustomDataset(Dataset):
    """
    Backward-compatible dataset.

    Legacy mode:
        CustomDataset(X, V, W, device)

    Lazy mode:
        CustomDataset(
            x=..., y=..., t=..., vmark=..., g_values=..., vm=..., W=...,
            excluded_flat_indices=...
        )
    """
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop("device", None)

        if "x" in kwargs:
            self._mode = "lazy"

            self.x = torch.from_numpy(np.asarray(kwargs["x"], dtype=np.float32))
            self.y = torch.from_numpy(np.asarray(kwargs["y"], dtype=np.float32))
            self.t = torch.from_numpy(np.asarray(kwargs["t"], dtype=np.float32))
            self.vmark = torch.from_numpy(np.asarray(kwargs["vmark"], dtype=np.float32))
            self.g_values = torch.from_numpy(np.asarray(kwargs["g_values"], dtype=np.float32))

            self.vm = np.asarray(kwargs["vm"], dtype=np.float32)
            self.W = np.asarray(kwargs["W"], dtype=np.float32)

            self.n_space = self.x.numel()
            self.n_time = self.t.numel()
            self.n_exp = self.g_values.numel()
            excluded = kwargs.get("excluded_flat_indices", None)
            self.index_map = ExcludingFlatIndexMap(self.n_space * self.n_time * self.n_exp, excluded)
        else:
            self._mode = "legacy"
            self.X = args[0]
            self.V = args[1]
            self.W = args[2]
            self.device = args[3] if len(args) > 3 else self.device

    def __len__(self):
        if self._mode == "lazy":
            return len(self.index_map)
        return len(self.X)

    def __getitem__(self, idx):
        if self._mode == "lazy":
            flat_idx = self.index_map.to_flat(int(idx))
            x_idx, t_idx, exp_idx = unravel_flat_index(flat_idx, self.n_space, self.n_time)

            X = torch.stack((
                self.x[x_idx],
                self.y[x_idx],
                self.t[t_idx],
                self.g_values[exp_idx],
                self.vmark[x_idx],
            ))
            V = torch.tensor([self.vm[t_idx, x_idx, exp_idx]], dtype=torch.float32)
            W = torch.tensor([self.W[t_idx, x_idx, exp_idx]], dtype=torch.float32)
            return {"X": X, "V": V, "W": W}

        return {
            "X": self.X[idx, :],
            "V": self.V[idx],
            "W": self.W[idx],
        }


class CollocationDataset(Dataset):
    """
    Backward-compatible collocation dataset.

    Legacy mode:
        CollocationDataset(X, device)

    Lazy mode:
        CollocationDataset(
            x=..., y=..., t=..., vmark=..., g_values=...
        )
    """
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop("device", None)

        if "x" in kwargs:
            self._mode = "lazy"

            self.x = torch.from_numpy(np.asarray(kwargs["x"], dtype=np.float32))
            self.y = torch.from_numpy(np.asarray(kwargs["y"], dtype=np.float32))
            self.t = torch.from_numpy(np.asarray(kwargs["t"], dtype=np.float32))
            self.vmark = torch.from_numpy(np.asarray(kwargs["vmark"], dtype=np.float32))
            self.g_values = torch.from_numpy(np.asarray(kwargs["g_values"], dtype=np.float32))

            self.n_space = self.x.numel()
            self.n_time = self.t.numel()
            self.n_exp = self.g_values.numel()
            self.total_size = self.n_space * self.n_time * self.n_exp
        else:
            self._mode = "legacy"
            self.X = args[0]
            self.device = args[1] if len(args) > 1 else self.device

    def __len__(self):
        if self._mode == "lazy":
            return self.total_size
        return len(self.X)

    def __getitem__(self, idx):
        if self._mode == "lazy":
            x_idx, t_idx, exp_idx = unravel_flat_index(int(idx), self.n_space, self.n_time)
            X = torch.stack((
                self.x[x_idx],
                self.y[x_idx],
                self.t[t_idx],
                self.g_values[exp_idx],
                self.vmark[x_idx],
            ))
            return {"X": X}

        return {
            "X": self.X[idx, :],
        }


class BoundaryDataset(Dataset):
    """
    Backward-compatible boundary dataset.

    Legacy mode:
        BoundaryDataset(X, X_normals, device)

    Lazy mode:
        BoundaryDataset(
            x=..., y=..., t=..., vmark=..., g_values=..., normals=...
        )
    """
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop("device", None)

        if "x" in kwargs:
            self._mode = "lazy"

            self.x = torch.from_numpy(np.asarray(kwargs["x"], dtype=np.float32))
            self.y = torch.from_numpy(np.asarray(kwargs["y"], dtype=np.float32))
            self.t = torch.from_numpy(np.asarray(kwargs["t"], dtype=np.float32))
            self.vmark = torch.from_numpy(np.asarray(kwargs["vmark"], dtype=np.float32))
            self.g_values = torch.from_numpy(np.asarray(kwargs["g_values"], dtype=np.float32))
            self.normals = torch.from_numpy(np.asarray(kwargs["normals"], dtype=np.float32))

            self.n_space = self.x.numel()
            self.n_time = self.t.numel()
            self.n_exp = self.g_values.numel()
            self.total_size = self.n_space * self.n_time * self.n_exp
        else:
            self._mode = "legacy"
            self.X = args[0]
            self.X_normals = args[1]
            self.device = args[2] if len(args) > 2 else self.device

    def __len__(self):
        if self._mode == "lazy":
            return self.total_size
        return len(self.X)

    def __getitem__(self, idx):
        if self._mode == "lazy":
            x_idx, t_idx, exp_idx = unravel_flat_index(int(idx), self.n_space, self.n_time)
            X = torch.stack((
                self.x[x_idx],
                self.y[x_idx],
                self.t[t_idx],
                self.g_values[exp_idx],
                self.vmark[x_idx],
            ))
            X_normals = self.normals[x_idx]
            return {"X": X, "X_normals": X_normals}

        return {
            "X": self.X[idx, :],
            "X_normals": self.X_normals[idx, :],
        }
# -----------------------------
# DataModule
# -----------------------------
@dataclass
class ScarPINNDataModule:
    config: dict
    dataset_class: Type[Dataset] = CustomDataset
    collocation_dataset_class: Type[Dataset] = CollocationDataset
    boundary_dataset_class: Type[Dataset] = BoundaryDataset
    device: torch.device = None
    current_phase: str = None

    # Outputs populated by setup_data()
    dataloader: Optional[DataLoader] = None
    collocation_dataloader: Optional[DataLoader] = None
    boundary_dataloader: Optional[DataLoader] = None
    X_test: Optional[torch.Tensor] = None
    V_test: Optional[torch.Tensor] = None
    W_test: Optional[torch.Tensor] = None

    X_collocation_train: Optional[torch.Tensor] = None
    X_boundary: Optional[torch.Tensor] = None
    X_normals: Optional[torch.Tensor] = None
    X_ic_train: Optional[torch.Tensor] = None
    V_ic_train: Optional[torch.Tensor] = None
    W_ic_train: Optional[torch.Tensor] = None

    
    features_all_numpy: Optional[np.ndarray] = None
    features_boundary_all: Optional[np.ndarray] = None
    features_collocation: Optional[np.ndarray] = None
    features_ic_all_numpy: Optional[np.ndarray] = None
    X_normals_all: Optional[np.ndarray] = None
    v_flat_numpy: Optional[np.ndarray] = None
    W_flat_numpy: Optional[np.ndarray] = None

    # Min/max tensors
    mins: Optional[torch.Tensor] = None
    maxes: Optional[torch.Tensor] = None

    def setup_data(self):
        print("Loading data")
        if self.device is None:
            self.device = detect_device()
        print("Using device:", self.device)

        phys = self.config["MODEL"]["physics"]
        self.a = float(phys.get("a", 0.15))
        self.k = float(phys.get("k", 8.0))
        self.mu1 = float(phys.get("mu1", 0.2))
        self.mu2 = float(phys.get("mu2", 0.3))
        self.eps = float(phys.get("eps", 0.002))
        self.b = float(phys.get("b", 0.15))
        self.h = float(phys.get("h", 0.1))
        self.beta = float(phys.get("beta", 1.4e-5))
        self.Cm = float(phys.get("Cm", 1e-3))
        self.mS_cm = float(phys.get("mS_cm", 10.0))
        self.vm_rest = float(phys.get("vm_rest", -80.0))
        self.vm_norm = float(phys.get("vm_norm", 100.0))
        self.t_norm = float(phys.get("t_norm", 12.9))
        self.touAcm2 = 100 / 12.9
        self.tomAcm2 = 100 * 1e-3 / 12.9

        cfgD = self.config["DATA"]
        slice_t = cfgD["sampling"]["slice_t"]
        self.num_domain = cfgD["sampling"]["num_domain"]
        self.num_boundary = cfgD["sampling"]["num_boundary"]
        self.num_init = cfgD["sampling"]["num_init"]
        ic_space_slice = cfgD["sampling"]["ic_space_slice"]
        ic_time_steps = cfgD["sampling"]["ic_time_steps"]

        batch_size = cfgD["loader"]["batch_size"]
        self.batch_size_lbfgs = cfgD["loader"]["batch_size_lbfgs"]

        umtomm = cfgD["constants"]["um_to_mm"]

        np.random.seed(42)

        # -----------------
        # Load geometry
        # -----------------
        vertices, _, _ = get_mesh_data()
        vertices = np.asarray(vertices[:, :2] * umtomm, dtype=np.float32)
        vmark = np.load("./data/vertex_markers.npy").astype(np.float32, copy=False)

        # -----------------
        # Experiment selection
        # -----------------
        g_il_values = np.array([
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
            0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1
        ], dtype=np.float32)

        indices_experiments = np.arange(len(g_il_values))
        sampled_indices = indices_experiments[(indices_experiments + 1) % 4 != 0][:12]
        indices_collocation = sampled_indices

        g_train = g_il_values[sampled_indices]
        g_collocation = g_il_values#[indices_collocation]

        # -----------------
        # Load time series
        # -----------------
        vm_all = np.load("./data/vm_variation_scar_full.npy", mmap_mode="r")
        W_all = np.load("./data/W_variation_scar_full.npy", mmap_mode="r")

        n_timepoints = vm_all.shape[0]
        dt = 5.0
        t_full = np.linspace(0.0, (n_timepoints - 1) * dt, n_timepoints, dtype=np.float32) / self.t_norm
        t = np.asarray(t_full[::slice_t], dtype=np.float32)

        vm_train = np.asarray(vm_all[::slice_t, :, :][:, :, sampled_indices], dtype=np.float32)
        W_train = np.asarray(W_all[::slice_t, :, :][:, :, sampled_indices], dtype=np.float32)

        # -----------------
        # Spatial bases
        # -----------------
        x = vertices[:, 0]
        y = vertices[:, 1]

        x_ic = x[::ic_space_slice]
        y_ic = y[::ic_space_slice]
        vmark_ic = vmark[::ic_space_slice]
        t_ic = t[:ic_time_steps]

        # -----------------
        # Collocation bases
        # -----------------
        collocation_points = np.load("./data/collocation_points.npy").astype(np.float32) * umtomm
        collocation_markers = np.load("./data/collocation_markers.npy").astype(np.float32, copy=False)

        x_inner = collocation_points[:, 0]
        y_inner = collocation_points[:, 1]
        vmark_inner = collocation_markers

        # -----------------
        # Boundary bases
        # -----------------
        vertices_boundary = np.load("./data/boundary_edge_centroids.npy").astype(np.float32) * umtomm
        vertices_boundary = vertices_boundary[:, :2]
        boundary_normals = np.load("./data/boundary_normals.npy").astype(np.float32)[:, :2]
        vmark_boundary = np.load("./data/boundary_markers.npy").astype(np.float32, copy=False)

        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]

        # -----------------
        # Random sampled test set (excluded from train)
        # -----------------
        total_dense = vm_train.shape[0] * vm_train.shape[1] * vm_train.shape[2]
        num_test_samples = min(1024, total_dense)

        idx_test = np.sort(
            np.random.choice(total_dense, num_test_samples, replace=False).astype(np.int64)
        )

        features_test_numpy, V_test_numpy, W_test_numpy = build_dense_batch(
            idx_test,
            x=x,
            y=y,
            t=t,
            g_values=g_train,
            vmark=vmark,
            vm=vm_train,
            W=W_train,
        )

        self.X_test = torch.tensor(features_test_numpy, dtype=torch.float32, device=self.device)
        self.V_test = torch.tensor(V_test_numpy, dtype=torch.float32, device=self.device)
        self.W_test = torch.tensor(W_test_numpy, dtype=torch.float32, device=self.device)

        # -----------------
        # Dense training loader (lazy)
        # -----------------
        self.dataset = self.dataset_class(
            x=x,
            y=y,
            t=t,
            vmark=vmark,
            g_values=g_train,
            vm=vm_train,
            W=W_train,
            excluded_flat_indices=idx_test,
            device=self.device,
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=2
        )
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = 2

        self.dataloader = DataLoader(self.dataset, **loader_kwargs)
        num_batches = len(self.dataloader)
        print(f"Training data: {len(self.dataset)} samples, {num_batches} batches of size {batch_size}")
        self.config["DATA"]["loader"]["num_batches"] = num_batches

        # -----------------
        # Collocation: full lazy loader + sampled tensor for compatibility
        # -----------------
        self.collocation_dataset = self.collocation_dataset_class(
            x=x_inner,
            y=y_inner,
            t=t,
            vmark=vmark_inner,
            g_values=g_collocation,
            device=self.device,
        )

        colloc_loader_kwargs = dict(
            batch_size=self.num_domain,
            shuffle=True,
            num_workers=0,
            #pin_memory=(self.device.type == "cuda"),
            #persistent_workers=True,
            prefetch_factor=None
        )
        #if colloc_loader_kwargs["num_workers"] > 0:
        #    colloc_loader_kwargs["prefetch_factor"] = 2

        self.collocation_dataloader = DataLoader(self.collocation_dataset, **colloc_loader_kwargs)

        total_collocation = x_inner.shape[0] * t.shape[0] * g_collocation.shape[0]
        idx_colocation = np.random.choice(total_collocation, self.num_domain, replace=False)
        X_collocation_numpy, _, _, _ = build_feature_block(
            idx_colocation,
            x=x_inner,
            y=y_inner,
            t=t,
            g_values=g_collocation,
            vmark=vmark_inner,
        )
        self.X_collocation_train = torch.tensor(X_collocation_numpy, dtype=torch.float32, device=self.device)

        print(f"Collocation data: {len(self.collocation_dataset)} samples, collocation batch size {self.num_domain}")
        collocation_data_size_gb = (
            self.X_collocation_train.element_size() * self.X_collocation_train.nelement() / (1024 ** 3)
        )
        print(f"Sampled collocation tensor size: {collocation_data_size_gb:.4f} GB")

        boundary_batch_size = cfgD["loader"].get("boundary_batch_size", self.num_boundary)

        self.boundary_dataset = self.boundary_dataset_class(
            x=x_boundary,
            y=y_boundary,
            t=t,
            vmark=vmark_boundary,
            g_values=g_collocation,
            normals=boundary_normals,
            device=self.device,
        )

        boundary_loader_kwargs = dict(
            batch_size=boundary_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=2
        )
        

        self.boundary_dataloader = DataLoader(self.boundary_dataset, **boundary_loader_kwargs)

        print(f"Boundary data: {len(self.boundary_dataset)} samples, {len(self.boundary_dataloader)} batches of size {boundary_batch_size}")

        # -----------------
        # Boundary sampled tensor
        # -----------------
        total_boundary = x_boundary.shape[0] * t.shape[0] * g_collocation.shape[0]
        idx_boundary = np.random.choice(total_boundary, self.num_boundary, replace=False)

        X_boundary_numpy, X_normals_numpy = build_boundary_batch(
            idx_boundary,
            x=x_boundary,
            y=y_boundary,
            t=t,
            g_values=g_collocation,
            vmark=vmark_boundary,
            normals=boundary_normals,
        )

        self.X_boundary = torch.tensor(X_boundary_numpy, dtype=torch.float32, device=self.device)
        self.X_normals = torch.tensor(X_normals_numpy, dtype=torch.float32, device=self.device)

        # -----------------
        # IC sampled tensor
        # -----------------
        vm_ic = vm_train[:ic_time_steps, ::ic_space_slice, :]
        W_ic = W_train[:ic_time_steps, ::ic_space_slice, :]

        total_ic = x_ic.shape[0] * t_ic.shape[0] * g_train.shape[0]
        idx_ic = np.random.choice(total_ic, self.num_init, replace=False)

        X_ic_numpy, V_ic_numpy, W_ic_numpy = build_dense_batch(
            idx_ic,
            x=x_ic,
            y=y_ic,
            t=t_ic,
            g_values=g_train,
            vmark=vmark_ic,
            vm=vm_ic,
            W=W_ic,
        )

        self.X_ic_train = torch.tensor(X_ic_numpy, dtype=torch.float32, device=self.device)
        self.V_ic_train = torch.tensor(V_ic_numpy, dtype=torch.float32, device=self.device)
        self.W_ic_train = torch.tensor(W_ic_numpy, dtype=torch.float32, device=self.device)

        # -----------------
        # Compatibility placeholders
        # -----------------
        self.features_all_numpy = None
        self.features_boundary_all = None
        self.features_collocation = None
        self.features_ic_all_numpy = None
        self.X_normals_all = None
        self.v_flat_numpy = None
        self.W_flat_numpy = None

        self.X_data_train_cpu = None
        self.V_data_train_cpu = None
        self.W_data_train_cpu = None

        # -----------------
        # Min/Max over feature columns
        # -----------------
        mins_numpy = np.array([
            x.min(),
            y.min(),
            t.min(),
            g_train.min(),
            vmark.min(),
        ], dtype=np.float32)

        maxes_numpy = np.array([
            x.max(),
            y.max(),
            t.max(),
            g_train.max(),
            vmark.max(),
        ], dtype=np.float32)

        self.mins = torch.tensor(mins_numpy, dtype=torch.float32, device=self.device)
        self.maxes = torch.tensor(maxes_numpy, dtype=torch.float32, device=self.device)


    def setup_data_with_conductivities(self):
        """
        Loads geometry/data from disk and processes into:
        - train_loader (DataLoader)
        - test tensors (quick eval)
        - boundary arrays (for BC)
        - collocation arrays (for PDE)
        - initial-condition arrays (for IC)
        - input min/max tensors (computed from the training data)
        """
        # -----------------
        # Device
        # -----------------
        print("Loading data")
        if self.device is None:
            self.device = detect_device()
        print("Using device:", self.device)

        # -----------------
        # Constants & config
        # -----------------
        phys = self.config["MODEL"]["physics"]
        self.a   = float(phys.get("a", 0.15))
        self.k   = float(phys.get("k", 8.0))
        self.mu1 = float(phys.get("mu1", 0.2))
        self.mu2 = float(phys.get("mu2", 0.3))
        self.eps = float(phys.get("eps", 0.002))
        self.b   = float(phys.get("b", 0.15))
        self.h   = float(phys.get("h", 0.1))
        self.beta = float(phys.get("beta", 1.4e-5))
        self.Cm   = float(phys.get("Cm", 1e-3))
        self.mS_cm = float(phys.get("mS_cm", 10.0))
        self.vm_rest = float(phys.get("vm_rest", -80.0))  # mV
        self.vm_norm = float(phys.get("vm_norm", 100.0))
        self.t_norm = float(phys.get("t_norm",12.9))
        self.touAcm2 = 100 / 12.9
        self.tomAcm2 = 100 * 1e-3 / 12.9

        cfgD = self.config["DATA"]
        slice_t = cfgD["sampling"]["slice_t"]
        self.num_domain = cfgD["sampling"]["num_domain"]
        self.num_boundary = cfgD["sampling"]["num_boundary"]
        self.num_init = cfgD["sampling"]["num_init"]
        ic_space_slice = cfgD["sampling"]["ic_space_slice"]
        ic_time_steps = cfgD["sampling"]["ic_time_steps"]
        nx_test_slice = cfgD["sampling"]["nx_test_slice"]
        nt_test_slice = cfgD["sampling"]["nt_test_slice"]
        indices_collocation = cfgD["sampling"]["indices_collocation"]
        #if string and string is "all",pick all indices
        if isinstance(indices_collocation, str) and indices_collocation == "all":
                indices_collocation = list(range(19))
        

        batch_size = cfgD["loader"]["batch_size"]
        self.batch_size_lbfgs = cfgD["loader"]["batch_size_lbfgs"]

        umtomm = cfgD["constants"]["um_to_mm"]#1e-3
        s_mtomm2_TU = 12.9/1.4 #cfgD["constants"]["s_mtomm2_TU"] # 12.9/1.4

        np.random.seed(42)

        # -----------------
        # Load data
        # -----------------
        vertices, triangles, _ = get_mesh_data()
        vertices = (vertices * umtomm)  # mm

        g_il_values = np.array([
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
            0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1
        ])
        indices_experiments = np.arange(len(g_il_values))
        sampled_indices = indices_experiments[(indices_experiments + 1) % 4 != 0][:12]
        #print("############# Sampled indices #############", sampled_indices)
        #print("g_il_values in training data", g_il_values[sampled_indices])

        vm_all = np.load("./data/vm_variation_scar_full.npy")

        W_all = np.load("./data/W_variation_scar_full.npy")

        n_timepoints = vm_all.shape[0]
        dt = 5
        t = np.linspace(0, (n_timepoints - 1) * dt, n_timepoints) / self.t_norm

        t_test = t[::nt_test_slice]
        t = t[::slice_t]

        x_test = vertices[::nx_test_slice, 0]
        y_test = vertices[::nx_test_slice, 1]
        indices_test = [-1]

        vm_test = vm_all[::nt_test_slice, ::nx_test_slice, indices_test]

        vm_all = vm_all[::slice_t, :, sampled_indices]
        W_test = W_all[::nt_test_slice, ::nx_test_slice, indices_test]
        W_all = W_all[::slice_t, :, sampled_indices]

        conductivities_all = np.load('./data/conductivities_variation_scar_full.npy')
        conductivities_boundary = np.load("./data/conductivities_boundary_1.npy")
        conductivities_all = conductivities_all * s_mtomm2_TU

        indices_collocation = [0, 1,2, 3, 14, 16, 18]
        conductivities_collocation = conductivities_all[:, :, :, indices_collocation]
        conductivities_test = conductivities_all[::nx_test_slice, :, :, indices_test]
        conductivities_all = conductivities_all[:, :, :, sampled_indices]
        conductivities_boundary = (conductivities_boundary * s_mtomm2_TU)[:, :, :, indices_collocation]

        # Boundary points and normals
        vertices_boundary = np.load("./data/boundary_edge_centroids.npy") * umtomm
        vertices_boundary = vertices_boundary[:, :2]
        boundary_points_old, triangles_boundary_old = get_boundary()
        boundary_points_old = boundary_points_old[:, :2] * umtomm
        boundary_normals = np.load("./data/boundary_normals.npy")[:, :2]

        # Excluding boundary points from vertices
        vertices_expanded = vertices[:, np.newaxis]
        boundary_vertices_expanded = boundary_points_old[np.newaxis, :]
        is_vertex_on_boundary = np.any(
            np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1
        )
        b_verts = vertices[is_vertex_on_boundary, :]
        np.save("./data/boundary_points_old.npy", b_verts)
        unique_vertices = vertices[~is_vertex_on_boundary, :]
        inner_points_indices = np.load("./data/inner_points_indices.npy")
        x_inner = vertices[inner_points_indices, 0]
        y_inner = vertices[inner_points_indices, 1]

        X_inner,T_inner = np.meshgrid(x_inner,t)
        Y_inner,T_inner = np.meshgrid(y_inner,t)
        X_inner = X_inner.reshape(-1,1)
        T_inner = T_inner.reshape(-1,1)
        Y_inner = Y_inner.reshape(-1,1)

        sigma_a_collocation = conductivities_collocation[inner_points_indices, 0, 0, :]
        sigma_b_collocation = conductivities_collocation[inner_points_indices, 0, 1, :]
        sigma_c_collocation = conductivities_collocation[inner_points_indices, 1, 0, :]
        sigma_d_collocation = conductivities_collocation[inner_points_indices, 1, 1, :]

        vm_all_inner = vm_all[:, ~is_vertex_on_boundary, :]
        W_all_inner = W_all[:, ~is_vertex_on_boundary, :]
        conductivities_all = conductivities_all[~is_vertex_on_boundary, :, :, :]

        # Space/time grids
        x = unique_vertices[:, 0]
        y = unique_vertices[:, 1]
        x_ic = unique_vertices[::ic_space_slice, 0]
        y_ic = unique_vertices[::ic_space_slice, 1]

        X, T = np.meshgrid(x, t)
        Y, T = np.meshgrid(y, t)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        X_test, T_test = np.meshgrid(x_test, t_test)
        Y_test, T_test = np.meshgrid(y_test, t_test)
        X_test = X_test.reshape(-1, 1)
        T_test = T_test.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
        V_test = vm_test.reshape(-1, 1)
        W_test = W_test.reshape(-1, 1)

        # Boundary meshgrid
        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)

        # Boundary normals grids
        X_boundary_normals, T_boundary_normals = np.meshgrid(boundary_normals[:, 0], t)
        Y_boundary_normals, T_boundary_normals = np.meshgrid(boundary_normals[:, 1], t)
        X_boundary_normals = X_boundary_normals.reshape(-1, 1)
        Y_boundary_normals = Y_boundary_normals.reshape(-1, 1)
        X_normals = np.hstack((X_boundary_normals, Y_boundary_normals))
        self.X_normals_all = np.zeros((X_boundary.shape[0] * vm_test.shape[1], 2), dtype=np.float32)

        # Conductivities split
        self.conductivities_all = conductivities_all[:, :, :, :]
        sigma_a_train = self.conductivities_all[:, 0, 0, :]
        sigma_b_train = self.conductivities_all[:, 0, 1, :]
        sigma_c_train = self.conductivities_all[:, 1, 0, :]
        sigma_d_train = self.conductivities_all[:, 1, 1, :]
        sigma_a_ic = conductivities_all[::ic_space_slice, 0, 0, :]
        sigma_b_ic = conductivities_all[::ic_space_slice, 0, 1, :]
        sigma_c_ic = conductivities_all[::ic_space_slice, 1, 0, :]
        sigma_d_ic = conductivities_all[::ic_space_slice, 1, 1, :]

        X_c, T_c = np.meshgrid(x, t)

        # Test conductivities (meshed over t_test)
        sigma_a_test = conductivities_test[:, 0, 0]
        sigma_b_test = conductivities_test[:, 0, 1]
        sigma_c_test = conductivities_test[:, 1, 0]
        sigma_d_test = conductivities_test[:, 1, 1]
        sigma_a_test, _ = np.meshgrid(sigma_a_test, t_test)
        sigma_b_test, _ = np.meshgrid(sigma_b_test, t_test)
        sigma_c_test, _ = np.meshgrid(sigma_c_test, t_test)
        sigma_d_test, _ = np.meshgrid(sigma_d_test, t_test)
        sigma_a_test = sigma_a_test.reshape(-1, 1)
        sigma_b_test = sigma_b_test.reshape(-1, 1)
        sigma_c_test = sigma_c_test.reshape(-1, 1)
        sigma_d_test = sigma_d_test.reshape(-1, 1)
        experiment_feature_test = np.ones(sigma_a_test.shape) * (g_il_values[-1])
        features_test_numpy = np.hstack((
            X_test, Y_test, T_test, sigma_a_test, sigma_b_test, sigma_c_test, sigma_d_test, experiment_feature_test
        ))

        # IC grids
        t_ic = t[:ic_time_steps]
        X_ic, T_ic = np.meshgrid(x_ic, t_ic)
        Y_ic, T_ic = np.meshgrid(y_ic, t_ic)
        X_ic = X_ic.reshape(-1, 1)
        T_ic = T_ic.reshape(-1, 1)
        Y_ic = Y_ic.reshape(-1, 1)

        vm_train = vm_all_inner[:, :, :]
        W_train = W_all_inner[:, :, :]

        self.features_all_numpy = np.zeros((vm_train.shape[0] * vm_train.shape[1] * vm_train.shape[2], 8))
        self.features_collocation = np.zeros((X_inner.shape[0] * sigma_a_collocation.shape[-1], 8))
        self.features_boundary_all = np.zeros((x_boundary.shape[0] * t.shape[0] * conductivities_boundary.shape[-1], 8))
        self.features_ic_all_numpy = np.zeros((X_ic.shape[0] * vm_train.shape[2], 8), dtype=np.float32)
        self.V_ic_all_numpy = np.zeros((X_ic.shape[0] * vm_train.shape[2], 1), dtype=np.float32)
        self.W_ic_all_numpy = np.zeros((X_ic.shape[0] * vm_train.shape[2], 1), dtype=np.float32)
        self.v_flat_numpy = np.zeros((vm_train.shape[0] * vm_train.shape[1] * vm_train.shape[2], 1))
        self.W_flat_numpy = np.zeros((W_train.shape[0] * W_train.shape[1] * W_train.shape[2], 1))

        # Collocation features 
        for i in range(conductivities_collocation.shape[3]):
            start = i *X_inner.shape[0]
            end = (i + 1) * X_inner.shape[0]

            sigma_a_collocation_mesh, _ = np.meshgrid(sigma_a_collocation[:, i], t)
            sigma_b_collocation_mesh, _ = np.meshgrid(sigma_b_collocation[:, i], t)
            sigma_c_collocation_mesh, _ = np.meshgrid(sigma_c_collocation[:, i], t)
            sigma_d_collocation_mesh, _ = np.meshgrid(sigma_d_collocation[:, i], t)
            sigma_a_collocation_flat = sigma_a_collocation_mesh.reshape(-1, 1)
            sigma_b_collocation_flat = sigma_b_collocation_mesh.reshape(-1, 1)
            sigma_c_collocation_flat = sigma_c_collocation_mesh.reshape(-1, 1)
            sigma_d_collocation_flat = sigma_d_collocation_mesh.reshape(-1, 1)
            experiment_feature = np.ones(sigma_a_collocation_flat.shape) * g_il_values[indices_collocation[i]]
            Xi_collocation = np.hstack((
                X_inner, Y_inner, T_inner,
                sigma_a_collocation_flat, sigma_b_collocation_flat,
                sigma_c_collocation_flat, sigma_d_collocation_flat,
                experiment_feature,
            ))
            self.features_collocation[start:end, :] = Xi_collocation

            # Boundary points for this collocation experiment
            sigma_a_boundary = conductivities_boundary[:, 0, 0, i]
            sigma_b_boundary = conductivities_boundary[:, 0, 1, i]
            sigma_c_boundary = conductivities_boundary[:, 1, 0, i]
            sigma_d_boundary = conductivities_boundary[:, 1, 1, i]
            sigma_a_boundary, _ = np.meshgrid(sigma_a_boundary, t)
            sigma_b_boundary, _ = np.meshgrid(sigma_b_boundary, t)
            sigma_c_boundary, _ = np.meshgrid(sigma_c_boundary, t)
            sigma_d_boundary, _ = np.meshgrid(sigma_d_boundary, t)
            sigma_a_boundary = sigma_a_boundary.reshape(-1, 1)
            sigma_b_boundary = sigma_b_boundary.reshape(-1, 1)
            sigma_c_boundary = sigma_c_boundary.reshape(-1, 1)
            sigma_d_boundary = sigma_d_boundary.reshape(-1, 1)
            start_bc = i * X_boundary.shape[0]
            end_bc = (i + 1) * X_boundary.shape[0]
            experiment_feature = np.ones(sigma_a_boundary.shape) * g_il_values[indices_collocation[i]]
            self.features_boundary_all[start_bc:end_bc, :] = np.hstack((
                X_boundary, Y_boundary, T_boundary,
                sigma_a_boundary, sigma_b_boundary, sigma_c_boundary, sigma_d_boundary,
                experiment_feature,
            ))
            # normals
            start_normals = i * X_normals.shape[0]
            end_normals = (i + 1) * X_normals.shape[0]
            self.X_normals_all[start_normals:end_normals, :] = X_normals

        # Dense training features per experiment
        for i in range(vm_train.shape[2]):
            start = i * vm_train.shape[0] * vm_train.shape[1]
            end = (i + 1) * vm_train.shape[0] * vm_train.shape[1]
            self.v_flat_numpy[start:end] = vm_train[:, :, i].reshape(-1, 1)
            self.W_flat_numpy[start:end] = W_train[:, :, i].reshape(-1, 1)

            sigma_a, _ = np.meshgrid(sigma_a_train[:, i], t)
            sigma_b, _ = np.meshgrid(sigma_b_train[:, i], t)
            sigma_c, _ = np.meshgrid(sigma_c_train[:, i], t)
            sigma_d, _ = np.meshgrid(sigma_d_train[:, i], t)
            sigma_a_flat = sigma_a.reshape(-1, 1)
            sigma_b_flat = sigma_b.reshape(-1, 1)
            sigma_c_flat = sigma_c.reshape(-1, 1)
            sigma_d_flat = sigma_d.reshape(-1, 1)
            experiment_feature = np.ones(sigma_a_flat.shape) * g_il_values[sampled_indices[i]]

            X_i = np.hstack((
                X, Y, T,
                sigma_a_flat, sigma_b_flat, sigma_c_flat, sigma_d_flat,
                experiment_feature,
            ))
            self.features_all_numpy[start:end, :] = X_i

            # IC sampling
            sigma_a_ic_, _ = np.meshgrid(sigma_a_ic[:, i], t[:ic_time_steps])
            sigma_b_ic_, _ = np.meshgrid(sigma_b_ic[:, i], t[:ic_time_steps])
            sigma_c_ic_, _ = np.meshgrid(sigma_c_ic[:, i], t[:ic_time_steps])
            sigma_d_ic_, _ = np.meshgrid(sigma_d_ic[:, i], t[:ic_time_steps])
            sigma_a_ic_flat = sigma_a_ic_.reshape(-1, 1)
            sigma_b_ic_flat = sigma_b_ic_.reshape(-1, 1)
            sigma_c_ic_flat = sigma_c_ic_.reshape(-1, 1)
            sigma_d_ic_flat = sigma_d_ic_.reshape(-1, 1)
            experiment_feature_ic = np.ones(sigma_a_ic_flat.shape) * g_il_values[sampled_indices[i]]
            start_ic = i * (x_ic.shape[0] * t[:ic_time_steps].shape[0])
            end_ic = (i + 1) * (x_ic.shape[0] * t[:ic_time_steps].shape[0])
            self.features_ic_all_numpy[start_ic:end_ic, :] = np.hstack((
                X_ic, Y_ic, T_ic,
                sigma_a_ic_flat, sigma_b_ic_flat, sigma_c_ic_flat, sigma_d_ic_flat,
                experiment_feature_ic,
            ))
            self.V_ic_all_numpy[start_ic:end_ic] = vm_train[:ic_time_steps, ::ic_space_slice, i].reshape(-1, 1)
            self.W_ic_all_numpy[start_ic:end_ic] = W_train[:ic_time_steps, ::ic_space_slice, i].reshape(-1, 1)

        # -----------------
        # Random sampling for PDE/BC/IC sets
        # -----------------
        idx_colocation = np.random.choice(self.features_collocation.shape[0], self.num_domain, replace=False)
        X_collocation_numpy = self.features_collocation[idx_colocation]
        self.X_collocation_train = torch.tensor(self.features_collocation,dtype=torch.float32)#torch.tensor(X_collocation_numpy, dtype=torch.float32, device=self.device)
        #Using loader
        self.collocation_dataset = self.collocation_dataset_class(self.X_collocation_train, self.device)
        self.collocation_dataloader = DataLoader(
            self.collocation_dataset,
            batch_size=self.num_domain,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=2,
        )
        

        idx_boundary = np.random.choice(self.features_boundary_all.shape[0], self.num_boundary, replace=False)
        self.X_boundary = torch.tensor(self.features_boundary_all[idx_boundary], dtype=torch.float32, device=self.device)
        self.X_normals = torch.tensor(self.X_normals_all[idx_boundary], dtype=torch.float32, device=self.device)

        idx_ic = np.random.choice(self.features_ic_all_numpy.shape[0], self.num_init, replace=False)
        self.X_ic_train = torch.tensor(self.features_ic_all_numpy[idx_ic], dtype=torch.float32, device=self.device)
        self.V_ic_train = torch.tensor(self.V_ic_all_numpy[idx_ic], dtype=torch.float32, device=self.device)
        self.W_ic_train = torch.tensor(self.W_ic_all_numpy[idx_ic], dtype=torch.float32, device=self.device)

        # -----------------
        # Data 
        # -----------------
        self.X_data_train_cpu = torch.tensor(self.features_all_numpy, dtype=torch.float32)
        self.V_data_train_cpu = torch.tensor(self.v_flat_numpy, dtype=torch.float32)
        self.W_data_train_cpu = torch.tensor(self.W_flat_numpy, dtype=torch.float32)

        self.dataset = self.dataset_class(self.X_data_train_cpu, self.V_data_train_cpu, self.W_data_train_cpu, self.device)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=2,
        )
        num_batches = len(self.dataloader)
        print(f"Training data: {len(self.dataset)} samples, {num_batches} batches of size {batch_size}")
        #save number of batches to config
        self.config["DATA"]["loader"]["num_batches"] = num_batches

        self.X_test = torch.tensor(features_test_numpy, dtype=torch.float32, device=self.device)
        self.V_test = torch.tensor(V_test, dtype=torch.float32, device=self.device)
        self.W_test = torch.tensor(W_test, dtype=torch.float32, device=self.device)

        # -----------------
        # Min/Max over training data 
        # -----------------
        maxes_numpy = self.features_all_numpy.max(axis=0)
        mins_numpy = self.features_all_numpy.min(axis=0)
        self.maxes = torch.tensor(maxes_numpy, dtype=torch.float32, device=self.device)
        self.mins = torch.tensor(mins_numpy, dtype=torch.float32, device=self.device)



    def get_minmax_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.mins is not None and self.maxes is not None, "Call setup_data() first."
        return self.mins, self.maxes

    @torch.no_grad()
    def resample(self):
        """Resample collocation set """

        idx_boundary = np.random.choice(self.features_boundary_all.shape[0], self.num_boundary, replace=False)
        self.X_boundary = torch.tensor(self.features_boundary_all[idx_boundary], dtype=torch.float32, device=self.device)
        self.X_normals = torch.tensor(self.X_normals_all[idx_boundary], dtype=torch.float32, device=self.device)
