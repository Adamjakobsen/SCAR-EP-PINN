from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from case_data import CaseData, load_case_data, load_fields, resolve_case, case_dataset_path, resolve_cases

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    """Map training indices to grid rows, skipping held-out rows."""
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

    def to_flat_array(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return idx + np.searchsorted(self.adjusted, idx, side="right")


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
    V = vm[exp_idx, t_idx, x_idx].astype(np.float32, copy=False).reshape(-1, 1)
    Wb = W[exp_idx, t_idx, x_idx].astype(np.float32, copy=False).reshape(-1, 1)
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


N_TRAIN_EXPERIMENTS = 12


def training_experiment_indices(n_sims: int) -> np.ndarray:
    """Exclude every fourth sweep value and take the first 12 remaining values."""
    indices = np.arange(n_sims)
    return indices[(indices + 1) % 4 != 0][:N_TRAIN_EXPERIMENTS]


def time_axis(n_times: int, dt_ms: float, t_norm: float, slice_t: int = 1) -> np.ndarray:
    """Time in model units (ms / t_norm)."""
    full = np.arange(n_times, dtype=np.float32) * float(dt_ms) / float(t_norm)
    return np.asarray(full[::slice_t], dtype=np.float32)


def feature_minmax(x, y, t, g_train, vmark) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of [x, y, t, g_il, marker], using training conductivities only.

    The PINN maps these bounds to [-2, 2]. Interpolation values stay within
    this range; extrapolation values can fall outside it.
    """
    mins = np.array(
        [x.min(), y.min(), t.min(), g_train.min(), vmark.min()], dtype=np.float32
    )
    maxes = np.array(
        [x.max(), y.max(), t.max(), g_train.max(), vmark.max()], dtype=np.float32
    )
    return mins, maxes


class RandomBatchSampler(Sampler):
    """Yield random index batches.

    Without replacement, each epoch stores a permutation of all rows.
    With replacement, each batch is sampled independently.
    """

    def __init__(self, n_samples, batch_size, replacement=False, generator=None):
        self.n_samples = int(n_samples)
        self.batch_size = int(batch_size)
        self.replacement = bool(replacement)
        self.generator = generator
        self.num_batches = max(1, -(-self.n_samples // self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.replacement:
            for _ in range(self.num_batches):
                yield torch.randint(
                    self.n_samples, (self.batch_size,), generator=self.generator
                ).numpy()
        else:
            order = torch.randperm(self.n_samples, generator=self.generator).numpy()
            for start in range(0, self.n_samples, self.batch_size):
                yield order[start:start + self.batch_size]


class LazyFeatureDataset(Dataset):
    """Build [x_mm, y_mm, t, g_il, marker] batches from flat grid indices.

    Rows are ordered by experiment, time, then point. Indexing accepts a
    list or array of indices and returns the corresponding batch.
    """

    def __init__(self, *, x, y, t, vmark, g_values, device=None):
        self.device = device
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.t = np.ascontiguousarray(t, dtype=np.float32)
        self.vmark = np.ascontiguousarray(vmark, dtype=np.float32)
        self.g_values = np.ascontiguousarray(g_values, dtype=np.float32)

        self.n_space = self.x.shape[0]
        self.n_time = self.t.shape[0]
        self.n_exp = self.g_values.shape[0]
        self.total_size = self.n_space * self.n_time * self.n_exp

    def __len__(self):
        return self.total_size

    @staticmethod
    def _as_index_array(index) -> np.ndarray:
        if isinstance(index, slice):
            raise TypeError("slice indexing is not supported; pass a list of indices")
        return np.atleast_1d(np.asarray(index, dtype=np.int64))

    def _to_flat(self, index) -> np.ndarray:
        return self._as_index_array(index)

    def _features(self, index):
        flat = self._to_flat(index)
        X, x_idx, t_idx, exp_idx = build_feature_block(
            flat, self.x, self.y, self.t, self.g_values, self.vmark
        )
        return torch.from_numpy(X), x_idx, t_idx, exp_idx

    def __getitem__(self, index):
        X, _, _, _ = self._features(index)
        return {"X": X}


class CustomDataset(LazyFeatureDataset):
    """Vertex samples with Vm and W targets, excluding held-out rows."""

    def __init__(self, *, vm, W, excluded_flat_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.vm = np.asarray(vm, dtype=np.float32)
        self.W = np.asarray(W, dtype=np.float32)
        self.index_map = ExcludingFlatIndexMap(self.total_size, excluded_flat_indices)

    def __len__(self):
        return len(self.index_map)

    def _to_flat(self, index) -> np.ndarray:
        return self.index_map.to_flat_array(self._as_index_array(index))

    def __getitem__(self, index):
        X, x_idx, t_idx, exp_idx = self._features(index)
        return {
            "X": X,
            "V": torch.from_numpy(
                self.vm[exp_idx, t_idx, x_idx].reshape(-1, 1).astype(np.float32, copy=False)
            ),
            "W": torch.from_numpy(
                self.W[exp_idx, t_idx, x_idx].reshape(-1, 1).astype(np.float32, copy=False)
            ),
        }


class CollocationDataset(LazyFeatureDataset):
    """PDE-residual samples on the triangle centroids."""


class BoundaryDataset(LazyFeatureDataset):
    """No-flux BC samples on the boundary edge centroids, with outward normals."""

    def __init__(self, *, normals, **kwargs):
        super().__init__(**kwargs)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)

    def __getitem__(self, index):
        X, x_idx, _, _ = self._features(index)
        return {"X": X, "X_normals": torch.from_numpy(self.normals[x_idx])}


@dataclass
class ScarPINNDataModule:
    config: dict
    case: Optional[str] = None
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

    X_boundary: Optional[torch.Tensor] = None
    X_normals: Optional[torch.Tensor] = None
    X_ic_train: Optional[torch.Tensor] = None
    V_ic_train: Optional[torch.Tensor] = None
    W_ic_train: Optional[torch.Tensor] = None


    mins: Optional[torch.Tensor] = None
    maxes: Optional[torch.Tensor] = None

    case_data: Optional[CaseData] = None

    def _make_loader(
        self, dataset, batch_size: int, num_workers: int, replacement: bool = False
    ) -> DataLoader:
        sampler = RandomBatchSampler(
            len(dataset), batch_size, replacement=replacement
        )
        kwargs = dict(
            sampler=sampler,
            batch_size=None,          # sampler already yields batches
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        if num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(dataset, **kwargs)

    def setup_data(self):
        print("Loading data")
        if self.device is None:
            self.device = detect_device()
        print("Using device:", self.device)

        self.t_norm = float(self.config["MODEL"].get("physics", {}).get("t_norm", 12.9))

        cfgD = self.config["DATA"]
        slice_t = cfgD["sampling"]["slice_t"]
        self.num_domain = cfgD["sampling"]["num_domain"]
        self.num_boundary = cfgD["sampling"]["num_boundary"]
        self.num_init = cfgD["sampling"]["num_init"]
        ic_space_slice = cfgD["sampling"]["ic_space_slice"]
        ic_time_steps = cfgD["sampling"]["ic_time_steps"]
        nx_test_slice = cfgD["sampling"]["nx_test_slice"]

        batch_size = cfgD["loader"]["batch_size"]
        self.batch_size_lbfgs = cfgD["loader"]["batch_size_lbfgs"]
        num_workers = int(cfgD["loader"].get("num_workers", 4))

        umtomm = cfgD["constants"]["um_to_mm"]

        np.random.seed(42)

        self.case = resolve_case(self.config, self.case)
        dataset_path = case_dataset_path(self.case)
        data = load_case_data(self.case, dataset_path)
        self.case_data = data
        print(f"Dataset: {dataset_path}")
        print(f"Geometry: {data.summary()}")

        vertices = data.vertices          # (V, 2), already in mm
        vmark = data.vertex_markers

        g_il_values = data.g_il
        sampled_indices = training_experiment_indices(len(g_il_values))

        g_train = g_il_values[sampled_indices]
        g_collocation = g_il_values

        # Field axes: experiment, time, vertex.
        vm_train, W_train = load_fields(dataset_path, sampled_indices, slice_t=slice_t)
        print(
            f"Fields: vm {vm_train.shape} "
            f"({vm_train.nbytes / 2**30:.2f} GiB x2), "
            f"experiments {np.round(g_train, 4).tolist()}"
        )

        vm_train = vm_train[:, :, ::nx_test_slice]
        W_train = W_train[:, :, ::nx_test_slice]

        t = time_axis(data.n_times, data.dt_ms, self.t_norm, slice_t)

        x = vertices[::nx_test_slice, 0]
        y = vertices[::nx_test_slice, 1]

        x_ic = x[::ic_space_slice]
        y_ic = y[::ic_space_slice]
        vmark_ic = vmark[::ic_space_slice]
        t_ic = t[:ic_time_steps]

        # Collocation points and tags come from triangle centroids.
        x_inner = data.collocation_points[:, 0]
        y_inner = data.collocation_points[:, 1]
        vmark_inner = data.collocation_markers

        # Boundary edges include the outer wall and interior holes.
        boundary_normals = data.boundary_normals
        vmark_boundary = data.boundary_markers
        x_boundary = data.boundary_points[:, 0]
        y_boundary = data.boundary_points[:, 1]

        # Hold out these rows from the training dataset.
        total_dense = vm_train.shape[0] * vm_train.shape[1] * vm_train.shape[2]
        num_test_samples = min(int(cfgD["sampling"].get("num_test_samples", 1024)), total_dense - 1)
        if total_dense < 2 or num_test_samples < 1:
            raise ValueError("Need at least one training row and one held-out row")

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

        self.dataloader = self._make_loader(self.dataset, batch_size, num_workers)
        num_batches = len(self.dataloader)
        print(f"Training data: {len(self.dataset)} samples, {num_batches} batches of size {batch_size}, {x.shape[0]} vertices, {t.shape[0]} time steps, {g_train.shape[0]} experiments")
        self.config["DATA"]["loader"]["num_batches"] = num_batches

        self.collocation_dataset = self.collocation_dataset_class(
            x=x_inner,
            y=y_inner,
            t=t,
            vmark=vmark_inner,
            g_values=g_collocation,
            device=self.device,
        )

        self.collocation_dataloader = self._make_loader(
            self.collocation_dataset, self.num_domain, num_workers
        )

        print(
            f"Collocation data: {len(self.collocation_dataset)} samples, "
            f"batch size {self.num_domain}"
        )

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

        self.boundary_dataloader = self._make_loader(
            self.boundary_dataset, boundary_batch_size, num_workers
        )

        print(f"Boundary data: {len(self.boundary_dataset)} samples, {len(self.boundary_dataloader)} batches of size {boundary_batch_size}")

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

        # Initial conditions use the selected FEM snapshots.
        vm_ic = vm_train[:, :ic_time_steps, ::ic_space_slice]
        W_ic = W_train[:, :ic_time_steps, ::ic_space_slice]

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

        # Retain the boundary arrays for resample().
        self._boundary_base = dict(
            x=x_boundary,
            y=y_boundary,
            t=t,
            g_values=g_collocation,
            vmark=vmark_boundary,
            normals=boundary_normals,
        )

        mins_numpy, maxes_numpy = feature_minmax(x, y, t, g_train, vmark)

        self.mins = torch.tensor(mins_numpy, dtype=torch.float32, device=self.device)
        self.maxes = torch.tensor(maxes_numpy, dtype=torch.float32, device=self.device)

    def get_minmax_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.mins is not None and self.maxes is not None, "Call setup_data() first."
        return self.mins, self.maxes

    @torch.no_grad()
    def resample(self):
        """Draw a fresh boundary collocation batch for the BC loss."""
        base = getattr(self, "_boundary_base", None)
        if base is None:
            raise RuntimeError("Call setup_data() before resample().")

        total_boundary = (
            base["x"].shape[0] * base["t"].shape[0] * base["g_values"].shape[0]
        )
        idx_boundary = np.random.choice(total_boundary, self.num_boundary, replace=False)
        X_boundary_numpy, X_normals_numpy = build_boundary_batch(idx_boundary, **base)

        self.X_boundary = torch.tensor(
            X_boundary_numpy, dtype=torch.float32, device=self.device
        )
        self.X_normals = torch.tensor(
            X_normals_numpy, dtype=torch.float32, device=self.device
        )


def iter_case_data_modules(config, cases=None, *, device=None):
    """Yield one data module per case, each with its own configuration."""
    names = resolve_cases(config, cases=cases)
    paths = [case_dataset_path(name).resolve() for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple cases must resolve to different HDF5 files")
    for case in names:
        selected = deepcopy(config)
        selected.setdefault("DATA", {})["case"] = case
        module = ScarPINNDataModule(selected, case=case, device=device)
        module.setup_data()
        yield module
