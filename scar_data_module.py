"""Original PINN-compatible per-case HDF5 loader. No revised physics imports.

Features are [x_mm, y_mm, time_ms/12.9, g_il, region_marker]. Each case has
its own data module; the original four-input PINN is not given mixed geometries.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from case_data import CaseData, load_case_data, load_fields, resolve_case, resolve_dataset_path, resolve_cases

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

    def to_flat_array(self, idx: np.ndarray) -> np.ndarray:
        """Vectorised to_flat for a whole batch of dataset indices."""
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


# -----------------------------
# Experiment / feature conventions
#
# Shared with the plotting and prediction path so the two cannot drift: the
# PINN's input scaler is fitted on these exact values, and reusing a checkpoint
# with different mins/maxes silently mis-scales every prediction.
# -----------------------------
N_TRAIN_EXPERIMENTS = 12


def training_experiment_indices(n_sims: int) -> np.ndarray:
    """Indices of the experiments used for the data loss.

    Drops every 4th value of the sweep and keeps the first 12, leaving 3
    interpolation and 4 extrapolation values held out.
    """
    indices = np.arange(n_sims)
    return indices[(indices + 1) % 4 != 0][:N_TRAIN_EXPERIMENTS]


def time_axis(n_times: int, dt_ms: float, t_norm: float, slice_t: int = 1) -> np.ndarray:
    """Normalised time axis the model was trained on (ms / t_norm)."""
    full = np.arange(n_times, dtype=np.float32) * float(dt_ms) / float(t_norm)
    return np.asarray(full[::slice_t], dtype=np.float32)


def feature_minmax(x, y, t, g_train, vmark) -> tuple[np.ndarray, np.ndarray]:
    """Per-column min/max of [x, y, t, g_il, marker] for the model's scaler.

    Note g_il spans only the *training* experiments, deliberately: held-out
    interpolation and extrapolation values scale outside [-1, 1].
    """
    mins = np.array(
        [x.min(), y.min(), t.min(), g_train.min(), vmark.min()], dtype=np.float32
    )
    maxes = np.array(
        [x.max(), y.max(), t.max(), g_train.max(), vmark.max()], dtype=np.float32
    )
    return mins, maxes


# -----------------------------
# Samplers
# -----------------------------
class RandomBatchSampler(Sampler):
    """Yields batches of random indices as one array per batch.

    ``replacement=False`` reproduces ordinary shuffling: an epoch is an exact
    permutation, so every row is seen once. That costs a ``randperm`` of the
    whole grid, which is fine for the data loss (~2e8 rows) but not for the
    collocation grid (~8.5e8 rows = a 6.8 GB allocation, rebuilt every time
    the loader is re-iterated).

    ``replacement=True`` draws each batch independently and allocates nothing.
    For residual sampling that is the better trade: the collocation and
    boundary grids are arbitrary sampling grids rather than a fixed training
    set, and a 4096-row draw from 1e8+ rows collides with negligible
    probability.
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


# -----------------------------
# Dataset classes
# -----------------------------
class LazyFeatureDataset(Dataset):
    """A (point, time, experiment) grid addressed by one flat index.

    Nothing is materialised: each fetch rebuilds its feature rows
    ``[x, y, t, g_il, marker]`` on demand, which is what keeps grids of
    ~1e8-1e9 samples tractable.

    Indexing is **batched**. ``dataset[list_of_indices]`` returns a whole
    collated batch built with vectorised numpy. Feeding these datasets one
    index at a time costs ~200x more per sample, enough to make the loader,
    not the GPU, the bottleneck -- so the DataModule drives them with a
    BatchSampler and ``batch_size=None``.
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
        """Dataset indices -> flat grid indices. Overridden where rows are held out."""
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
    """Data-loss samples on the mesh vertices, carrying Vm and W targets.

    ``excluded_flat_indices`` holds the held-out test rows, skipped without
    materialising the complementary index array.
    """

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


# -----------------------------
# DataModule
# -----------------------------
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


    # Min/max tensors
    mins: Optional[torch.Tensor] = None
    maxes: Optional[torch.Tensor] = None

    # Geometry and point sets for the selected case
    case_data: Optional[CaseData] = None

    def _make_loader(
        self, dataset, batch_size: int, num_workers: int, replacement: bool = False
    ) -> DataLoader:
        """Loader that fetches one whole batch per dataset call.

        The sampler hands the dataset an array of indices and ``batch_size=None``
        disables auto-collation, so a batch is assembled with vectorised numpy
        instead of 4096 Python-level __getitem__ calls plus a collate. That is
        ~200x cheaper per batch and is what keeps the GPU, rather than the
        loader workers, as the bottleneck.
        """
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

        # Only the time normaliser is needed here; the PINN reads the rest of
        # MODEL.physics from the config itself.
        self.t_norm = float(self.config["MODEL"].get("physics", {}).get("t_norm", 12.9))

        cfgD = self.config["DATA"]
        slice_t = cfgD["sampling"]["slice_t"]
        self.num_domain = cfgD["sampling"]["num_domain"]
        self.num_boundary = cfgD["sampling"]["num_boundary"]
        self.num_init = cfgD["sampling"]["num_init"]
        ic_space_slice = cfgD["sampling"]["ic_space_slice"]
        ic_time_steps = cfgD["sampling"]["ic_time_steps"]

        batch_size = cfgD["loader"]["batch_size"]
        self.batch_size_lbfgs = cfgD["loader"]["batch_size_lbfgs"]
        num_workers = int(cfgD["loader"].get("num_workers", 4))

        umtomm = cfgD["constants"]["um_to_mm"]

        np.random.seed(42)

        # -----------------
        # Load geometry, point sets and fields for this case
        # -----------------
        self.case = resolve_case(self.config, self.case)
        dataset_path = resolve_dataset_path(self.config, self.case)
        data = load_case_data(self.case, dataset_path)
        self.case_data = data
        print(f"Dataset: {dataset_path}")
        print(f"Geometry: {data.summary()}")

        vertices = data.vertices          # (V, 2), already in mm
        vmark = data.vertex_markers

        # -----------------
        # Experiment selection
        # -----------------
        g_il_values = data.g_il
        sampled_indices = training_experiment_indices(len(g_il_values))

        g_train = g_il_values[sampled_indices]
        g_collocation = g_il_values

        # -----------------
        # Load time series as (n_exp, n_time, n_space)
        # -----------------
        vm_train, W_train = load_fields(dataset_path, sampled_indices, slice_t=slice_t)
        print(
            f"Fields: vm {vm_train.shape} "
            f"({vm_train.nbytes / 2**30:.2f} GiB x2), "
            f"experiments {np.round(g_train, 4).tolist()}"
        )

        t = time_axis(data.n_times, data.dt_ms, self.t_norm, slice_t)

        # -----------------
        # Spatial bases (data loss lives on the mesh vertices)
        # -----------------
        x = vertices[:, 0]
        y = vertices[:, 1]

        x_ic = x[::ic_space_slice]
        y_ic = y[::ic_space_slice]
        vmark_ic = vmark[::ic_space_slice]
        t_ic = t[:ic_time_steps]

        # -----------------
        # Collocation bases: triangle centroids. A centroid carries its own
        # triangle's exact region tag. Fibers remain a FiberNet approximation;
        # the original PINN uses the lazily assembled centroid/time/g grid.
        # -----------------
        x_inner = data.collocation_points[:, 0]
        y_inner = data.collocation_points[:, 1]
        vmark_inner = data.collocation_markers

        # -----------------
        # Boundary bases: edge centroids with outward normals. Includes the
        # interior loops left by the triangles removed from the scar, not just
        # the outer wall.
        # -----------------
        boundary_normals = data.boundary_normals
        vmark_boundary = data.boundary_markers
        x_boundary = data.boundary_points[:, 0]
        y_boundary = data.boundary_points[:, 1]

        # -----------------
        # Random sampled test set (excluded from train)
        # -----------------
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

        self.dataloader = self._make_loader(self.dataset, batch_size, num_workers)
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

        # Bases kept for resample(): rebuilding the boundary batch from these
        # is far cheaper than materialising every (point, time, g) feature row.
        self._boundary_base = dict(
            x=x_boundary,
            y=y_boundary,
            t=t,
            g_values=g_collocation,
            vmark=vmark_boundary,
            normals=boundary_normals,
        )

        # -----------------
        # Min/Max over feature columns
        # -----------------
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
    """Yield initialized modules one case at a time, without mutating config.

    Example: for dm in iter_case_data_modules(cfg, ["case_1", "case_2"]): ...
    Keeping every yielded module also retains every case's selected FEM arrays.
    """
    names = resolve_cases(config, cases=cases)
    paths = [resolve_dataset_path(config, name).resolve() for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple cases must resolve to different HDF5 files")
    for case in names:
        selected = deepcopy(config)
        selected.setdefault("DATA", {})["case"] = case
        module = ScarPINNDataModule(selected, case=case, device=device)
        module.setup_data()
        yield module
