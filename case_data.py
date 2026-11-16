from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

SUPPORTED_CASES = ("case_1", "case_2", "case_3")
DEFAULT_CASE = "case_1"
XY_MICRO_TO_MILLI = 1e-3
SCAR_TAG = 3


def resolve_case(config: dict, case: Optional[str] = None) -> str:
    name = case if case is not None else config.get("DATA", {}).get("case") or DEFAULT_CASE
    if name not in SUPPORTED_CASES:
        raise ValueError(f"Unknown case {name!r}; choose from {', '.join(SUPPORTED_CASES)}")
    return name


def resolve_cases(config: dict, *, case=None, cases=None) -> list[str]:
    """Return the selected cases in order."""
    if case is not None and cases is not None:
        raise ValueError("Choose case or cases, not both")
    if case is not None:
        selected = [case]
    elif cases is not None:
        selected = cases
    else:
        selected = config.get("DATA", {}).get("cases")
        if selected is None:
            selected = [resolve_case(config)]
    if isinstance(selected, str):
        selected = [selected]
    names = [resolve_case(config, item) for item in selected]
    if not names or len(set(names)) != len(names):
        raise ValueError("Cases must be nonempty and unique")
    return names


def fibernet_checkpoint_path(config: dict, case: str) -> Path:
    name = resolve_case(config, case)
    template = str(config.get("DATA", {}).get(
        "fibernet_checkpoint", "./experiments/fibernet/{case}/fibernet.pth"))
    if "{case}" not in template:
        raise ValueError("DATA.fibernet_checkpoint must include {case} to isolate geometries")
    return Path(template.format(case=name))


def case_dataset_path(case: str) -> Path:
    return Path("data") / f"scar_ep_{case}.h5"


@dataclass
class CaseData:
    """Mesh and simulation metadata for one case."""

    case: str
    path: Path

    # Mesh (mm)
    vertices: np.ndarray        # (V, 2)
    topology: np.ndarray        # (E, 3)
    element_markers: np.ndarray  # (E,)
    element_fibres: np.ndarray  # (E, 2) unit
    vertex_markers: np.ndarray  # (V,)

    # Collocation = triangle centroids (mm)
    collocation_points: np.ndarray   # (E, 2)
    collocation_markers: np.ndarray  # (E,)
    collocation_fibres: np.ndarray   # (E, 2) unit

    # Boundary = edge centroids (mm)
    boundary_edges: np.ndarray       # (Nb, 2) vertex indices, for drawing outlines
    boundary_components: np.ndarray  # (Nb,) connected boundary component id
    boundary_points: np.ndarray   # (Nb, 2)
    boundary_normals: np.ndarray  # (Nb, 2) unit, outward from tissue
    boundary_markers: np.ndarray  # (Nb,)
    boundary_fibres: np.ndarray   # (Nb, 2) unit

    # Sweep
    g_il: np.ndarray     # (S,)
    time_ms: np.ndarray  # (T,)
    dt_ms: float

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_collocation(self) -> int:
        return len(self.collocation_points)

    @property
    def n_boundary(self) -> int:
        return len(self.boundary_points)

    @property
    def n_sims(self) -> int:
        return len(self.g_il)

    @property
    def n_times(self) -> int:
        return len(self.time_ms)

    def summary(self) -> str:
        return (
            f"{self.case}: {self.n_vertices} vertices, "
            f"{self.n_collocation} collocation points, "
            f"{self.n_boundary} boundary points, "
            f"{self.n_sims} sims x {self.n_times} frames"
        )


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length, leaving zero rows untouched."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, 1e-12, None)


def load_case_data(
    case: str, path: str | Path, dtype: Any = np.float32
) -> CaseData:
    """Load mesh metadata, converting coordinates from um to mm."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No dataset for {case} at {path}. Build it with "
            f"`python build_dataset.py --cases {case}` and add the boundary with "
            f"`python boundary_normals.py --cases {case}`."
        )

    with h5py.File(path, "r") as fh:
        for required in ("vm", "w", "boundary_normals", "boundary_elements"):
            if required not in fh:
                raise KeyError(
                    f"{path} is missing '{required}'. Re-run build_dataset.py and "
                    f"boundary_normals.py for {case}."
                )

        vertices = fh["coordinates"][:, :2] * XY_MICRO_TO_MILLI
        topology = fh["topology"][:]
        element_markers = fh["element_markers"][:]
        element_fibres = _unit(fh["fibres"][:, :2])
        vertex_markers = fh["vertex_markers"][:]

        boundary_elements = fh["boundary_elements"][:]
        boundary_edges = fh["boundary_edges"][:]
        boundary_components = fh["boundary_components"][:]
        boundary_points = fh["boundary_edge_centroids"][:, :2] * XY_MICRO_TO_MILLI
        boundary_normals = fh["boundary_normals"][:, :2]
        boundary_markers = fh["boundary_markers"][:]

        g_il = fh["g_il"][:]
        time_ms = fh["time_ms"][:]
        dt_ms = float(fh.attrs["dt_ms"])

    # Each centroid uses its triangle's fibre and tag.
    collocation_points = vertices[topology].mean(axis=1)

    return CaseData(
        case=case,
        path=path,
        vertices=vertices.astype(dtype),
        topology=topology.astype(np.int64),
        element_markers=element_markers.astype(dtype),
        element_fibres=element_fibres.astype(dtype),
        vertex_markers=vertex_markers.astype(dtype),
        collocation_points=collocation_points.astype(dtype),
        collocation_markers=element_markers.astype(dtype),
        collocation_fibres=element_fibres.astype(dtype),
        boundary_edges=boundary_edges.astype(np.int64),
        boundary_components=boundary_components.astype(np.int64),
        boundary_points=boundary_points.astype(dtype),
        boundary_normals=boundary_normals.astype(dtype),
        boundary_markers=boundary_markers.astype(dtype),
        boundary_fibres=element_fibres[boundary_elements].astype(dtype),
        g_il=g_il.astype(dtype),
        time_ms=time_ms.astype(dtype),
        dt_ms=dt_ms,
    )


def load_field(
    path: str | Path,
    name: str,
    sim_indices: np.ndarray,
    slice_t: int = 1,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Read vm or w in (simulation, time, vertex) order."""
    sim_indices = np.asarray(sim_indices, dtype=np.int64)
    if sim_indices.size and np.any(np.diff(sim_indices) <= 0):
        raise ValueError("sim_indices must be strictly increasing for HDF5 reads")
    with h5py.File(path, "r") as fh:
        return fh[name][sim_indices, ::slice_t, :].astype(dtype, copy=False)


def load_fields(
    path: str | Path,
    sim_indices: np.ndarray,
    slice_t: int = 1,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        load_field(path, "vm", sim_indices, slice_t, dtype),
        load_field(path, "w", sim_indices, slice_t, dtype),
    )


def load_case(config: dict, case: Optional[str] = None) -> CaseData:
    name = resolve_case(config, case)
    return load_case_data(name, case_dataset_path(name))


def fibernet_training_arrays(data: CaseData):
    """Return centroid coordinates, fibre targets, and coordinate bounds."""
    X = np.concatenate((data.collocation_points, data.boundary_points), axis=0).astype(np.float32)
    y = np.concatenate((data.collocation_fibres, data.boundary_fibres), axis=0).astype(np.float32)
    y = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-8, None)
    return X, y, np.min(X, axis=0), np.max(X, axis=0)
