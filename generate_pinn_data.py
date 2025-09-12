
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
from scar_data_module import ScarPINNDataModule
import numpy as np
import torch

from pinn import PINN2D as PINN
from utils import get_mesh_data, get_boundary, load_config

# -----------------------------
# Constants & configuration
# -----------------------------
G_IL_VALUES = np.array([
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
    0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1
])

VM_PATH = Path("./data/vm_variation_scar_full.npy")            # (T, V, E) where T timepoints, V vertices, E experiments
COND_PATH = Path("./data/conductivities_variation_scar_full.npy")
SAVE_ACT_FEM = Path("./data/activations_all_FEM.npy")
SAVE_ACT_PINN = Path("./data/activations_all_PINN.npy")
SAVE_V_PINN = Path("./data/vm_PINN.npy")


DT_MS = 5.0                    # time step in ms
TIME_NORMALIZER = 12.9         # Time Units (TU) in ms, as in Aliev-Panfilov model
COND_SCALE = 12.9 / 1.4        # to convert conductivities from S/m to mm^2/TU 
XY_MICRO_TO_MILLI = 1e-3
ACTIVATION_THRESHOLD = -20.0   # mV

VPRED_BATCH_SIZE = 1000

@dataclass
class Geometry:
    vertices: np.ndarray  # (V, 2)
    triangles: np.ndarray  # (T, 3) 


# -----------------------------
# Data loading helpers
# -----------------------------

def load_geometry() -> Geometry:
    vertices, triangles, _ = get_mesh_data()
    vertices = vertices * XY_MICRO_TO_MILLI
    return Geometry(vertices=vertices, triangles=triangles)


def load_vm_and_conductivities() -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-map the big arrays instead of loading them fully into RAM.
    This avoids copying a huge (T, V, E) array into system memory.
    """
    # memory-map (read-only) the big numpy files
    vm_all = np.load(VM_PATH, mmap_mode="r")                 # shape (T, V, E) but not loaded into RAM
    conductivities_all = np.load(COND_PATH, mmap_mode="r")   # shape (V, 2, 2, E)
    # cast conductivities to float32 and apply scale (this will create a small in-RAM array per-used-slice later)
    conductivities_all = (conductivities_all * COND_SCALE).astype(np.float32, copy=False)
    return vm_all, conductivities_all

#def load_vm_and_conductivities() -> Tuple[np.ndarray, np.ndarray]:
#    vm_all = np.load(VM_PATH)  # shape (T, V, E)
#    conductivities_all = np.load(COND_PATH) * COND_SCALE  # shape (V, 2, 2, E)
#    return vm_all, conductivities_all


def build_time_vector(n_timepoints: int) -> np.ndarray:
    t_start = 0.0
    t_end = (n_timepoints - 1) * DT_MS
    # normalized time axis, as in original code
    return np.linspace(t_start, t_end, n_timepoints) / TIME_NORMALIZER


# -----------------------------
# Activation time computation
# -----------------------------

def fem_activation_times(vm_over_time: np.ndarray, t: np.ndarray, threshold: float) -> np.ndarray:
    """
    Memory-safe computation of per-vertex activation times from FEM voltage traces.
    vm_over_time: (T, V)  -- memory-mapped is fine
    t: normalized time vector of length T
    returns: (V,) activation times in original time units (multiplied by TIME_NORMALIZER)
    """
    T, V = vm_over_time.shape
    act_times = np.full(V, np.nan, dtype=float)
    not_activated = np.ones(V, dtype=bool)

    # scan forward in time, mark activation time for vertices that cross threshold
    for ti in range(T):
        if not not_activated.any():
            break
        v_row = vm_over_time[ti, :]               # this reads a single time slice (V,) from memmap — minimal memory
        crossing = (v_row > threshold) & not_activated
        if crossing.any():
            act_times[crossing] = t[ti] * TIME_NORMALIZER
            not_activated[crossing] = False

    return act_times


def pinn_activation_times(
    pinn: PINN,
    device: torch.device,
    base_features: np.ndarray,
    t: np.ndarray,
    threshold: float,
    batch_size: int = 200_000,
) -> np.ndarray:
    """
    Compute per-vertex activation times from the PINN by scanning forward in time
    until threshold crossing. Uses batched inference across vertices.

    base_features: (V, 6) array with columns [x, y, sigma_a, sigma_b, sigma_c, sigma_d]
    t: normalized time vector of length T
    returns: (V,) activation times in original units
    """
    V = base_features.shape[0]
    act_times = np.full(V, np.nan, dtype=float)
    not_activated = np.ones(V, dtype=bool)

    with torch.no_grad():
        for t_current in t:
            if not not_activated.any():
                break
            # Prepare the batch for all currently not-activated vertices
            idx = np.where(not_activated)[0]
            n_curr = idx.size
            # Construct [x, y, t, sigma_a, sigma_b, sigma_c, sigma_d, exp]
            t_col = np.full((n_curr, 1), t_current, dtype=np.float32)
            data = np.hstack((base_features[idx, :2], t_col, base_features[idx, 2:]))
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

            v_pred = predict_in_batches(pinn, data, device, batch_size=VPRED_BATCH_SIZE)
            #v_pred = pinn.predict(data_tensor)[:, 0].detach().cpu().numpy()

            newly_activated = v_pred > threshold
            if newly_activated.any():
                act_idx = idx[newly_activated]
                act_times[act_idx] = t_current * TIME_NORMALIZER
                not_activated[act_idx] = False

    return act_times


def get_activation_data(
    pinn: PINN,
    experiments: Iterable[int],
    experiment_path: str | Path,
) -> None:
    """Compute and save FEM and PINN activation time maps for the given experiments."""
    geom = load_geometry()
    vm_all, cond_all = load_vm_and_conductivities()

    device = pinn.device
    x = geom.vertices[:, 0]
    y = geom.vertices[:, 1]

    n_timepoints, n_vertices, _ = vm_all.shape
    t = build_time_vector(n_timepoints)

    activations_all_FEM = np.zeros((n_vertices, len(G_IL_VALUES)), dtype=float)
    activations_all_PINN = np.zeros((n_vertices, len(G_IL_VALUES)), dtype=float)
    
    for i, exp_idx in enumerate(experiments):
        print(f"Experiment {i} (index={exp_idx})")
        # (V, 2, 2) for this experiment
        conductivities = cond_all[:, :, :, exp_idx]
        sigma_a = conductivities[:, 0, 0]
        sigma_b = conductivities[:, 0, 1]
        sigma_c = conductivities[:, 1, 0]
        sigma_d = conductivities[:, 1, 1]

        # FEM activation times
        fem_act = fem_activation_times(vm_all[:, :, exp_idx], t, ACTIVATION_THRESHOLD)
        activations_all_FEM[:, exp_idx] = fem_act

        # Base spatial features for PINN (without time and exp columns)
        base = np.column_stack((x, y, sigma_a, sigma_b, sigma_c, sigma_d)).astype(np.float32)
        # Append experiment indicator as an extra column to base_features 
        exp_col = np.full((n_vertices, 1), G_IL_VALUES[exp_idx], dtype=np.float32)
        base_with_exp = np.hstack((base, exp_col))  
        def pinn_act_times_for_exp() -> np.ndarray:
            act_times = np.full(n_vertices, np.nan, dtype=float)
            not_activated = np.ones(n_vertices, dtype=bool)
            with torch.no_grad():
                for t_current in t:
                    if not not_activated.any():
                        break
                    idx = np.where(not_activated)[0]
                    n_curr = idx.size
                    t_col = np.full((n_curr, 1), t_current, dtype=np.float32)
                    exp_col_batch = np.full((n_curr, 1), G_IL_VALUES[exp_idx], dtype=np.float32)
                    data = np.hstack((base[idx, :2], t_col, base[idx, 2:], exp_col_batch))
                    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
                    #v_pred = pinn.predict(data_tensor)[:, 0].detach().cpu().numpy()
                    v_pred = predict_in_batches(pinn, data, device, batch_size=VPRED_BATCH_SIZE)

                    newly_activated = v_pred > ACTIVATION_THRESHOLD
                    if newly_activated.any():
                        act_idx = idx[newly_activated]
                        act_times[act_idx] = t_current * TIME_NORMALIZER
                        not_activated[act_idx] = False
            return act_times

        pinn_act = pinn_act_times_for_exp()
        activations_all_PINN[:, exp_idx] = pinn_act

        # Relative L1 error (only where both defined)
        valid = ~np.isnan(fem_act) & ~np.isnan(pinn_act)
        if valid.any():
            l1_error = np.sum(np.abs(pinn_act[valid] - fem_act[valid])) / np.sum(np.abs(fem_act[valid]))
            print(f"=========== Experiment {exp_idx} L1 error: {l1_error:.4f} ============")
        else:
            print(f"=========== Experiment {exp_idx} L1 error: NaN (no valid points) ============")

    np.save(SAVE_ACT_FEM, activations_all_FEM)
    np.save(SAVE_ACT_PINN, activations_all_PINN)


def predict_in_batches(pinn, data: np.ndarray, device: torch.device, batch_size: int = 10000) -> np.ndarray:
    preds = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = pinn.predict(batch_tensor)[:, 0].detach().cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)

def get_v_data(pinn: PINN) -> np.ndarray:
    """Compute full PINN voltage predictions V_PINN with shape (T, V, E)."""
    geom = load_geometry()
    vm_all, cond_all = load_vm_and_conductivities()

    device = pinn.device
    x = geom.vertices[:, 0]
    y = geom.vertices[:, 1]

    T_len, V_count, E_count = vm_all.shape
    t = build_time_vector(T_len)

    #V_PINN = np.zeros_like(vm_all)
    V_PINN = np.memmap(SAVE_V_PINN, dtype=np.float32, mode="w+", shape=vm_all.shape)

    with torch.no_grad():
        for exp_idx, g_il in enumerate(G_IL_VALUES):
            print(f"Experiment {exp_idx}")
            conductivities = cond_all[:, :, :, exp_idx]
            sigma_a = conductivities[:, 0, 0]
            sigma_b = conductivities[:, 0, 1]
            sigma_c = conductivities[:, 1, 0]
            sigma_d = conductivities[:, 1, 1]

            # Base per-vertex features
            base = np.column_stack((x, y, sigma_a, sigma_b, sigma_c, sigma_d)).astype(np.float32)

            for ti, t_curr in enumerate(t):
                t_col = np.full((V_count, 1), t_curr, dtype=np.float32)
                exp_col = np.full((V_count, 1), g_il, dtype=np.float32)
                data = np.hstack((base[:, :2], t_col, base[:, 2:], exp_col))  # (V, 8)
                data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
                v_pred = predict_in_batches(pinn, data, device, batch_size=VPRED_BATCH_SIZE)
                #v_pred = pinn.predict(data_tensor)[:, 0].detach().cpu().numpy()
                V_PINN[ti, :, exp_idx] = v_pred

            # Optional diagnostics
            v_true = vm_all[:, :, exp_idx]
            abs_sum = np.sum(np.abs(v_true - V_PINN[:, :, exp_idx]))
            v_sum = np.sum(np.abs(v_true))
            if v_sum > 0:
                print("L1 relative error:", abs_sum / v_sum)

    return V_PINN

def compute_minmax_from_vm(vm_path: Path, sample_stride: int = 1) -> Tuple[float, float]:
    """
    Compute approximate global min/max of the VM memmap by scanning time-slices.
    sample_stride: read every `sample_stride`-th timepoint (use >1 to speed up).
    Returns (min_val, max_val) as Python floats.
    This never loads the whole array into RAM.
    """
    vm_mm = np.load(vm_path, mmap_mode="r")   # (T, V, E)
    T = vm_mm.shape[0]
    global_min = float("inf")
    global_max = float("-inf")
    # scan time slices, optionally subsample timepoints to speed up
    for ti in range(0, T, sample_stride):
        # read the entire slice for all vertices & experiments at time ti -> shape (V, E)
        slice_ = vm_mm[ti, :, :]
        # compute min/max on that slice (cheap)
        smin = float(np.min(slice_))
        smax = float(np.max(slice_))
        if smin < global_min:
            global_min = smin
        if smax > global_max:
            global_max = smax
    # fallback if file empty/unexpected
    if global_min == float("inf"):
        global_min, global_max = 0.0, 1.0
    return global_min, global_max

if __name__ == "__main__":
    experiment_path = Path("./experiments/run_20250909_193150")
    model_path = experiment_path / Path("model_Adam_epoch_4.pth")
    config_path = experiment_path / "config.yaml"
    config = load_config(str(config_path))

    # --- Compute min/max from vm file with memmap (no full load, low RAM) ---
    print("Computing min/max from VM memmap (this will scan time slices)...")
    vm_min, vm_max = compute_minmax_from_vm(VM_PATH, sample_stride=1)  # you can increase stride to 2 or 5 if needed
    # Convert to torch tensors like your DataModule would (float32)
    
    input_dim = config.get("input_dim", 8)  # fallback to 8 if not in config
    min_tensor = torch.full((input_dim,), vm_min, dtype=torch.float32)
    max_tensor = torch.full((input_dim,), vm_max, dtype=torch.float32)

    #print("VM min/max:", float(min_tensor), float(max_tensor))

    # instantiate model (use CPU device for loading, then move to CUDA)
    pinn = PINN(config, min_tensor, max_tensor)

    # Detect device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    # load state dict to CPU first to avoid accidental GPU RAM spikes in torch.load
    state_dict = torch.load(str(model_path), map_location="cpu")
    pinn.load_state_dict(state_dict)
    pinn.to(device)
    pinn.eval()
    # optional: use fp16 if your model works in half precision
    # pinn.half()
    # then ensure predict uses float16 tensors

    # small safety: clear state_dict from RAM
    del state_dict
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # run on a subset of experiments first for debugging (set to all later)
    experiments = np.arange(0, len(G_IL_VALUES), dtype=int)

    # Run activation and v prediction
    get_activation_data(pinn, experiments, experiment_path)

    v_pred_all = get_v_data(pinn)
    if isinstance(v_pred_all, np.memmap):
        v_pred_all.flush()

    # Final cleanup
    del v_pred_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()