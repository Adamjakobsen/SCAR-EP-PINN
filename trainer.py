
from __future__ import annotations

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scar_data_module import ScarPINNDataModule, CustomDataset


class PINNTrainer:
    """
    Orchestrates 3-phase training on a PINN model using YAML config.
    """

    def __init__(self, model,dm, config: dict):
        self.model = model
        self.config = config

        self.dm =dm

        # Will be populated by setup_data()
        self.dataset = None
        self.device = None

        # Optimizers & schedulers from config
        o1 = config["OPTIM"]["phase1"]
        self.optimizer_phase1 = optim.Adam(
            self.model.parameters(),
            lr=float(o1["lr"]),
            weight_decay=float(o1["weight_decay"])
        )
        self.scheduler_data = None
        if o1.get("scheduler", {}).get("name") == "ExponentialLR":
            self.scheduler_data = optim.lr_scheduler.ExponentialLR(
                self.optimizer_phase1, gamma=float(o1["scheduler"]["gamma"])
            )

        o2 = config["OPTIM"]["phase2"]
        self.optimizer_phase2 = optim.Adam(
            self.model.parameters(),
            lr=float(o2["lr"]),
            weight_decay=float(o2["weight_decay"])
        )
        self.scheduler_phase2 = None
        if o2.get("scheduler", {}).get("name") == "StepLR":
            self.scheduler_phase2 = optim.lr_scheduler.StepLR(
                self.optimizer_phase2,
                step_size=int(o2["scheduler"]["step_size"]),
                gamma=float(o2["scheduler"]["gamma"])
            )

        ol = config["OPTIM"]["lbfgs"]
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=float(ol["lr"]),
            max_iter=int(ol["max_iter"]),
            history_size=int(ol["history_size"]),
            line_search_fn=str(ol["line_search_fn"]),
            tolerance_grad=float(ol.get("tolerance_grad", 1e-7)),
            tolerance_change=float(ol.get("tolerance_change", 1e-9)),
        )
        self.scheduler_lbfgs = None
        if ol.get("scheduler", {}).get("name") == "ExponentialLR":
            self.scheduler_lbfgs = optim.lr_scheduler.ExponentialLR(
                self.optimizer_lbfgs, gamma=float(ol["scheduler"]["gamma"])
            )

        # bookkeeping
        self.iteration = 0
        self.losses = []

        # resampling cadence
        self.resample_adam_every = int(config["OPTIM"]["resampling"]["adam_every_batches"])
        self.resample_lbfgs_every = int(config["OPTIM"]["resampling"]["lbfgs_every_batches"])

        self.clip_grad_norm = float(o2.get("clip_grad_norm", 0.0))

        # where to save
        self.out_dir = str(config["PROJECT"]["path_directory"])
        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data acquisition 
    # -------------------------------------------------------------------------
    def setup_data(self) -> None:
        """
        Build all loaders/tensors and pass (mins, maxes) into the model's scaler.
        """
        
        

        self.device = self.dm.device
        self.dataloader = self.dm.dataloader
        self.dataset = self.dm.dataset

        self.X_collocation_train = self.dm.X_collocation_train
        self.X_boundary = self.dm.X_boundary
        self.X_normals = self.dm.X_normals

        self.X_ic_train = self.dm.X_ic_train
        self.V_ic_train = self.dm.V_ic_train
        self.W_ic_train = self.dm.W_ic_train

        self.X_test = self.dm.X_test
        self.V_test = self.dm.V_test
        self.W_test = self.dm.W_test

        self.batch_size_lbfgs = int(self.config["DATA"]["loader"]["batch_size_lbfgs"])

        
        self.model.to(self.device)


    @torch.no_grad()
    def resample(self):
        self.dm.resample()
        self.X_collocation_train = self.dm.X_collocation_train

    # -------------------------------------------------------------------------
    # Training phases
    # -------------------------------------------------------------------------
    def data_phase(self):
        n_epochs = int(self.config["OPTIM"]["phase1"]["epochs"])
        best_rel_l1_v = float("inf")

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Data Phase Epoch {epoch+1}/{n_epochs}", leave=False)
            for batch in pbar:
                X_data = batch["X"].to(self.device)
                V_data = batch["V"].to(self.device)
                W_data = batch["W"].to(self.device)

                self.optimizer_phase1.zero_grad()
                mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)
                loss = mse_v + 0.5 * mse_w
                loss.backward()
                self.optimizer_phase1.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                    mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)

                pbar.set_postfix({
                    "MSE_v": f"{mse_v.item():.4e}",
                    "MSE_w": f"{mse_w.item():.4e}",
                    "Rel L1 V": f"{rel_v.item():.4e}",
                    "Rel L1 W": f"{rel_w.item():.4e}",
                    "MAE V": f"{mae_v.item():.4e}",
                    "MAE W": f"{mae_w.item():.4e}",
                })

                self.losses.append([self.iteration, mse_v.item(), mse_w.item(),
                                    rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.iteration += 1

                del X_data, V_data, W_data

            epoch_loss /= max(1, len(self.dataloader))
            if self.scheduler_data is not None:
                self.scheduler_data.step()

            rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
            best_rel_l1_v = min(best_rel_l1_v, rel_v.item())
            print(f"[Data Phase] Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss (avg): {epoch_loss:.4e} | "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}")

    def adam_phase(self):
        n_epochs = int(self.config["OPTIM"]["phase2"]["epochs"])
        for epoch in range(n_epochs):
            epoch_loss_v = 0.0
            pbar = tqdm(self.dataloader, desc=f"Adam Phase Epoch {epoch+1}/{n_epochs}", leave=False)
            for b_idx, batch in enumerate(pbar, start=1):
                X_data = batch["X"].to(self.device)
                V_data = batch["V"].to(self.device)
                W_data = batch["W"].to(self.device)

                self.optimizer_phase2.zero_grad()

                pde_loss, ode_loss = self.model.pde2d_conductivities(self.X_collocation_train)
                bc_loss = self.model.bc_loss(self.X_boundary, self.X_normals)
                ic_v, ic_w = self.model.ic_loss(self.X_ic_train, self.V_ic_train, self.W_ic_train)
                mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)
                epoch_loss_v += mse_v.item()

                total = (self.model.w_pde * pde_loss +
                         self.model.w_ode * ode_loss +
                         self.model.w_bc * bc_loss +
                         self.model.w_ic * (ic_v + ic_w) +
                         self.model.w_data * (mse_v + mse_w))

                total.backward()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer_phase2.step()

                with torch.no_grad():
                    rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                    mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)

                pbar.set_postfix({
                    "PDE": f"{pde_loss.item():.2e}",
                    "ODE": f"{ode_loss.item():.2e}",
                    "BC": f"{bc_loss.item():.2e}",
                    "MSE_v": f"{mse_v.item():.2e}",
                    "MSE_w": f"{mse_w.item():.2e}",
                    "Rel L1 V": f"{rel_v.item():.2e}",
                    "Rel L1 W": f"{rel_w.item():.2e}",
                    "MAE V": f"{mae_v.item():.2e}",
                    "MAE W": f"{mae_w.item():.2e}",
                })

                if self.resample_adam_every > 0 and (b_idx % self.resample_adam_every == 0):
                    self.resample()

                self.losses.append([self.iteration, mse_v.item(), mse_w.item(),
                                    rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.iteration += 1

                del X_data, V_data, W_data

            if self.scheduler_phase2 is not None:
                self.scheduler_phase2.step()

            rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
            mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
            print(f"[Adam Phase] Epoch {epoch+1}/{n_epochs} | "
                  f"Epoch loss avg V: {epoch_loss_v / max(1, len(self.dataloader)):.4e}, "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}, "
                  f"MAE V: {mae_v.item():.4e}, MAE W: {mae_w.item():.4e}")

            # checkpoint
            path = os.path.join(self.out_dir, f"model_Adam_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)

        # final save after Adam
        path = os.path.join(self.out_dir, "model_Adam.pth")
        torch.save(self.model.state_dict(), path)
        np.save(os.path.join(self.out_dir, "losses.npy"), self.losses)
        print(f"Model saved to: {path}")

    def lbfgs_phase(self):
        n_epochs = int(self.config["OPTIM"]["lbfgs"]["epochs"])

        for epoch in range(n_epochs):
            pbar = tqdm(self.dataloader, desc=f"L-BFGS Phase Epoch {epoch+1}/{n_epochs}", leave=False)
            for b_idx, batch in enumerate(pbar, start=1):
                X_data = batch["X"].to(self.device)
                V_data = batch["V"].to(self.device)
                W_data = batch["W"].to(self.device)

                def closure():
                    self.optimizer_lbfgs.zero_grad()
                    pde_loss, ode_loss = self.model.pde2d_conductivities(self.X_collocation_train)
                    bc_loss = self.model.bc_loss(self.X_boundary, self.X_normals)
                    ic_v, ic_w = self.model.ic_loss(self.X_ic_train, self.V_ic_train, self.W_ic_train)
                    mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)

                    total = (self.model.w_pde * pde_loss +
                             self.model.w_ode * ode_loss +
                             self.model.w_bc * bc_loss +
                             self.model.w_ic * (ic_v + ic_w) +
                             self.model.w_data * (mse_v + mse_w))
                    total.backward()

                    with torch.no_grad():
                        mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
                        rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                    pbar.set_postfix({
                        "PDE": f"{pde_loss.item():.2e}",
                        "ODE": f"{ode_loss.item():.2e}",
                        "BC": f"{bc_loss.item():.2e}",
                        "MSE_v": f"{mse_v.item():.2e}",
                        "MSE_w": f"{mse_w.item():.2e}",
                        "MAE V": f"{mae_v.item():.2e}",
                        "MAE W": f"{mae_w.item():.2e}",
                    })
                    self.losses.append([self.iteration, mse_v.item(), mse_w.item(),
                                        rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                    return total

                if self.resample_lbfgs_every > 0 and (b_idx % self.resample_lbfgs_every == 0):
                    self.resample()

                self.iteration += 1
                self.optimizer_lbfgs.step(closure)

                del X_data, V_data, W_data

            if self.scheduler_lbfgs is not None:
                self.scheduler_lbfgs.step()

            with torch.no_grad():
                rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
            print(f"[L-BFGS Phase] Epoch {epoch+1}/{n_epochs} | "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}, "
                  f"MAE V: {mae_v.item():.4e}, MAE W: {mae_w.item():.4e}")

            # checkpoint
            path = os.path.join(self.out_dir, f"model_lbfgs_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)
            np.save(os.path.join(self.out_dir, "losses.npy"), self.losses)

    # -------------------------------------------------------------------------
    # High-level driver
    # -------------------------------------------------------------------------
    def train(self) -> None:
        self.setup_data()
        print("\n=== [Phase 1: Data Pretraining] ===")
        self.data_phase()

        print("\n=== [Phase 2: PDE + Data Training (Adam)] ===")
        self.adam_phase()

        # L-BFGS typically after Adam
        print("\n=== [Phase 3: PDE + Data Training (L-BFGS)] ===")
        self.dataloader_lbfgs = DataLoader(self.dataset, batch_size=self.batch_size_lbfgs, shuffle=True)
        self.lbfgs_phase()
