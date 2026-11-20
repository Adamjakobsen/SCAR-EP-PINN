
from __future__ import annotations

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
from scar_data_module import ScarPINNDataModule, CustomDataset
from fibernet import FiberNet
from case_data import fibernet_training_arrays, fibernet_checkpoint_path
import matplotlib.pyplot as plt


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
                self.optimizer_phase1, gamma=float(o1["scheduler"]["gamma"],step_size=int(o1["scheduler"]["step_size"]))
            )
        elif o1.get("scheduler", {}).get("name") == "StepLR":
            self.scheduler_data = optim.lr_scheduler.StepLR(
                self.optimizer_phase1,
                step_size=int(o1["scheduler"]["step_size"]),
                gamma=float(o1["scheduler"]["gamma"])
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
        self.losses_test = []
        self.losses_phys = []

        # resampling cadence
        self.resample_adam_every = int(config["OPTIM"]["resampling"]["adam_every_batches"])
        self.resample_lbfgs_every = int(config["OPTIM"]["resampling"]["lbfgs_every_batches"])

        self.clip_grad_norm = float(o2.get("clip_grad_norm", 0.0))

        # where to save
        self.out_dir = str(config["PROJECT"]["path_directory"])
        print(f"Outputs (models, losses) will be saved to: {self.out_dir}")
        os.makedirs(self.out_dir, exist_ok=True)
    

    
    def build_fibernet(self):
        X,y,mins,maxs = fibernet_training_arrays(self.dm.case_data)
        fibernet = FiberNet(
            input_dim=2,  # x, y
            output_dim=2, # e_fib_x, e_fib_y
            hidden_layers=3,
            hidden_units=512,
            mins=mins,
            maxs=maxs
        ).to(self.device)
        checkpoint = fibernet_checkpoint_path(self.config, self.dm.case_data.case)
        state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        # Check the checkpoint's coordinate scaling against this case.
        for key in ("layer_norm.weight", "layer_norm.bias"):
            if key not in state_dict or not torch.equal(
                    state_dict[key].detach().cpu(), fibernet.state_dict()[key].detach().cpu()):
                raise ValueError(f"FiberNet checkpoint scaling does not match {self.dm.case_data.case}: {checkpoint}")
        fibernet.load_state_dict(state_dict)
        fibernet.eval()
        for p in fibernet.parameters():
            p.requires_grad = False
        print(f"Loaded {self.dm.case_data.case} FiberNet from {checkpoint}")
        return fibernet

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

        self.collocation_dataloader = self.dm.collocation_dataloader
        self.collocation_dataset = self.dm.collocation_dataset
        self.collocation_iter = itertools.cycle(self.collocation_dataloader)

        self.boundary_dataloader = self.dm.boundary_dataloader
        self.boundary_dataset = self.dm.boundary_dataset
        self.boundary_iter = itertools.cycle(self.boundary_dataloader)

        #self.X_collocation_train = self.dm.X_collocation_train
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
        self.fibernet=self.build_fibernet()



    @torch.no_grad()
    def resample(self):
        self.dm.resample()
        self.X_boundary = self.dm.X_boundary
        self.X_normals = self.dm.X_normals
        #self.X_collocation_train = self.dm.X_collocation_train

    def _plot_loss_array(self, arr, labels, filename, title, logy=True, column_order=None):
        if arr is None or len(arr) == 0:
            return

        data = np.asarray(arr, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 2:
            return

        x = data[:, 0]

        if column_order is None:
            column_order = list(range(1, min(data.shape[1], len(labels) + 1)))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for col_idx in column_order:
            label_idx = col_idx - 1
            if col_idx >= data.shape[1] or label_idx >= len(labels):
                continue

            y = data[:, col_idx]
            if logy:
                y = np.clip(y, 1e-16, None)
                ax.semilogy(x, y, label=labels[label_idx], linewidth=1.2)
            else:
                ax.plot(x, y, label=labels[label_idx], linewidth=1.2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss / metric")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=150)
        plt.close(fig)


    def save_loss_plots(self):
        self._plot_loss_array(
            self.losses,
            labels=["mse_v", "mse_w", "rel_v", "rel_w", "mae_v", "mae_w"],
            filename="losses.png",
            title="Training losses / metrics",
            logy=True,
        )

        self._plot_loss_array(
            self.losses_test,
            labels=["mse_v_test", "mse_w_test", "rel_v", "rel_w", "mae_v", "mae_w"],
            filename="losses_test.png",
            title="Test losses / metrics",
            logy=True,
        )

        self._plot_loss_array(
            self.losses_phys,
            labels=["pde", "ode", "bc", "ic_v", "ic_w"],
            filename="losses_phys.png",
            title="Physics losses",
            logy=True,
            column_order=[3, 4, 5, 1, 2]
        )
    
        



    # -------------------------------------------------------------------------
    # Training phases
    # -------------------------------------------------------------------------
    def data_phase(self):
        phase1_config = self.config["OPTIM"]["phase1"]
        n_epochs = int(phase1_config["epochs"])
        target_updates = phase1_config.get("updates")
        if target_updates is not None:
            target_updates = int(target_updates)
            if target_updates <= 0:
                raise ValueError("OPTIM.phase1.updates must be a positive integer")

        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch <= 0:
            raise ValueError("Phase 1 requires a non-empty data loader")
        if target_updates is not None:
            n_epochs = (target_updates + steps_per_epoch - 1) // steps_per_epoch
        phase_updates = 0
        RL1_best = float("inf")
        
        for epoch in range(n_epochs):
            
            epoch_loss = 0.0
            steps_this_epoch = steps_per_epoch
            if target_updates is not None:
                steps_this_epoch = min(steps_per_epoch, target_updates - phase_updates)
            batches = itertools.islice(self.dataloader, steps_this_epoch)
            pbar = tqdm(
                batches,
                total=steps_this_epoch,
                desc=f"Data Phase Epoch {epoch+1}/{n_epochs}",
                leave=False,
            )
            for batch in pbar:
                X_data = batch["X"].to(self.device)
                V_data = batch["V"].to(self.device)
                W_data = batch["W"].to(self.device)

                self.optimizer_phase1.zero_grad()
                mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)
                loss = mse_v + 0.5 * mse_w
                loss.backward()
                self.optimizer_phase1.step()
                #current lr from scheduler
                lr = self.scheduler_data.get_last_lr()[0]
                if lr>6e-5:
                    self.scheduler_data.step()

                
                epoch_loss += loss.item()
                
                with torch.no_grad():
                    rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                    mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
                    mse_v_test, mse_w_test = self.model.data_loss(self.X_test, self.V_test, self.W_test)
                    self.losses_test.append([self.iteration, mse_v_test.item(), mse_w_test.item(),
                                            rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                if rel_v.item()<RL1_best:
                    RL1_best=rel_v.item()
                    #Save best model
                    path = os.path.join(self.out_dir, f"model_data_best.pth")
                    torch.save(self.model.state_dict(), path)
                pbar.set_postfix({
                    "LR": f"{lr:.2e}",
                    "MSE_v": f"{mse_v.item():.4e}",
                    #"MSE_V_t": f"{mse_v_test.item():.4e}",
                    "MSE_w": f"{mse_w.item():.4e}",
                    "Rel L1 V": f"{rel_v.item():.4e}",
                    #"Rel L1 W": f"{rel_w.item():.4e}",
                    "MAE V": f"{mae_v.item():.4e}",
                    #"MAE W": f"{mae_w.item():.4e}",
                })

                self.losses.append([self.iteration, mse_v.item(), mse_w.item(),
                                    rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.iteration += 1
                phase_updates += 1

                del X_data, V_data, W_data

                
            epoch_loss /= max(1, steps_this_epoch)

            rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
            RL1_best = min(RL1_best, rel_v.item())
            print(f"[Data Phase] Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss (avg): {epoch_loss:.4e} | "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}")
            # checkpoint
            path = os.path.join(self.out_dir, f"model_data_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)
            np.save(os.path.join(self.out_dir, "losses.npy"), self.losses)
            np.save(os.path.join(self.out_dir, "losses_test.npy"), self.losses_test)
        print(f"[Data Phase] Completed {phase_updates} optimizer updates.")

    def adam_phase(self):
        phase2_config = self.config["OPTIM"]["phase2"]
        n_epochs = int(phase2_config["epochs"])
        target_updates = phase2_config.get("updates")
        if target_updates is not None:
            target_updates = int(target_updates)
            if target_updates <= 0:
                raise ValueError("OPTIM.phase2.updates must be a positive integer")

        steps_per_epoch = max(
            len(self.dataloader),
            len(self.collocation_dataloader),
            len(self.boundary_dataloader),
        )
        if steps_per_epoch <= 0:
            raise ValueError("Phase 2 requires non-empty data, collocation, and boundary loaders")
        if target_updates is not None:
            n_epochs = (target_updates + steps_per_epoch - 1) // steps_per_epoch
        phase_updates = 0
  
        #Physics losses


        RL1_best = float("inf")

        for epoch in range(n_epochs):

            epoch_loss_v = 0.0
            steps_this_epoch = steps_per_epoch
            if target_updates is not None:
                steps_this_epoch = min(steps_per_epoch, target_updates - phase_updates)

            data_iter = itertools.cycle(self.dataloader)
            collocation_iter = itertools.cycle(self.collocation_dataloader)
            boundary_iter = itertools.cycle(self.boundary_dataloader)

            pbar = tqdm(
                range(steps_this_epoch),
                total=steps_this_epoch,
                desc=f"Adam Phase Epoch {epoch+1}/{n_epochs}",
                leave=False,
            )

            
            for b_idx, _ in enumerate(pbar, start=1):
                batch = next(data_iter)
                collocation_batch = next(collocation_iter)
                boundary_batch = next(boundary_iter)

                non_blocking = (self.device.type == "cuda")

                X_data = batch["X"].to(self.device, non_blocking=non_blocking)
                V_data = batch["V"].to(self.device, non_blocking=non_blocking)
                W_data = batch["W"].to(self.device, non_blocking=non_blocking)

                X_collocation = collocation_batch["X"].to(self.device, non_blocking=non_blocking)

                Xb = boundary_batch["X"].to(self.device, non_blocking=non_blocking)
                normals = boundary_batch["X_normals"].to(self.device, non_blocking=non_blocking)


                self.optimizer_phase2.zero_grad()
                pde_loss, ode_loss = self.model.pde2d_conductivities(X_collocation,self.fibernet)

                bc_loss = self.model.bc_loss(Xb, normals,self.fibernet)
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
                self.scheduler_phase2.step()
                lr = self.scheduler_phase2.get_last_lr()[0]

                with torch.no_grad():
                    rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
                    mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
                    mse_v_test, mse_w_test = self.model.data_loss(self.X_test, self.V_test, self.W_test)
                self.losses_test.append([self.iteration, mse_v_test.item(), mse_w_test.item(),
                                             rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.losses_phys.append([self.iteration, pde_loss.item(), ode_loss.item(), bc_loss.item(),
                                        ic_v.item(), ic_w.item()])
                pbar.set_postfix({
                    "LR": f"{lr:.2e}",
                    "PDE": f"{pde_loss.item():.2e}",
                    "ODE": f"{ode_loss.item():.2e}",
                    "BC": f"{bc_loss.item():.2e}",
                    "MSE_v": f"{mse_v.item():.2e}",
                    "MSE_v_t": f"{mse_v_test.item():.4e}",
                    "MSE_w": f"{mse_w.item():.2e}",
                    "Rel L1 V": f"{rel_v.item():.2e}",
                    "Rel L1 W": f"{rel_w.item():.2e}",
                    "MAE V": f"{mae_v.item():.2e}",
                    "MAE W": f"{mae_w.item():.2e}",
                })


                self.losses.append([self.iteration, mse_v.item(), mse_w.item(),
                                    rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])

                self.iteration += 1
                phase_updates += 1

                del X_data, V_data, W_data, X_collocation
                

                
            if rel_v.item()<RL1_best:
                RL1_best=rel_v.item()
                #Save best model
                path = os.path.join(self.out_dir, f"model_best.pth")
                torch.save(self.model.state_dict(), path)

            rel_v, rel_w = self.model.eval_rel_l1(self.X_test, self.V_test, self.W_test)
            mae_v, mae_w = self.model.eval_mae(self.X_test, self.V_test, self.W_test)
            print(f"[Adam Phase] Epoch {epoch+1}/{n_epochs} | "
                  f"Epoch loss avg V: {epoch_loss_v / max(1, len(self.dataloader)):.4e}, "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}, "
                  f"MAE V: {mae_v.item():.4e}, MAE W: {mae_w.item():.4e}")
            # checkpoint
            path = os.path.join(self.out_dir, f"model_Adam_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)
            np.save(os.path.join(self.out_dir, "losses.npy"), self.losses)
            np.save(os.path.join(self.out_dir, "losses_test.npy"), self.losses_test)
            np.save(os.path.join(self.out_dir, "losses_phys.npy"), self.losses_phys)
        print(f"[Adam Phase] Completed {phase_updates} optimizer updates.")
        #run.finish()
        




    def train(self) -> None:
        self.setup_data()
        phases = self.config["TRAINER"].get("phases", ["phase1", "phase2"])

        if "phase1" in phases:
            print("\n=== [Phase 1: Data Pretraining] ===")
            self.data_phase()

        if "phase2" in phases:
            print("\n=== [Phase 2: PDE + Data Training (Adam)] ===")
            self.adam_phase()


    
    def train_from_checkpoint(self, checkpoint_path: str,phases: list) -> None:
        """
        Load model weights from checkpoint and continue training from there.
        """

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Loaded model weights from {checkpoint_path}")
        # Get losses.npy,losses_phys.npy and losses_test.npy from checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_dir, "losses.npy")):
            print("Loading previous losses from", os.path.join(checkpoint_dir, "losses.npy"))
            self.losses = np.load(os.path.join(checkpoint_dir, "losses.npy")).tolist()
        if os.path.isfile(os.path.join(checkpoint_dir, "losses_test.npy")):
            print("Loading previous test losses from", os.path.join(checkpoint_dir, "losses_test.npy"))
            self.losses_test = np.load(os.path.join(checkpoint_dir, "losses_test.npy")).tolist()
        if os.path.isfile(os.path.join(checkpoint_dir, "losses_phys.npy")):
            print("Loading previous physics losses from", os.path.join(checkpoint_dir, "losses_phys.npy"))
            self.losses_phys = np.load(os.path.join(checkpoint_dir, "losses_phys.npy")).tolist()
        self.setup_data()
        if "phase1" in phases:
            print("\n=== [Continuing Phase 1: Data Pretraining] ===")
            self.data_phase()
        if "phase2" in phases:
            print("\n=== [Continuing Phase 2: PDE + Data Training (Adam)] ===")
            self.adam_phase()


    def sigmoid(x,a, b,iterations_start):
        iterations_per_epoch = 49075
        
        return 1 / (1 + np.exp( -(a*(x-iterations_start)/iterations_per_epoch + b)))
