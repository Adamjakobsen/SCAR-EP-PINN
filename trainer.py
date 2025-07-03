import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # <-- Added
from torch.utils.data import Dataset, DataLoader
from utils import get_data, get_boundary
import os

class PINNTrainer:
    """
    A trainer class that orchestrates training steps on a given PINN model.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        #load from /Users/adam/SCAR_PINN/experiments/run_20250427_181745/model_Adam.pth
        #self.model.load_state_dict(torch.load("/Users/adam/SCAR_PINN/experiments/run_20250427_181745/model_Adam.pth"))
        # Phase 1 (data-focused)
        self.optimizer_phase1 = optim.Adam(
            self.model.parameters(),
            lr=config["TRAINING"]["lr"]
        )
        # Phase 2 (full PDE + data)
        self.optimizer_phase2 = optim.Adam(
            self.model.parameters(),
            lr=config["TRAINING"]["lr_phase2"],
            weight_decay=config["TRAINING"]["weight_decay"]
        )
        # L-BFGS
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=config["TRAINING"]["lr_lbfgs"],
            max_iter=20,
            line_search_fn="strong_wolfe",
            history_size=100
        )

    def train(self, datamodule):
        """
        Coordinates the multi-phase training approach:
          1) Data pretraining
          2) PDE training (Adam)
          3) PDE training (L-BFGS)
        data = datamodule.setup()
        train_loader = data['train_loader']
        X_test = data['X_test']
        V_test = data['V_test']
        W_test = data['W_test']

        X_colloc = data['X_colloc']
        X_boundary = data['X_boundary']
        boundary_normals = data['boundary_normals']
        X_ic = data['X_ic']
        V_ic = data['V_ic']
        W_ic = data['W_ic']
        """
        self.setup_data()

        self.iteration = 0
        self.losses= []

        print("\n=== [Phase 1: Data Pretraining] ===")
        self.data_phase(self.dataloader, self.X_test, self.V_test, self.W_test)

        print("\n=== [Phase 2: PDE + Data Training (Adam)] ===")
        self.adam_phase(self.dataloader, self.X_test, self.V_test, self.W_test,
                        self.X_collocation_train, self.X_boundary, self.X_normals, self.X_ic_train, self.V_ic_train, self.W_ic_train)
        self.dataloader_lbfgs = DataLoader(self.dataset, batch_size=self.batch_size_lbfgs, shuffle=True)
        print("\n=== [Phase 3: PDE + Data Training (L-BFGS)] ===")
        self.lbfgs_phase(self.dataloader_lbfgs, self.X_test, self.V_test, self.W_test,
                        self.X_collocation_train, self.X_boundary, self.X_normals, self.X_ic_train, self.V_ic_train, self.W_ic_train)
    def data_phase(self, train_loader, X_test, V_test, W_test):
        """
        Phase 1: Data-focused training (only data loss).
        """
        n_epochs_data = self.config["TRAINING"]["n_epochs_data"]
        scheduler_data = optim.lr_scheduler.ExponentialLR(self.optimizer_phase1, gamma=0.5)
        best_rel_l1_v = 1e8

        for epoch in range(n_epochs_data):
            epoch_loss = 0.0
            # Wrap the DataLoader with tqdm for a progress bar
            pbar = tqdm(train_loader, desc=f"Data Phase Epoch {epoch+1}/{n_epochs_data}", leave=False)
            for batch in pbar:
                X_data = batch['X'].to(self.device)
                V_data = batch['V'].to(self.device)
                W_data = batch['W'].to(self.device)

                self.optimizer_phase1.zero_grad()
                mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)
                
                
                loss = mse_v + 0.5*mse_w
                loss.backward()
                self.optimizer_phase1.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    # Compute the relative L1 norm on the test set
                    # This is a quick check, not part of the training loss
                    #RL1 on test
                    rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
                    mae_v, mae_w = self.model.eval_mae(X_test, V_test, W_test)
                pbar.set_postfix({
                    "MSE_v": f"{mse_v.item():.4e}",
                    "MSE_w": f"{mse_w.item():.4e}",
                    "Rel L1 V": f"{rel_v.item():.4e}",
                    "Rel L1 W": f"{rel_w.item():.4e}",
                    "MAE V": f"{mae_v.item():.4e}",
                    "MAE W": f"{mae_w.item():.4e}"

                })
                #Delete batch
                del X_data
                del V_data
                del W_data
                self.losses.append([self.iteration, mse_v.item(), mse_w.item(), rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.iteration += 1
                

            epoch_loss /= len(train_loader)

            # Check test metric at the end of the epoch
            rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
            if rel_v.item() < best_rel_l1_v:
                best_rel_l1_v = rel_v.item()

            print(f"[Data Phase] Epoch {epoch+1}/{n_epochs_data} | "
                  f"Train Loss (avg): {epoch_loss:.4e} | "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}")
            
            scheduler_data.step()
            print("lr:", scheduler_data.get_last_lr())

    def adam_phase(self, train_loader, X_test, V_test, W_test,
                   X_colloc, X_boundary, boundary_normals,
                   X_ic, V_ic, W_ic):
        """
        Phase 2: Training with PDE + Data using Adam.
        """
        n_epochs_Adam = self.config["TRAINING"]["n_epochs_Adam"]
        # Example StepLR, or any other
        scheduler = optim.lr_scheduler.StepLR(self.optimizer_phase2, step_size=1, gamma=0.9)
        best_metric = 1e8

        for epoch in range(n_epochs_Adam):
            epoch_loss = 0.0
            batch_idx = 0
            pbar = tqdm(train_loader, desc=f"Adam Phase Epoch {epoch+1}/{n_epochs_Adam}", leave=False)
            for batch in pbar:
                X_data = batch['X'].to(self.device)
                V_data = batch['V'].to(self.device)
                W_data = batch['W'].to(self.device)
                self.optimizer_phase2.zero_grad()

                # PDE / ODE
                pde_loss, ode_loss = self.model.pde2d_conductivities(self.X_collocation_train)
                # BC
                bc_loss = self.model.bc_loss(X_boundary, boundary_normals)
                # IC
                ic_v, ic_w = self.model.ic_loss(X_ic, V_ic, W_ic)
                # Data loss
                mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)
                epoch_loss+= mse_v.item() 

                total_loss = (self.model.w_pde * pde_loss +
                              self.model.w_ode * ode_loss +
                              self.model.w_bc * bc_loss +
                              self.model.w_ic * (ic_v + ic_w) +
                              self.model.w_data * (mse_v + mse_w))

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer_phase2.step()
                #RL1 on test
                with torch.no_grad():
                    # Compute the relative L1 norm on the test set
                    rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
                    mae_v, mae_w = self.model.eval_mae(X_test, V_test, W_test)

                pbar.set_postfix({
                    "PDE": f"{pde_loss.item():.2e}",
                    "ODE": f"{ode_loss.item():.2e}",
                    "BC": f"{bc_loss.item():.2e}",
                    "MSE_v": f"{mse_v.item():.2e}",
                    "MSE_w": f"{mse_w.item():.2e}",
                    "Rel L1 V": f"{rel_v.item():.2e}",
                    "Rel L1 W": f"{rel_w.item():.2e}",
                    "MAE V": f"{mae_v.item():.2e}",
                    "MAE W": f"{mae_w.item():.2e}"
                })
                #resample every 100 batches
                #if (batch_idx + 1) % 1000 == 0:
                #    self.resample()
                    
                batch_idx += 1
                self.losses.append([self.iteration, mse_v.item(), mse_w.item(), rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                self.iteration += 1
                #Delete batch
                del X_data
                del V_data
                del W_data
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()}")
            

            # quick check on test
            rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
            mae_v, mae_w = self.model.eval_mae(X_test, V_test, W_test)
            if rel_v.item() < best_metric:
                best_metric = rel_v.item()
                # Optionally save best checkpoint here
            epoch_loss /= len(train_loader)
            print(f"[Adam Phase] Epoch {epoch+1}/{n_epochs_Adam} | "
                  f"Epoch loss avg V: {epoch_loss}, Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e},MAE V: {mae_v.item():.4e}, MAE W: {mae_w.item():.4e}")
            #save intermediate model
            path = os.path.join(self.config["path_directory"], f"model_Adam_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)
        #Save model
        path = os.path.join(self.config["path_directory"], "model_Adam.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {self.config['path_directory']}/model_Adam.pth")
        #Save losses
        np.save(os.path.join(self.config["path_directory"], "losses.npy"), self.losses)
    def lbfgs_phase(self, train_loader, X_test, V_test, W_test,
                    X_colloc, X_boundary, boundary_normals,
                    X_ic, V_ic, W_ic):
        """
        Phase 3: Training with PDE + Data using L-BFGS.
        """
        n_epochs_lbfgs = self.config["TRAINING"]["n_epochs_lbfgs"]
        # Possibly apply an exponential decay or similar
        scheduler_lbfgs = optim.lr_scheduler.ExponentialLR(self.optimizer_lbfgs, gamma=0.98)

        for epoch in range(n_epochs_lbfgs):
            
            # L-BFGS typically accumulates over the entire data in each iteration,
            # so you might just run over the same loader or a large offline set:
            pbar = tqdm(train_loader, desc=f"L-BFGS Phase Epoch {epoch+1}/{n_epochs_lbfgs}", leave=False)
            batch_idx = 0
            for batch in pbar:
                X_data = batch['X'].to(self.device)
                V_data = batch['V'].to(self.device)
                W_data = batch['W'].to(self.device)
                def closure():
                    self.optimizer_lbfgs.zero_grad()
                    pde_loss, ode_loss = self.model.pde2d_conductivities(self.X_collocation_train)
                    bc_loss = self.model.bc_loss(X_boundary, boundary_normals)
                    ic_v, ic_w = self.model.ic_loss(X_ic, V_ic, W_ic)
                    mse_v, mse_w = self.model.data_loss(X_data, V_data, W_data)

                    total_loss = (self.model.w_pde * pde_loss +
                                  self.model.w_ode * ode_loss +
                                  self.model.w_bc * bc_loss +
                                  self.model.w_ic * (ic_v + ic_w) +
                                  self.model.w_data * (mse_v + mse_w))

                    total_loss.backward()
                    with torch.no_grad():
                        mae_v, mae_w = self.model.eval_mae(X_test, V_test, W_test)
                        rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
                    pbar.set_postfix({
                        "PDE": f"{pde_loss.item():.2e}",
                        "ODE": f"{ode_loss.item():.2e}",
                        "BC": f"{bc_loss.item():.2e}",
                        "MSE_v": f"{mse_v.item():.2e}",
                        "MSE_w": f"{mse_w.item():.2e}",
                        "MAE V": f"{mae_v.item():.2e}",
                        "MAE W": f"{mae_w.item():.2e}"
                    })
                    self.iteration += 1
                    self.losses.append([self.iteration, mse_v.item(), mse_w.item(), rel_v.item(), rel_w.item(), mae_v.item(), mae_w.item()])
                    return total_loss
                if (batch_idx + 1) % 50 == 0:
                    self.resample()
                    print("Resampling data...")
                batch_idx += 1

                self.optimizer_lbfgs.step(closure)
                del X_data
                del V_data
                del W_data

            scheduler_lbfgs.step()
            print(f"Learning rate: {scheduler_lbfgs.get_last_lr()}")

            # quick check at epoch end
            with torch.no_grad():
                rel_v, rel_w = self.model.eval_rel_l1(X_test, V_test, W_test)
                mae_v, mae_w = self.model.eval_mae(X_test, V_test, W_test)
            print(f"[L-BFGS Phase] Epoch {epoch+1}/{n_epochs_lbfgs} | "
                  f"Rel L1 V: {rel_v.item():.4e}, Rel L1 W: {rel_w.item():.4e}, MAE V: {mae_v.item():.4e}, MAE W: {mae_w.item():.4e}")
            #save intermediate model
            path = os.path.join(self.config["path_directory"], f"model_lbfgs_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to: {self.config['path_directory']}/model_lbfgs_epoch_{epoch+1}.pth")
            #Save losses
            np.save(os.path.join(self.config["path_directory"], "losses.npy"), self.losses)
    def setup_data(self):
        """
        Loads geometry/data from disk and processes into:
        - train_loader (DataLoader)
        - test data (for quick evaluation)
        - boundary arrays (for BC)
        - collocation arrays (for PDE)
        - initial-condition arrays (for IC)
        """
        #set device to mps if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)
        self.a = 0.15
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.h = 0.1
        self.beta = 0.14*1e-4  # 1/cm
        self.Cm = 1e-3  # mF/cm^2
        self.mS_cm = 10  # mS/cm
        self.t_norm = 12.9
        self.touAcm2 = 100/12.9
        self.tomAcm2 = 100*1e-3/12.9
        slice_xy = self.config["DATA"]["slice_xy"]
        slice_t = self.config["DATA"]["slice_t"]
        self.num_domain = self.config["DATA"]["num_domain"]
        self.num_boundary = self.config["DATA"]["num_boundary"]
        self.num_data = self.config["DATA"]["num_data"]
        self.num_init = self.config["DATA"]["num_init"]
        self.num_space = self.config["DATA"]["num_space"]
        ic_space_slice = self.config["DATA"]["ic_space_slice"]
        ic_time_steps = self.config["DATA"]["ic_time_steps"]
        nx_test_slice = self.config["DATA"]["nx_test_slice"]
        nt_test_slice = self.config["DATA"]["nt_test_slice"]
        batch_size = self.config["DATA"]["batch_size"]
        self.batch_size_lbfgs = self.config["DATA"]["batch_size_lbfgs"]
        umtomm = 1e-3
        s_mtomm2_TU = 12.9 / 1.4
        # g_il_values = np.array(
        #    [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        np.random.seed(42)

        # print("Loading data...")
        n_exp = self.config["DATA"]["n_exp"]
        #every_idx = self.config["DATA"]["every_exp_idx"]
        #from_idx = self.config["DATA"]["from_idx"]
        # Load data
        vertices, triangles, _ = get_data()

        vertices = (vertices*umtomm)  # mm
        g_il_values = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
                                0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1])
        # g_il_values = np.array(
        #    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        # indices_experiments = np.arange(len(g_il_values)-6)
        indices_experiments = np.arange(len(g_il_values))
        
        print("Indices experiments", indices_experiments)
        # sampled_indices = indices_experiments[(
        #    indices_experiments + 1) % 3 != 0]
        # add idx 13
        # sampled_indices = np.append(sampled_indices, 13)
        sampled_indices = indices_experiments[(
            indices_experiments + 1) % 4 != 0]
        sampled_indices=sampled_indices[:12]
        # sampled_indices = np.append(sampled_indices, 2)
        #sampled_indices = np.append(sampled_indices, 2)
        print("Sampled indices", sampled_indices)

        print("############# Sampled indices #############", sampled_indices)
        print("g_il_values", g_il_values[sampled_indices])
        vm_all = np.load("./data/vm_variation_scar_full.npy")
        print("Vm shape", vm_all.shape)
        # vm_all = (vm_all-self.vm_rest)/self.vm_norm
        W_all = np.load("./data/W_variation_scar_full.npy")
        n_timepoints = vm_all.shape[0]
        dt = 5
        t_start = 0

        t_end = (n_timepoints-1)*dt

        t = np.linspace(t_start, t_end, n_timepoints) / self.t_norm

        t_test = t[::nt_test_slice]
        t = t[::slice_t]

        x_test = vertices[::nx_test_slice, 0]
        y_test = vertices[::nx_test_slice, 1]

        indices_test = [-1]

        vm_test = vm_all[::nt_test_slice, ::nx_test_slice, indices_test]

        vm_all = vm_all[::slice_t, :, sampled_indices]
        print("vm_all shape============", vm_all.shape)

        print("vm_all shape============", vm_all.shape)

        W_test = W_all[::nt_test_slice, ::nx_test_slice, indices_test]
        W_all = W_all[::slice_t, :, sampled_indices]

        conductivities_all = np.load(
            './data/conductivities_variation_scar_full.npy')

        conductivities_boundary = np.load(
            "./data/conductivities_boundary_1.npy")

        conductivities_all = conductivities_all * s_mtomm2_TU

        # last two experiments [:, :, :, -2:]
        indices_collocation = [0, 1, 3, 14, 15, 16, 17, 18]
        conductivities_collocation = conductivities_all[:,
                                                        :, :, indices_collocation]
        print("Conductivities collocation============ ",
            conductivities_collocation.shape)

        conductivities_test = conductivities_all[::nx_test_slice,
                                                :, :, indices_test]
        conductivities_all = conductivities_all[:,
                                                :, :, sampled_indices]
        conductivities_boundary = conductivities_boundary * s_mtomm2_TU
        conductivities_boundary = conductivities_boundary[:,
                                                        :, :, indices_collocation]

        print("Conductivities============ ", conductivities_all.shape)

        # vertices_boundary, triangles_boundary = get_boundary()

        # vertices_boundary = vertices_boundary * umtomm
        vertices_boundary = np.load(
            "./data/boundary_edge_centroids.npy")*umtomm
        vertices_boundary = vertices_boundary[:, :2]
        print("Boundary points", vertices_boundary.shape)
        boundary_points_old, triangles_boundary_old = get_boundary()
        boundary_points_old = boundary_points_old[:, :2] * umtomm

        # Boundary points and normals
        boundary_normals = np.load("./data/boundary_normals.npy")
        boundary_normals = boundary_normals[:, :2]
        print("Boundary normals", boundary_normals.shape)

        # scar vertices
        #is_vertex_on_scar = np.load("./data/scar_indices.npy")
        #print("Scar vertices", is_vertex_on_scar.shape)

        # scar_vertices = vertices[is_vertex_on_scar, :]
        #scar_vertices = np.load("./data/gregion3_coordinates.npy") * umtomm
        #print("Scar vertices", scar_vertices.shape)
        #scar_vertices_expanded = scar_vertices[np.newaxis, :]

        # Excluding boundary points from vertices
        vertices_expanded = vertices[:, np.newaxis]
        print("Vertices shape", vertices_expanded.shape)
        boundary_vertices_expanded = boundary_points_old[np.newaxis, :]
        # find indices of boundary points by using np.isclose

        is_vertex_on_boundary = np.any(
            np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        b_verts = vertices[is_vertex_on_boundary, :]
        # save boundary points
        np.save("./data/boundary_points_old.npy", b_verts)
        # is_vertex_on_scar = np.any(
        #    np.all(vertices_expanded == scar_vertices_expanded, axis=-1), axis=-1)
        unique_vertices = vertices[~is_vertex_on_boundary, :]
        # is_vertex_on_boundary = np.any(
        #   np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        # flatten along last dimension
        sigma_a_collocation = conductivities_collocation[~is_vertex_on_boundary, 0, 0, :]
        sigma_b_collocation = conductivities_collocation[~is_vertex_on_boundary, 0, 1, :]
        sigma_c_collocation = conductivities_collocation[~is_vertex_on_boundary, 1, 0, :]
        sigma_d_collocation = conductivities_collocation[~is_vertex_on_boundary, 1, 1, :]
        print("sigma_a_collocation shape", sigma_a_collocation.shape)

        vm_all_inner = vm_all[:, ~is_vertex_on_boundary, :]
        W_all_inner = W_all[:, ~is_vertex_on_boundary, :]
        conductivities_all = conductivities_all[~is_vertex_on_boundary, :, :, :]
        # Pick spacial points
        # idx_space = np.random.choice(unique_vertices.shape[0], num_space, replace=False)
        x = unique_vertices[:, 0]
        y = unique_vertices[:, 1]
        x_ic = unique_vertices[::ic_space_slice, 0]
        y_ic = unique_vertices[::ic_space_slice, 1]
        vm_train = vm_all_inner[:, :, :]
        print("vm_train shape", vm_train.shape[2])
        W_train = W_all_inner[:, :, :]
        n_x = vm_train.shape[1]

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

        # Create meshgrid for boundary points
        x_boundary = vertices_boundary[:, 0]
        y_boundary = vertices_boundary[:, 1]
        X_boundary, T_boundary = np.meshgrid(x_boundary, t)
        Y_boundary, T_boundary = np.meshgrid(y_boundary, t)
        X_boundary = X_boundary.reshape(-1, 1)
        T_boundary = T_boundary.reshape(-1, 1)
        Y_boundary = Y_boundary.reshape(-1, 1)
        ones = np.ones_like(X_boundary)
        # vertices_expanded = vertices[:, np.newaxis]
        # boundary_vertices_expanded = vertices_boundary[np.newaxis, :]
        # is_vertex_on_boundary = np.any(
        #    np.all(vertices_expanded == boundary_vertices_expanded, axis=-1), axis=-1)
        # is_in_boundary_points = np.any(np.all(
        #    boundary_points[:, np.newaxis] == vertices_boundary[np.newaxis, :], axis=-1), axis=-1)
        # boundary_normals = boundary_normals[is_in_boundary_points, :]
        # Create meshgrid for boundary normals
        X_boundary_normals, T_boundary_normals = np.meshgrid(
            boundary_normals[:, 0], t)
        Y_boundary_normals, T_boundary_normals = np.meshgrid(
            boundary_normals[:, 1], t)
        X_boundary_normals = X_boundary_normals.reshape(-1, 1)
        Y_boundary_normals = Y_boundary_normals.reshape(-1, 1)
        X_normals = np.hstack((X_boundary_normals, Y_boundary_normals))
        self.X_normals_all = np.zeros(
            (X_boundary.shape[0]*vm_test.shape[1], 2), dtype=np.float32)

        # self.X_boundary_all = np.hstack(
        #    (X_boundary, Y_boundary, T_boundary, sigma_a_boundary, sigma_b_boundary, sigma_c_boundary, sigma_d_boundary))
        # Conductivities
        self.conductivities_all = conductivities_all[:, :, :, :]

        sigma_a_train = self.conductivities_all[:, 0, 0, :]
        sigma_b_train = self.conductivities_all[:, 0, 1, :]
        sigma_c_train = self.conductivities_all[:, 1, 0, :]
        sigma_d_train = self.conductivities_all[:, 1, 1, :]
        sigma_a_ic = conductivities_all[::ic_space_slice, 0, 0, :]
        sigma_b_ic = conductivities_all[::ic_space_slice, 0, 1, :]
        sigma_c_ic = conductivities_all[::ic_space_slice, 1, 0, :]
        sigma_d_ic = conductivities_all[::ic_space_slice, 1, 1, :]

        # sigma_a_scar = conductivities_all[is_vertex_on_scar, 0, 0, :]
        # sigma_b_scar = conductivities_all[is_vertex_on_scar, 0, 1, :]
        # sigma_c_scar = conductivities_all[is_vertex_on_scar, 1, 0, :]
        # sigma_d_scar = conductivities_all[is_vertex_on_scar, 1, 1, :]

        X_c, T_c = np.meshgrid(x, t)

        sigma_a_test = conductivities_test[:, 0, 0]
        sigma_b_test = conductivities_test[:, 0, 1]
        sigma_c_test = conductivities_test[:, 1, 0]
        sigma_d_test = conductivities_test[:, 1, 1]
        sigma_a_test, T_test_ = np.meshgrid(sigma_a_test, t_test)
        sigma_b_test, T_test_ = np.meshgrid(sigma_b_test, t_test)
        sigma_c_test, T_test_ = np.meshgrid(sigma_c_test, t_test)
        sigma_d_test, T_test_ = np.meshgrid(sigma_d_test, t_test)
        sigma_a_test = sigma_a_test.reshape(-1, 1)
        sigma_b_test = sigma_b_test.reshape(-1, 1)
        sigma_c_test = sigma_c_test.reshape(-1, 1)
        sigma_d_test = sigma_d_test.reshape(-1, 1)

        experiment_feature_test = np.ones(
            sigma_a_test.shape) * (g_il_values[-1])

        features_test_numpy = np.hstack((X_test, Y_test, T_test, sigma_a_test,
                                        sigma_b_test, sigma_c_test, sigma_d_test, experiment_feature_test))

        t_ic = t[:ic_time_steps]
        X_ic, T_ic = np.meshgrid(x_ic, t_ic)
        Y_ic, T_ic = np.meshgrid(y_ic, t_ic)
        X_ic = X_ic.reshape(-1, 1)
        T_ic = T_ic.reshape(-1, 1)
        Y_ic = Y_ic.reshape(-1, 1)

        # x_scar, t_scar = np.meshgrid(scar_vertices[:, 0], t)
        # y_scar, t_scar = np.meshgrid(scar_vertices[:, 1], t)
        # X_scar = x_scar.reshape(-1, 1)
        # T_scar = t_scar.reshape(-1, 1)
        # Y_scar = y_scar.reshape(-1, 1)

        self.features_all_numpy = np.zeros(
            (vm_train.shape[0]*vm_train.shape[1]*vm_train.shape[2], 8))
        self.features_collocation = np.zeros(
            (vm_train.shape[0]*vm_train.shape[1]*sigma_a_collocation.shape[-1], 8))
        print("features collocation shape", self.features_collocation.shape)
        self.features_boundary_all = np.zeros(
            (x_boundary.shape[0]*t.shape[0]*conductivities_boundary.shape[-1], 8))
        self.features_ic_all_numpy = np.zeros(
            (X_ic.shape[0]*vm_train.shape[2], 8), dtype=np.float32)
        self.V_ic_all_numpy = np.zeros(
            (X_ic.shape[0]*vm_train.shape[2], 1), dtype=np.float32)
        self.W_ic_all_numpy = np.zeros(
            (X_ic.shape[0]*vm_train.shape[2], 1), dtype=np.float32)
        self.v_flat_numpy = np.zeros(
            (vm_train.shape[0]*vm_train.shape[1]*vm_train.shape[2], 1))
        self.W_flat_numpy = np.zeros(
            (W_train.shape[0]*W_train.shape[1]*W_train.shape[2], 1))
        # self.v_flat_scar = np.zeros(
        #    (vm_train.shape[0]*scar_vertices.shape[0]*vm_train.shape[2], 1))
        # self.W_flat_scar = np.zeros(
        #    (vm_train.shape[0]*scar_vertices.shape[0]*vm_train.shape[2], 1))
        # self.features_scar_all = np.zeros(
        #    (scar_vertices.shape[0]*t.shape[0]*vm_train.shape[2], 7))
        for i in range(conductivities_collocation.shape[3]):
            print("i", i)
            start = i*vm_train.shape[0]*vm_train.shape[1]
            end = (i+1)*vm_train.shape[0]*vm_train.shape[1]
            print("sigma_a_collocation shape", sigma_a_collocation.shape)

            sigma_a_collocation_mesh, _ = np.meshgrid(
                sigma_a_collocation[:, i], t)
            sigma_b_collocation_mesh, _ = np.meshgrid(
                sigma_b_collocation[:, i], t)
            sigma_c_collocation_mesh, _ = np.meshgrid(
                sigma_c_collocation[:, i], t)
            sigma_d_collocation_mesh, _ = np.meshgrid(
                sigma_d_collocation[:, i], t)
            sigma_a_collocation_flat = sigma_a_collocation_mesh.reshape(-1, 1)
            sigma_b_collocation_flat = sigma_b_collocation_mesh.reshape(-1, 1)
            sigma_c_collocation_flat = sigma_c_collocation_mesh.reshape(-1, 1)
            sigma_d_collocation_flat = sigma_d_collocation_mesh.reshape(-1, 1)
            experiment_feature = np.ones(
                sigma_a_collocation_flat.shape) * g_il_values[indices_collocation[i]]
            Xi_collocation = np.hstack((X, Y, T, sigma_a_collocation_flat, sigma_b_collocation_flat,
                                    sigma_c_collocation_flat, sigma_d_collocation_flat, experiment_feature))
            self.features_collocation[start:end, :] = Xi_collocation

            # Boundary points
            sigma_a_boundary = conductivities_boundary[:, 0, 0, i]
            sigma_b_boundary = conductivities_boundary[:, 0, 1, i]
            sigma_c_boundary = conductivities_boundary[:, 1, 0, i]
            sigma_d_boundary = conductivities_boundary[:, 1, 1, i]
            sigma_a_boundary, T_ = np.meshgrid(sigma_a_boundary, t)
            sigma_b_boundary, T_ = np.meshgrid(sigma_b_boundary, t)
            sigma_c_boundary, T_ = np.meshgrid(sigma_c_boundary, t)
            sigma_d_boundary, T_ = np.meshgrid(sigma_d_boundary, t)
            sigma_a_boundary = sigma_a_boundary.reshape(-1, 1)
            sigma_b_boundary = sigma_b_boundary.reshape(-1, 1)
            sigma_c_boundary = sigma_c_boundary.reshape(-1, 1)
            sigma_d_boundary = sigma_d_boundary.reshape(-1, 1)
            start_bc = i*X_boundary.shape[0]
            end_bc = (i+1)*X_boundary.shape[0]
            experiment_feature = np.ones(
                sigma_a_boundary.shape) * g_il_values[indices_collocation[i]]
            self.features_boundary_all[start_bc:end_bc, :] = np.hstack(
                (X_boundary, Y_boundary, T_boundary, sigma_a_boundary, sigma_b_boundary, sigma_c_boundary, sigma_d_boundary, experiment_feature))
            # normals
            start_normals = i*X_normals.shape[0]
            end_normals = (i+1)*X_normals.shape[0]
            self.X_normals_all[start_normals:end_normals, :] = X_normals

        for i in range(vm_train.shape[2]):
            start = i*vm_train.shape[0]*vm_train.shape[1]
            end = (i+1)*vm_train.shape[0]*vm_train.shape[1]
            self.v_flat_numpy[start:end] = vm_train[:, :, i].reshape(-1, 1)
            self.W_flat_numpy[start:end] = W_train[:, :, i].reshape(-1, 1)

            sigma_a, T_ = np.meshgrid(sigma_a_train[:, i], t)
            sigma_b, T_ = np.meshgrid(sigma_b_train[:, i], t)
            sigma_c, T_ = np.meshgrid(sigma_c_train[:, i], t)
            sigma_d, T_ = np.meshgrid(sigma_d_train[:, i], t)
            sigma_a_flat = sigma_a.reshape(-1, 1)
            sigma_b_flat = sigma_b.reshape(-1, 1)
            sigma_c_flat = sigma_c.reshape(-1, 1)
            sigma_d_flat = sigma_d.reshape(-1, 1)
            # add feature = i same length as sigma_a_flat
            experiment_feature = np.ones(
                sigma_a_flat.shape) * g_il_values[sampled_indices[i]]

            X_i = np.hstack(
                (X, Y, T, sigma_a_flat, sigma_b_flat, sigma_c_flat, sigma_d_flat, experiment_feature))
            self.features_all_numpy[start:end, :] = X_i
            # Sample intitial coindition points
            sigma_a_ic_, T_ic_ = np.meshgrid(sigma_a_ic[:, i], t_ic)
            sigma_b_ic_, T_ic_ = np.meshgrid(sigma_b_ic[:, i], t_ic)
            sigma_c_ic_, T_ic_ = np.meshgrid(sigma_c_ic[:, i], t_ic)
            sigma_d_ic_, T_ic_ = np.meshgrid(sigma_d_ic[:, i], t_ic)
            sigma_a_ic_flat = sigma_a_ic_.reshape(-1, 1)
            sigma_b_ic_flat = sigma_b_ic_.reshape(-1, 1)
            sigma_c_ic_flat = sigma_c_ic_.reshape(-1, 1)
            sigma_d_ic_flat = sigma_d_ic_.reshape(-1, 1)
            experiment_feature_ic = np.ones(
                sigma_a_ic_flat.shape) * g_il_values[sampled_indices[i]]
            start_ic = i*X_ic.shape[0]
            end_ic = (i+1)*X_ic.shape[0]
            self.features_ic_all_numpy[start_ic:end_ic, :] = np.hstack(
                (X_ic, Y_ic, T_ic, sigma_a_ic_flat, sigma_b_ic_flat, sigma_c_ic_flat, sigma_d_ic_flat, experiment_feature_ic))
            self.V_ic_all_numpy[start_ic:end_ic] = vm_train[:ic_time_steps,
                                                            ::ic_space_slice, i].reshape(-1, 1)
            self.W_ic_all_numpy[start_ic:end_ic] = W_train[:ic_time_steps,
                                                        ::ic_space_slice, i].reshape(-1, 1)

            # Sample scar points
            # sigma_a_scar_, T_scar_ = np.meshgrid(sigma_a_scar[:, i], t)
            # sigma_b_scar_, T_scar_ = np.meshgrid(sigma_b_scar[:, i], t)
            # sigma_c_scar_, T_scar_ = np.meshgrid(sigma_c_scar[:, i], t)
            # sigma_d_scar_, T_scar_ = np.meshgrid(sigma_d_scar[:, i], t)
            # sigma_a_scar_flat = sigma_a_scar_.reshape(-1, 1)
            # sigma_b_scar_flat = sigma_b_scar_.reshape(-1, 1)
            # sigma_c_scar_flat = sigma_c_scar_.reshape(-1, 1)
            # sigma_d_scar_flat = sigma_d_scar_.reshape(-1, 1)

            # start_scar = i*X_scar.shape[0]
            # end_scar = (i+1)*X_scar.shape[0]

            # self.features_scar_all[start_scar:end_scar, :] = np.hstack(
            #    (X_scar, Y_scar, T_scar, sigma_a_scar_flat, sigma_b_scar_flat, sigma_c_scar_flat, sigma_d_scar_flat))
            # self.v_flat_scar[start_scar:end_scar] = vm_all[:,
            #                                               is_vertex_on_scar, i].reshape(-1, 1)
            # self.W_flat_scar[start_scar:end_scar] = W_all[:,
            #                                              is_vertex_on_scar, i].reshape(-1, 1)

        # idx_scar_c = np.random.choice(
        #    self.features_scar_all.shape[0], 1000, replace=False)
        idx_colocation = np.random.choice(
            self.features_collocation.shape[0], self.num_domain, replace=False)
        # Add idx_scar_c to idx_colocation
        # idx_colocation = np.concatenate((idx_colocation, idx_scar_c))
        X_collocation_numpy = self.features_collocation[idx_colocation]
        # V_collocation_numpy = self.v_flat_collocation[idx_colocation]
        # W_collocation_numpy = self.W_flat2__collocation[idx_colocation]

        self.X_collocation_train = torch.tensor(
            X_collocation_numpy, dtype=torch.float32, device=self.device)
        print("X collocation shape", self.X_collocation_train.shape)
        # self.V_collocation = torch.tensor(
        #    V_collocation_numpy, dtype=torch.float32, device=self.device)
        # self.W_collocation = torch.tensor(
        #    W_collocation_numpy, dtype=torch.float32, device=self.device)

        idx_boundary = np.random.choice(
            self.features_boundary_all.shape[0], self.num_boundary, replace=False)
        self.X_boundary = torch.tensor(self.features_boundary_all[idx_boundary],
                                    dtype=torch.float32, device=self.device)

        self.X_normals = torch.tensor(self.X_normals_all[idx_boundary],
                                    dtype=torch.float32, device=self.device)
        # idx_scar = np.random.choice(
        #    self.features_scar_all.shape[0], 100000, replace=False)
        # self.X_scar = torch.tensor(self.features_scar_all[idx_scar],
        #                           dtype=torch.float32, device=self.device)
        # self.V_scar_train = torch.tensor(self.v_flat_scar[idx_scar],
        #                                 dtype=torch.float32, device=self.device)
        # self.W_scar_train = torch.tensor(self.W_flat_scar[idx_scar],
        #                                 dtype=torch.float32, device=self.device)
        idx_ic = np.random.choice(
            self.features_ic_all_numpy.shape[0], self.num_init, replace=False)
        self.X_ic_train = torch.tensor(self.features_ic_all_numpy[idx_ic],
                                    dtype=torch.float32, device=self.device)

        self.V_ic_train = torch.tensor(self.V_ic_all_numpy[idx_ic],
                                    dtype=torch.float32, device=self.device)
        self.W_ic_train = torch.tensor(
            self.W_ic_all_numpy[idx_ic], dtype=torch.float32, device=self.device)

        idx_data = np.random.choice(
            self.features_all_numpy.shape[0], self.num_data, replace=False)
        # every 100th point
        # idx_data = np.arange(0, self.features_all_numpy.shape[0], 72)
        # self.X_data_train = torch.tensor(
        #    self.features_all_numpy[idx_data], dtype=torch.float32, device=self.device)
        # print("Total features", self.features_all_numpy.shape)
        # print("X data shape", self.X_data_train.shape)
        # self.V_data_train = torch.tensor(
        #    self.v_flat_numpy[idx_data], dtype=torch.float32, device=self.device)
        # self.W_data_train = torch.tensor(
        #    self.W_flat_numpy[idx_data], dtype=torch.float32, device=self.device)
        # Adding collocation and ic to data loader
        # self.X_collocation = torch.tensor(
        #    X_collocation_numpy, dtype=torch.float32)

        # Put in pytorch dataloader
        self.X_data_train = torch.tensor(
            self.features_all_numpy, dtype=torch.float32)
        print("#### Total features", self.features_all_numpy.shape)
        self.V_data_train = torch.tensor(
            self.v_flat_numpy, dtype=torch.float32)
        self.W_data_train = torch.tensor(
            self.W_flat_numpy, dtype=torch.float32)
        self.dataset = CustomDataset(
            self.X_data_train, self.V_data_train, self.W_data_train, self.device)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)

        self.X_test = torch.tensor(
            features_test_numpy, dtype=torch.float32, device=self.device)
        print("X test shape", self.X_test.shape)
        self.V_test = torch.tensor(
            V_test, dtype=torch.float32, device=self.device)
        self.W_test = torch.tensor(
            W_test, dtype=torch.float32, device=self.device)

        max_x = self.features_all_numpy[:, 0].max()
        min_x = self.features_all_numpy[:, 0].min()
        max_y = self.features_all_numpy[:, 1].max()
        min_y = self.features_all_numpy[:, 1].min()
        max_t = self.features_all_numpy[:, 2].max()
        min_t = self.features_all_numpy[:, 2].min()
        max_sigma_a = self.features_all_numpy[:, 3].max()
        min_sigma_a = self.features_all_numpy[:, 3].min()
        max_sigma_b = self.features_all_numpy[:, 4].max()
        min_sigma_b = self.features_all_numpy[:, 4].min()
        max_sigma_c = self.features_all_numpy[:, 5].max()
        min_sigma_c = self.features_all_numpy[:, 5].min()
        max_sigma_d = self.features_all_numpy[:, 6].max()
        min_sigma_d = self.features_all_numpy[:, 6].min()
        min_gil = self.features_all_numpy[:, 7].min()
        max_gil = self.features_all_numpy[:, 7].max()
        maxes_numpy = [max_x, max_y, max_t, max_sigma_a,
                       max_sigma_b, max_sigma_c, max_sigma_d, max_gil]
        mins_numpy = [min_x, min_y, min_t, min_sigma_a,
                      min_sigma_b, min_sigma_c, min_sigma_d, min_gil]

        self.maxes = torch.tensor(
            maxes_numpy, dtype=torch.float32, device=self.device)
        self.mins = torch.tensor(
            mins_numpy, dtype=torch.float32, device=self.device)
        print("Maxes", self.maxes)
        print("Mins", self.mins)
        #return self.maxes, self.mins
    def resample(self):
        idx_colocation = np.random.choice(
        self.features_collocation.shape[0], self.num_domain, replace=False)

        X_collocation_numpy = self.features_collocation[idx_colocation]
        self.X_collocation_train = torch.tensor(
            X_collocation_numpy, dtype=torch.float32, device=self.device)

        
            

class CustomDataset(Dataset):
    def __init__(self, X, V, W, device):
        self.X = X
        self.V = V
        self.W = W
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx, :],
            'V': self.V[idx],
            'W': self.W[idx]
        }
