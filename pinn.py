import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class WaveAct(nn.Module):
    def __init__(self,omega_0,device):
        super(WaveAct, self).__init__() 
        self.A = nn.Parameter(torch.ones(1), requires_grad=True)
        self.phi = nn.Parameter(torch.ones(1), requires_grad=True)
        self.omega_0 = omega_0
    
    def forward(self, x):
        #return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)
        return self.A * torch.sin(self.phi + x*self.omega_0)

class PINN2D(nn.Module):
    """
    A Physics-Informed Neural Network (PINN) for 2D Aliev-Panfilov model on MRI-geometry.
    """

    def __init__(self, config, mins, maxes):
        super(PINN2D, self).__init__()
        print("Initializing PINN")
        self.config = config
        # Device check (CPU, MPS for Apple Silicon, or CUDA)
        self.device = self._choose_device()

        mcfg = config["MODEL"]

    
 
        self.n_in = 4
   
        self.n_in = 4
        self.n_out = int(mcfg.get("out_features", 2))
        self.n_layers  = int(mcfg.get("n_layers", 9))
        self.n_neurons = int(mcfg.get("n_neurons", 256))
        activation = str(mcfg.get("activation", "tanh"))
        activation_last = str(mcfg.get("activation_last", "silu"))
        self.mins = mins[:self.n_in]
        self.maxes = maxes[:self.n_in]

        


        # physics constants
        phys = mcfg.get("physics", {})
        self.a   = float(phys.get("a", 0.15))
        self.k   = float(phys.get("k", 8.0))
        self.mu1 = float(phys.get("mu1", 0.2))
        self.mu2 = float(phys.get("mu2", 0.3))
        self.eps = float(phys.get("eps", 0.002))
        self.b   = float(phys.get("b", 0.15))
        self.h   = float(phys.get("h", 0.1))
        self.mS_cm = float(phys.get("mS_cm", 10.0))
        self.vm_rest = float(phys.get("vm_rest", -80.0))  # mV
        self.vm_norm = float(phys.get("vm_norm", 100.0))

        # loss weights (used by your loss aggregation)
        lw = mcfg.get("loss_weights", {})
        self.w_pde  = float(lw.get("w_pde", 0.01))
        self.w_ode  = float(lw.get("w_ode", 0.01))
        self.w_bc   = float(lw.get("w_bc",  0.01))
        self.w_ic   = float(lw.get("w_ic",  1.0))
        self.w_data = float(lw.get("w_data",1.0))
                # Healthy 
        self.h_gil = 2.8 
        self.h_git = 0.26
        self.h_gel = 2.2
        self.h_get = 1.3

        # Scar/LGE (inside LGE). g_il in scar comes from the input feature.
        self.s_git = 0.002#float(s.get("g_it", 0.002))
        self.s_gel = 1.0#float(s.get("g_el", 1.0))
        self.s_get = 1.0#float(s.get("g_et", 1.0))
        self.kapa=12.9 #/1.4 #TODO: check with only 12.9
        
        
        if activation == "tanh":
            activation = nn.Tanh()
        elif activation == "silu":
            activation = nn.SiLU()
        
        if activation_last == "tanh":
            activation_last = nn.Tanh()
        elif activation_last == "silu":
            activation_last = nn.SiLU()

        

        a = torch.ones_like(self.maxes)*(-2)
        b = torch.ones_like(self.maxes)*2
        # Construct model
        layers = []
        # Scaling layer
        self.layer_norm = nn.Linear(self.n_in, self.n_in, dtype=torch.float32)

        self.weight_scale = (b-a)/(self.maxes-self.mins)

        self.bias_scale = (-self.mins*(b-a))/(self.maxes-self.mins)+a

        self.layer_norm.weight.data = torch.diag(self.weight_scale)
        self.layer_norm.bias.data = self.bias_scale

        # freeze layer
        for param in self.layer_norm.parameters():
            param.requires_grad = False
        layers = [self.layer_norm]

        layers += [nn.Linear(self.n_in, self.n_neurons,
                             dtype=torch.float32)]
        for l in range(self.n_layers):

            if l == self.n_layers-1:
                layers += [nn.Linear(self.n_neurons,
                                     self.n_neurons, dtype=torch.float32), activation_last]
            else:
                layers += [nn.Linear(self.n_neurons,
                                     self.n_neurons, dtype=torch.float32), activation]

        layers += [nn.Linear(self.n_neurons, 2, dtype=torch.float32)]

        self.model = nn.Sequential(*layers).to(self.device)
        self._initialize_weights()

        # Multi-GPU support (CUDA only)
        dcfg = config.get("DEVICE", {})
        if dcfg.get("multi_gpu", False) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            ids = dcfg.get("gpu_list") or list(range(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=ids)
        #print architecture
        print("PINN2D architecture:")
        print(self.model)

        
    


    def forward(self, x):

        return self.model(x[:,0:self.n_in])

    def predict(self, x):
        """Predicted membrane voltage (in mV)."""
        out = self.model(x[:,0:self.n_in])
        return out[:, 0:1] * self.vm_norm + self.vm_rest
    def freeze_layers(self, up_to: int):
        """
        Freezes the first 'up_to' modules in self.model.
        self.model is an nn.Sequential (or DataParallel wrapper).
        """
        
        # Get the actual sequential model, unwrapping DataParallel if needed
        model_to_freeze = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        print(f"--- Freezing first {up_to} modules of the model ---")
        ct = 0
        # model_to_freeze is the nn.Sequential, which is iterable
        for module in model_to_freeze:
            if ct < up_to:
                print(f"Freezing module {ct}: {module.__class__.__name__}")
                for param in module.parameters():
                    param.requires_grad = False
            else:
                print(f"Module {ct} remains trainable: {module.__class__.__name__}")
            ct += 1
        print("--------------------------------------------------")

    def data_loss(self, x_data, v_true, w_true):
        """
        Mean square error for (V, W) vs. data.
        v_true is a real voltage in mV, so scale accordingly in the PINN.
        """

        x_space = x_data[:, 0:1].requires_grad_()
        y_space = x_data[:, 1:2].requires_grad_()
        t_space = x_data[:, 2:3].requires_grad_()
        g_il = x_data[:, 3:4]
        markers = x_data[:, 4:5]

        for idx, marker in enumerate(markers):
            if marker.item() == 0:
                g_il[idx] = self.h_gil # Use healthy conductivity for healthy regions

        preds = self.forward(torch.cat([x_space, y_space, t_space, g_il], dim=1))
        # preds = self.forward(x_data)
        v_pred = preds[:, 0:1]
        w_pred = preds[:, 1:2]
        diff_v = v_pred - (v_true - self.vm_rest) / self.vm_norm
        diff_w = w_pred - w_true
        return torch.mean(diff_v**2), torch.mean(diff_w**2)

    def eval_mae(self, x_data, v_true, w_true):
        """MAE for (V, W)."""
        with torch.no_grad():
            out = self.forward(x_data)
            v_pred = out[:, 0:1] * self.vm_norm + self.vm_rest
            w_pred = out[:, 1:2]
            return torch.mean(torch.abs(v_pred - v_true)), torch.mean(torch.abs(w_pred - w_true))

    def eval_rel_l1(self, x_data, v_true, w_true):
        """
        Relative L1 norm 
        """
        with torch.no_grad():
            out = self.forward(x_data)
            v_pred = out[:, 0:1]
            w_pred = out[:, 1:2]
            rel_l1_v = torch.sum(torch.abs(v_pred - (v_true - self.vm_rest) / self.vm_norm)) / \
                       torch.sum(torch.abs((v_true - self.vm_rest) / self.vm_norm))
            rel_l1_w = torch.sum(torch.abs(w_pred - w_true)) / torch.sum(torch.abs(w_true))
        return rel_l1_v, rel_l1_w
    
    def _sigma_m_from_fiber_region(
        self,
        ef_x: torch.Tensor,
        ef_y: torch.Tensor,
        g_il_in: torch.Tensor,        # per-experiment scalar broadcasted over points
        region_marker: torch.Tensor,  # 1=healthy, 3=scar (float or int)
    ) -> torch.Tensor:
        """
        Build Σ_m at each point using:
        - Healthy:  (g_il,g_it,g_el,g_et) = (self.h_gil,self.h_git,self.h_gel,self.h_get)
        - Scar:     (g_il,g_it,g_el,g_et) = (g_il_in,self.s_git,self.s_gel,self.s_get)
        Rotation: R = [[ex, ey], [-ey, ex]],  ~Σ = R Σ R^T,
        Monodomain: Σ_m = (Σ_i~ + Σ_e~)^{-1} (Σ_i~ Σ_e~)  
        Returns: (N,2,2)
        """
        dtype = ef_x.dtype
        device = ef_x.device

        
        ex, ey = ef_x, ef_y
        R = torch.stack(
            [torch.cat([ex,  ey], dim=1),
            torch.cat([-ey, ex], dim=1)],
            dim=1
        )  # (N,2,2)
        Rt = R.transpose(1, 2)

        # Mask: scar if marker ~ 3
        is_scar = (region_marker >= (3 - 0.5))

        # Local conductivities
        g_il_loc = torch.where(is_scar, g_il_in, torch.full_like(g_il_in, self.h_gil))
        g_it_loc = torch.where(is_scar, torch.full_like(g_il_in, self.s_git),
                                        torch.full_like(g_il_in, self.h_git))
        g_el_loc = torch.where(is_scar, torch.full_like(g_il_in, self.s_gel),
                                        torch.full_like(g_il_in, self.h_gel))
        g_et_loc = torch.where(is_scar, torch.full_like(g_il_in, self.s_get),
                                        torch.full_like(g_il_in, self.h_get))

        # Diagonal Σ_i and Σ_e in fiber frame
        N = ef_x.size(0)
        Si = torch.zeros(N, 2, 2, dtype=dtype, device=device)
        Se = torch.zeros_like(Si)

        Si[:, 0, 0] = g_il_loc.squeeze(-1)  # longitudinal
        Si[:, 1, 1] = g_it_loc.squeeze(-1)  # transverse

        Se[:, 0, 0] = g_el_loc.squeeze(-1)
        Se[:, 1, 1] = g_et_loc.squeeze(-1)

        # Rotate to physical frame:  ~Σ = R Σ R^T
        Si_t = R.bmm(Si).bmm(Rt)
        Se_t = R.bmm(Se).bmm(Rt)

        # Monodomain projection
        A = Si_t + Se_t
        #A_inv =torch.linalg.pinv(A)                    # (N,2,2)
        B = Si_t.bmm(Se_t)                 # (N,2,2)
        Sigma_m = torch.linalg.solve(A, B) # A X = B  -> X = A^{-1} B
        #print("Sigma_m shape", Sigma_m.shape)
        return self.kapa*Sigma_m
        

    def pde2d_conductivities(self, x_collocation,fibernet):
        """
        PDE + ODE residual loss for the Aliev-Panfilov model
        """
        
        x_space = x_collocation[:, 0:1].requires_grad_()
        y_space = x_collocation[:, 1:2].requires_grad_()
        t_space = x_collocation[:, 2:3].requires_grad_()
        fibers = fibernet(torch.cat([x_space, y_space], dim=1))  # (N, 2)
        ex, ey = fibers[:, 0:1], fibers[:, 1:2]
        R = torch.stack(
            [torch.cat([ex,  ey], dim=1),
            torch.cat([-ey, ex], dim=1)],
            dim=1
        )
        markers = x_collocation[:, 4:5]

        g_il = x_collocation[:, 3:4]

        for idx, marker in enumerate(markers):
            if marker.item() == 0:
                g_il[idx] = self.h_gil # Use healthy conductivity for healthy regions

        y_pred = self.forward(
                torch.cat([x_space, y_space, t_space, g_il], dim=1)
            )
        v = y_pred[:, 0:1]
        w = y_pred[:, 1:2]
        dV_dt, dV_dx, dV_dy = torch.autograd.grad(
            outputs=v, inputs=(t_space, x_space, y_space),
            grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )

        Sigma = self._sigma_m_from_fiber_region(ex, ey, g_il, markers)
        
        Jx = Sigma[:, 0, 0:1]*dV_dx + Sigma[:, 0, 1:2]*dV_dy
        Jy = Sigma[:, 1, 0:1]*dV_dx + Sigma[:, 1, 1:2]*dV_dy

        # dJx_dx, dJy_dy = torch.autograd.grad(
        #     outputs=(Jx, Jy), inputs=(x_space, y_space),
        #     grad_outputs=(torch.ones_like(Jx), torch.ones_like(Jy)),
        #     create_graph=True, retain_graph=True
        # )

        dJx_dx = torch.autograd.grad(
            Jx,
            x_space,
            grad_outputs=torch.ones_like(Jx),
            create_graph=True,
            retain_graph=True
        )[0]

        dJy_dy = torch.autograd.grad(
            Jy,
            y_space,
            grad_outputs=torch.ones_like(Jy),
            create_graph=True,
            retain_graph=True
        )[0]

        
        PDE = -dV_dt + (dJx_dx + dJy_dy) - self.k*v*(v - self.a)*(v - 1.0) - w*v

        dW_dt = torch.autograd.grad(w, t_space, torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        ODE = dW_dt - (self.eps + (self.mu1*w)/(self.mu2 + v)) * (-w - self.k*v*(v - self.b - 1.0))
        return torch.mean(PDE**2), torch.mean(ODE**2)

    def bc_loss(self, x_boundary, normals,fibernet):
        """
        Neumann boundary condition: no-flux 
        """

        xb = x_boundary[:, 0:1].requires_grad_()
        yb = x_boundary[:, 1:2].requires_grad_()
        tb = x_boundary[:, 2:3].requires_grad_()
        g_il = x_boundary[:, 3:4]
        markers = x_boundary[:, 4:5]

        for idx, marker in enumerate(markers):
            if marker.item() == 0:
                g_il[idx] = self.h_gil # Use healthy conductivity for healthy regions

        fibers = fibernet(torch.cat([xb, yb], dim=1))  # (N, 2)
        ex, ey = fibers[:, 0:1], fibers[:, 1:2]
        v = self.forward(torch.cat([xb, yb, tb, g_il], dim=1))[:, 0:1]
        

        dV_dx = torch.autograd.grad(v, xb, torch.ones_like(v), create_graph=True)[0]
        dV_dy = torch.autograd.grad(v, yb, torch.ones_like(v), create_graph=True)[0]

        Sigma = self._sigma_m_from_fiber_region(ex, ey, g_il,markers)
        Jx = Sigma[:, 0, 0:1]*dV_dx + Sigma[:, 0, 1:2]*dV_dy
        Jy = Sigma[:, 1, 0:1]*dV_dx + Sigma[:, 1, 1:2]*dV_dy

        n = normals
        flux_n = n[:, 0:1]*Jx + n[:, 1:2]*Jy
        return flux_n.pow(2).mean()

    def ic_loss(self, x_ic, v_ic_true, w_ic_true):
        """
        Initial condition: at t=0, V=V_ic, W=W_ic.
        x_ic shape [N, 8], v_ic_true, w_ic_true shape [N, 1].
        """

        x_space = x_ic[:, 0:1].requires_grad_()
        y_space = x_ic[:, 1:2].requires_grad_()
        t_space = x_ic[:, 2:3].requires_grad_()
        g_il = x_ic[:, 3:4]
        markers = x_ic[:, 4:5]

        for idx, marker in enumerate(markers):
            if marker.item() == 0:
                g_il[idx] = self.h_gil # Use healthy conductivity for healthy regions

        out = self.forward(torch.cat([x_space, y_space, t_space, g_il], dim=1))
        v_pred = out[:, 0:1]
        w_pred = out[:, 1:2]
        # scale real voltage
        v_ic_scaled = (v_ic_true - self.vm_rest) / self.vm_norm
        return torch.mean((v_pred - v_ic_scaled)**2), torch.mean((w_pred - w_ic_true)**2)

    def _get_activation(self, act_str):
        if act_str == "tanh":
            return nn.Tanh()
        elif act_str == "silu":
            return nn.SiLU()
        elif act_str == "gelu":
            return nn.GELU()
        return nn.Tanh()  # default

    def _choose_device(self):
        # Mac MPS?
        if torch.backends.mps.is_available(): 
            print("Using MPS device")
            return torch.device("mps")
        # CUDA?
        if torch.cuda.is_available():
            print("Using CUDA device")
            return torch.device("cuda")
        # Default CPU
        else:
            print("Using CPU device")
            return torch.device("cpu")

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear) and layer.weight.requires_grad:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
