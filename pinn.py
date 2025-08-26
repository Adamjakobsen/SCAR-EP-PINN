import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



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

        # features
        self.n_in  = int(mcfg.get("in_features", 8))
        self.n_out = int(mcfg.get("out_features", 2))
        self.n_layers  = int(mcfg.get("n_layers", 9))
        self.n_neurons = int(mcfg.get("n_neurons", 256))
        activation = str(mcfg.get("activation", "tanh"))
        activation_last = str(mcfg.get("activation_last", "silu"))
        self.mins = mins
        self.maxes = maxes

        # physics constants
        phys = mcfg.get("physics", {})
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

        # loss weights (used by your loss aggregation)
        lw = mcfg.get("loss_weights", {})
        self.w_pde  = float(lw.get("w_pde", 0.01))
        self.w_ode  = float(lw.get("w_ode", 0.01))
        self.w_bc   = float(lw.get("w_bc",  0.01))
        self.w_ic   = float(lw.get("w_ic",  1.0))
        self.w_data = float(lw.get("w_data",1.0))

        if activation == "tanh":
            activation = nn.Tanh()
        elif activation == "silu":
            activation = nn.SiLU()
        
        if activation_last == "tanh":
            activation_last = nn.Tanh()
        elif activation_last == "silu":
            activation_last = nn.SiLU()

        

        a = torch.ones_like(self.maxes)*(-1)
        b = torch.ones_like(self.maxes)*1
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
        return self.model(x)

    def predict(self, x):
        """Predicted membrane voltage (in mV)."""
        out = self.model(x)
        return out[:, 0:1] * self.vm_norm + self.vm_rest

    def data_loss(self, x_data, v_true, w_true):
        """
        Mean square error for (V, W) vs. data.
        v_true is a real voltage in mV, so scale accordingly in the PINN.
        """
        preds = self.forward(x_data)
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
        Relative L1 norm (useful for a quick measure).
        """
        with torch.no_grad():
            out = self.forward(x_data)
            v_pred = out[:, 0:1]
            w_pred = out[:, 1:2]
            rel_l1_v = torch.sum(torch.abs(v_pred - (v_true - self.vm_rest) / self.vm_norm)) / \
                       torch.sum(torch.abs((v_true - self.vm_rest) / self.vm_norm))
            rel_l1_w = torch.sum(torch.abs(w_pred - w_true)) / torch.sum(torch.abs(w_true))
        return rel_l1_v, rel_l1_w

    def pde2d_conductivities(self, x_collocation):
        """
        PDE + ODE residual loss for the Aliev-Panfilov model, with conductivities.
        x_collocation expected shape: [N, 8] => (x, y, t, sigma_a, sigma_b, sigma_c, sigma_d, experiment_id)
        """
        x_space = x_collocation[:, 0:1].requires_grad_()
        y_space = x_collocation[:, 1:2].requires_grad_()
        t_space = x_collocation[:, 2:3].requires_grad_()

        sigma_a = x_collocation[:, 3:4].requires_grad_()
        sigma_b = x_collocation[:, 4:5].requires_grad_()
        sigma_c = x_collocation[:, 5:6].requires_grad_()
        sigma_d = x_collocation[:, 6:7].requires_grad_()
        experiment = x_collocation[:, 7:8] 

        y_pred = self.forward(
            torch.cat([x_space, y_space, t_space, sigma_a, sigma_b, sigma_c, sigma_d, experiment], dim=1)
        )
        v = y_pred[:, 0:1]
        w = y_pred[:, 1:2]

        dV_dt = torch.autograd.grad(v, t_space, torch.ones_like(v), create_graph=True)[0]
        dV_dx = torch.autograd.grad(v, x_space, torch.ones_like(v), create_graph=True)[0]
        dV_dy = torch.autograd.grad(v, y_space, torch.ones_like(v), create_graph=True)[0]
        f1_ = sigma_a * dV_dx + sigma_b * dV_dy
        f2_ = sigma_c * dV_dx + sigma_d * dV_dy
        f1 = torch.autograd.grad(f1_, x_space, torch.ones_like(f1_), create_graph=True)[0]
        f2 = torch.autograd.grad(f2_, y_space, torch.ones_like(f2_), create_graph=True)[0]
        PDE = -dV_dt + (f1 + f2) - self.k * v * (v - self.a) * (v - 1) - w * v

        dW_dt = torch.autograd.grad(w, t_space, torch.ones_like(w), create_graph=True)[0]
        ODE = dW_dt - (self.eps + (self.mu1 * w) / (self.mu2 + v)) * (-w - self.k * v * (v - self.b - 1))

        return torch.mean(PDE**2), torch.mean(ODE**2)

    def bc_loss(self, x_boundary, normals):
        """
        Neumann boundary condition: no-flux => normal · (sigma ∇V) = 0
        x_boundary shape [N, 8], normals [N, 2]
        """
        xb = x_boundary[:, 0:1].requires_grad_()
        yb = x_boundary[:, 1:2].requires_grad_()
        tb = x_boundary[:, 2:3].requires_grad_()

        sigma_a = x_boundary[:, 3:4].requires_grad_()
        sigma_b = x_boundary[:, 4:5].requires_grad_()
        sigma_c = x_boundary[:, 5:6].requires_grad_()
        sigma_d = x_boundary[:, 6:7].requires_grad_()
        experiment = x_boundary[:, 7:8]

        v_pred = self.forward(torch.cat([xb, yb, tb, sigma_a, sigma_b, sigma_c, sigma_d, experiment], dim=1))[:, 0:1]
        dV_dx = torch.autograd.grad(v_pred, xb, torch.ones_like(v_pred), create_graph=True)[0]
        dV_dy = torch.autograd.grad(v_pred, yb, torch.ones_like(v_pred), create_graph=True)[0]
        flux = (normals[:, 0:1] * (sigma_a * dV_dx + sigma_b * dV_dy) +
                normals[:, 1:2] * (sigma_c * dV_dx + sigma_d * dV_dy))
        return torch.mean(flux**2)

    def ic_loss(self, x_ic, v_ic_true, w_ic_true):
        """
        Initial condition: at t=0, V=V_ic, W=W_ic.
        x_ic shape [N, 8], v_ic_true, w_ic_true shape [N, 1].
        """
        out = self.forward(x_ic)
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
