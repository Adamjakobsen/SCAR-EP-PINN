import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PINN2D(nn.Module):
    """
    A Physics-Informed Neural Network (PINN) for 2D Aliev-Panfilov model on MRI-geometry.
    """

    def __init__(self, config):
        super(PINN2D, self).__init__()
        self.config = config
        # Device check (CPU, MPS for Apple Silicon, or CUDA)
        self.device = self._choose_device()

        # PDE constants
        self.a = 0.15
        self.k = 8.0
        self.mu1 = 0.2
        self.mu2 = 0.3
        self.eps = 0.002
        self.b = 0.15
        self.vm_rest = torch.tensor(-80.0, dtype=torch.float32, device=self.device)
        self.vm_norm = torch.tensor(100.0, dtype=torch.float32, device=self.device)
        mins = np.array([-0.1090, -0.1275,  0.0000,  0.0184, -4.6778, -4.6778,  0.0184,  0.0100])
        maxes = np.array([7.5614e1, 7.3461e1, 9.3023e1, 1.1352e1, 4.6778e0, 4.6778e0,
        1.1352e+1, 7.5000e-2])
        self.mins = torch.tensor(mins, dtype=torch.float32, device=self.device)
        self.maxes = torch.tensor(maxes, dtype=torch.float32, device=self.device)
        # Loss weights
        w = config["TRAINING"]["weights_loss"]
        self.w_pde = torch.tensor(w[0], dtype=torch.float32, device=self.device)
        self.w_ode = torch.tensor(w[1], dtype=torch.float32, device=self.device)
        self.w_bc = torch.tensor(w[2], dtype=torch.float32, device=self.device)
        self.w_ic = torch.tensor(w[3], dtype=torch.float32, device=self.device)
        self.w_data = torch.tensor(w[4], dtype=torch.float32, device=self.device)

        # Network architecture
        self.num_layers = config["TRAINING"]["n_layers"]
        self.num_neurons = config["TRAINING"]["n_neurons"]
        n_inputs = 8 if config["DATA"]["isotropic"] == 0 else 3
        activation = self.config["TRAINING"]["activation"]
        if activation == "tanh":
            activation = nn.Tanh()
        elif activation == "silu":
            activation = nn.SiLU()
        elif activation == "gelu":
            activation = nn.GELU()

        activation_last = self.config["TRAINING"]["activation_last"]
        if activation_last == "tanh":
            activation_last = nn.Tanh()
        elif activation_last == "silu":
            activation_last = nn.SiLU()
        elif activation_last == "gelu":
            activation_last = nn.GELU()
        # Initialise weights glorot/xavier uniform
        num_first = self.config["TRAINING"]["num_first"]
        num_last = self.config["TRAINING"]["num_last"]
        a = torch.ones_like(self.maxes)*(-1)
        b = torch.ones_like(self.maxes)*1
        # Construct model
        layers = []
        # Normalization "layer" (by default, identity initialization)
        self.layer_norm = nn.Linear(n_inputs, n_inputs, dtype=torch.float32)

        self.weight_scale = (b-a)/(self.maxes-self.mins)
        #self.weight_scale[-1] = 1
        # torch.ones_like(self.maxes)
        self.bias_scale = (-self.mins*(b-a))/(self.maxes-self.mins)+a
        #self.bias_scale[-1] = 0
        self.layer_norm.weight.data = torch.diag(self.weight_scale)
        self.layer_norm.bias.data = self.bias_scale
        # self.layer_norm.weight.data = torch.diag(1/self.stds)
        # self.layer_norm.bias.data = -self.means/self.stds
        # freeze layer
        for param in self.layer_norm.parameters():
            param.requires_grad = False
        layers = [self.layer_norm]

        layers += [nn.Linear(n_inputs, num_first,
                             dtype=torch.float32)]
        for l in range(self.num_layers):
            if l == 0:
                layers += [nn.Linear(num_first,
                                     num_first, dtype=torch.float32), activation]

            elif l == 1:
                layers += [nn.Linear(num_first,
                                     self.num_neurons, dtype=torch.float32), activation]

            # elif l == 2:
            #    layers += [nn.Linear(self.num_neurons,
            #                         self.num_neurons), nn.SiLU()]
            #    print("Silu activation added")
            elif l == 3:
                layers += [nn.Linear(self.num_neurons,
                                     self.num_neurons, dtype=torch.float32), activation]
            elif l == self.num_layers-1:
                layers += [nn.Linear(self.num_neurons,
                                     num_last, dtype=torch.float32), activation_last]
            else:
                layers += [nn.Linear(self.num_neurons,
                                     self.num_neurons, dtype=torch.float32), activation]

        layers += [nn.Linear(num_last, 2, dtype=torch.float32)]

        self.model = nn.Sequential(*layers).to(self.device)
        self._initialize_weights()
        if config["TRAINING"]["multi_gpu"]:
            gpu_list = config["TRAINING"]["gpu_list"]
            self.model = nn.DataParallel(self.model, device_ids=gpu_list)
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
        experiment = x_collocation[:, 7:8]  # not used in grads, but kept for indexing

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
