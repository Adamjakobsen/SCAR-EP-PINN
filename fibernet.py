import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = detect_device()


class WaveAct(nn.Module):
    def __init__(self, omega_0):
        super().__init__()
        self.A = nn.Parameter(torch.ones(1))
        self.phi = nn.Parameter(torch.ones(1))
        self.omega_0 = omega_0

    def forward(self, x):
        return self.A * torch.sin(self.phi + x * self.omega_0)


class FiberNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units, mins, maxs, load=None):
        super().__init__()

        self.device = DEVICE
        self.mins = torch.as_tensor(mins, dtype=torch.float32, device=self.device)
        self.maxes = torch.as_tensor(maxs, dtype=torch.float32, device=self.device)

        a = torch.full_like(self.maxes, -1.0)
        b = torch.full_like(self.maxes, 1.0)

        self.layer_norm = nn.Linear(input_dim, input_dim, dtype=torch.float32)

        denom = torch.clamp(self.maxes - self.mins, min=1e-8)
        self.weight_scale = (b - a) / denom
        self.bias_scale = (-self.mins * (b - a)) / denom + a

        with torch.no_grad():
            self.layer_norm.weight.copy_(torch.diag(self.weight_scale))
            self.layer_norm.bias.copy_(self.bias_scale)

        for param in self.layer_norm.parameters():
            param.requires_grad = False

        layers = [self.layer_norm]
        layers.append(nn.Linear(input_dim, hidden_units, dtype=torch.float32))
        layers.append(WaveAct(30))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units, dtype=torch.float32))
            layers.append(WaveAct(1))

        layers.append(nn.Linear(hidden_units, output_dim, dtype=torch.float32))
        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear) and layer.weight.requires_grad:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.to(self.device)

        if load:
            print(f"Loading model from {load}")
            state = torch.load(load, map_location=self.device)
            self.load_state_dict(state)

    def forward(self, x):
        raw = self.network(x)
        return F.normalize(raw, p=2, dim=1, eps=1e-8)


def setup_data():
    umtomm = 1e-3

    collocation_points = np.load("./data/collocation_points.npy") * umtomm
    collocation_fibers = np.load("./data/collocation_fibers.npy")[:, :2]

    boundary_points = np.load("./data/boundary_edge_centroids.npy") * umtomm
    boundary_fibers = np.load("./data/boundary_fibres.npy")[:, :2]

    print("Collocation points shape:", collocation_points.shape)
    print("Collocation fibers shape:", collocation_fibers.shape)
    print("Boundary points shape:", boundary_points.shape)
    print("Boundary fibers shape:", boundary_fibers.shape)

    X = np.concatenate((collocation_points, boundary_points), axis=0).astype(np.float32)
    y = np.concatenate((collocation_fibers, boundary_fibers), axis=0).astype(np.float32)

    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    y_norm = np.clip(y_norm, 1e-8, None)
    y = y / y_norm

    mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
    print("Mins:", mins)
    print("Maxs:", maxs)
    print("Y mins:", np.min(y, axis=0))
    print("Y maxs:", np.max(y, axis=0))

    return X, y, mins, maxs


def directional_loss(pred, target):
    dot = torch.sum(pred * target, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return 1.0 - torch.mean(dot)


@torch.no_grad()
def directional_metrics(pred, target):
    dot = torch.sum(pred * target, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)

    mean_cos = torch.mean(dot).item()
    mean_angle_deg = torch.mean(torch.rad2deg(torch.acos(dot))).item()
    return mean_cos, mean_angle_deg


def train():
    X, y, mins, maxs = setup_data()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )

    model = FiberNet(
        input_dim=2,
        output_dim=2,
        hidden_layers=3,
        hidden_units=512,
        mins=mins,
        maxs=maxs,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        threshold_mode="rel",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    val_X = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    loss_history = []
    best_val_loss = float("inf")

    for epoch in range(500):
        model.train()
        epoch_loss = 0.0
        epoch_cos = 0.0
        epoch_angle = 0.0
        batch_idx = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        lr = optimizer.param_groups[0]["lr"]

        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = directional_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            mean_cos, mean_angle_deg = directional_metrics(outputs.detach(), batch_y)

            epoch_loss += loss.item()
            epoch_cos += mean_cos
            epoch_angle += mean_angle_deg

            pbar.set_postfix({
                "Loss": f"{epoch_loss / (batch_idx + 1):.4e}",
                "MeanCos": f"{epoch_cos / (batch_idx + 1):.4f}",
                "AngleDeg": f"{epoch_angle / (batch_idx + 1):.2f}",
                "LR": f"{lr:.2e}",
            })
            batch_idx += 1

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "fibernet.pth")

        with torch.no_grad():
            model.eval()
            val_outputs = model(val_X)
            val_loss = directional_loss(val_outputs, val_y).item()
            val_cos, val_angle_deg = directional_metrics(val_outputs, val_y)

        print(
            f"Validation Loss: {val_loss:.6e} | "
            f"MeanCos: {val_cos:.6f} | "
            f"MeanAngleDeg: {val_angle_deg:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./experiments/fibernet/fibernet.pth")

        scheduler.step(val_loss)

        loss_history.append([
            epoch,
            epoch_loss / max(1, len(dataloader)),
            val_loss,
            epoch_cos / max(1, len(dataloader)),
            val_cos,
            epoch_angle / max(1, len(dataloader)),
            val_angle_deg,
        ])

    np.save("loss_history_fibernet.npy", np.array(loss_history, dtype=np.float32))
    


if __name__ == "__main__":
    train()