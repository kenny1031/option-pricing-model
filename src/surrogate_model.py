import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.onnx.utils import model_signature
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")

# Neutral Network
class SurrogateNN(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# Training function
def train_surrogate(
    csv_path=None,
    n_epochs=100,
    lr=1e-3,
    batch_size=64,
    test_size=0.2,
    plot=False,
):
    """Training function for surrogate model"""
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "synthetic_qmc.csv")
    df = pd.read_csv(csv_path)
    X = df[["S0", "K", "T", "r", "sigma"]].values
    y = df["price"].values.reshape(-1, 1)

    # Normalise inputs
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=test_size,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Dataloader for mini-batch training
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model, loss, optimiser
    model = SurrogateNN(input_dim=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    # Training loop
    train_losses, test_losses = [], []
    for epoch in range(n_epochs):
        # Train
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_train_loss = sum(batch_losses) / len(batch_losses)

        # Validation
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test)

        # Step scheduler
        scheduler.step(test_loss)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Test Loss={test_loss:.6f}")
    if plot:
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.title("Training vs Test Loss (scaled)")
        plt.show()

    return model, x_scaler, y_scaler

if __name__ == "__main__":
    model, x_scaler, y_scaler = train_surrogate(csv_path="../data/synthetic_qmc.csv")