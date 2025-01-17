import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ------------------
# Load Data
# ------------------
data = pd.read_csv("mamdata.csv")

# ------------------
# Extract Features and Target
# ------------------
kan_columns = [col for col in data.columns if col.startswith("KAN_Feature")]
mamba_columns = [col for col in data.columns if col.startswith("Mamba_TimeStep")]
target_column = "Target"

x_kan = torch.tensor(data[kan_columns].values, dtype=torch.float32)
x_mamba = torch.tensor(data[mamba_columns].values, dtype=torch.float32)
y = torch.tensor(data[target_column].values, dtype=torch.float32).unsqueeze(-1)

# ------------------
# Reshape Mamba data
# ------------------
sequence_length = 10
mamba_input_dim = 3
x_mamba = x_mamba.view(-1, sequence_length, mamba_input_dim)

# ------------------
# Train/Test Split
# ------------------
batch_size = x_kan.shape[0]
split_index = int(0.8 * batch_size)  # 80% training, 20% testing

x_kan_train, x_kan_test = x_kan[:split_index], x_kan[split_index:]
x_mamba_train, x_mamba_test = x_mamba[:split_index], x_mamba[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ------------------
# ARMA Model for Comparison
# ------------------
def train_arma(y_train, y_test):
    y_train_np = y_train.squeeze().cpu().numpy()
    y_test_np = y_test.squeeze().cpu().numpy()

    model = ARIMA(y_train_np, order=(2, 0, 2))
    arma_result = model.fit()

    forecast = arma_result.forecast(steps=5)
    y_test_last_5 = y_test_np[-5:]
    mape = np.mean(np.abs((y_test_last_5 - forecast) / y_test_last_5)) * 100
    print(f"ARMA MAPE (Last 5 Points): {mape:.2f}%")
    return forecast

# ------------------
# Evaluate Function (Last 5 Points)
# ------------------
def evaluate_last_five_points(model, x_kan_test, x_mamba_test, y_test):
    model.eval()
    with torch.no_grad():
        if isinstance(model, Mamba):
            test_pred = model(x_mamba_test)
        elif isinstance(model, DynamicKAN):
            test_pred = model(x_kan_test)
        else:
            test_pred = model(x_kan_test, x_mamba_test)

        avg_pred = torch.mean(test_pred[-5:], dim=0)
        avg_actual = torch.mean(y_test[-5:], dim=0)
        mape = torch.mean(torch.abs((avg_actual - avg_pred) / avg_actual)) * 100
    return mape.item()

# ------------------
# Define Model Classes
# ------------------
class DynamicWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicWeightGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_weight = nn.Linear(64, output_dim)
        self.fc_bias = nn.Linear(64, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        weight = self.softplus(self.fc_weight(h))
        bias = self.fc_bias(h)
        return weight, bias

class DynamicKAN(nn.Module):
    def __init__(self, input_dim, num_inner_units, num_outer_units):
        super(DynamicKAN, self).__init__()
        self.inner_weight_generators = nn.ModuleList([
            DynamicWeightGenerator(input_dim, input_dim) for _ in range(num_inner_units)
        ])
        self.outer_weight_generators = nn.ModuleList([
            DynamicWeightGenerator(num_inner_units, 1) for _ in range(num_outer_units)
        ])

    def forward(self, x):
        inner_outputs = []
        for generator in self.inner_weight_generators:
            weight, bias = generator(x)
            inner_output = torch.sum(weight * x + bias, dim=-1, keepdim=True)
            inner_outputs.append(inner_output)
        aggregated_inner_output = torch.cat(inner_outputs, dim=-1)

        outer_outputs = []
        for generator in self.outer_weight_generators:
            weight, bias = generator(aggregated_inner_output)
            outer_output = torch.sum(weight * aggregated_inner_output + bias, dim=-1, keepdim=True)
            outer_outputs.append(outer_output)

        final_output = torch.sum(torch.cat(outer_outputs, dim=-1), dim=-1, keepdim=True)
        return final_output

class Mamba(nn.Module):
    def __init__(self, input_dim, state_dim, sequence_length):
        super(Mamba, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(state_dim))
        self.B = nn.Parameter(torch.rand(input_dim, state_dim))
        self.C = nn.Parameter(torch.rand(1, state_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.state_dim, device=x.device)

        for t in range(x.shape[1]):
            u_t = x[:, t, :]
            h = torch.matmul(h, self.A) + torch.matmul(u_t, self.B)
        y = torch.matmul(h, self.C.T)
        return y

class HybridModel(nn.Module):
    def __init__(self, kan_input_dim, kan_inner_units, kan_outer_units,
                 mamba_input_dim, mamba_state_dim, sequence_length):
        super(HybridModel, self).__init__()
        self.kan = DynamicKAN(kan_input_dim, kan_inner_units, kan_outer_units)
        self.mamba = Mamba(mamba_input_dim, mamba_state_dim, sequence_length)
        self.weight_mamba = nn.Parameter(torch.tensor(0.5))
        self.weight_kan = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_kan, x_mamba):
        kan_output = self.kan(x_kan)
        mamba_output = self.mamba(x_mamba)
        combined = self.weight_kan * kan_output + self.weight_mamba * mamba_output
        return combined

# ------------------
# Early Stopping with MAPE
# ------------------
class EarlyStoppingWithMAPE:
    def __init__(self, patience=10, min_mape=3.0):
        self.patience = patience
        self.min_mape = min_mape
        self.mape_history = []
        self.early_stop = False

    def __call__(self, current_mape):
        self.mape_history.append(current_mape)
        if len(self.mape_history) > self.patience:
            self.mape_history.pop(0)
        if all(m < self.min_mape for m in self.mape_history):
            self.early_stop = True

# ------------------
# Prepare Models & Training Setup
# ------------------
criterion = nn.MSELoss()

kan_input_dim = x_kan.shape[1]
mamba_state_dim = 16
num_inner_units = 2
num_outer_units = 2

models = {
    "KAN Only": DynamicKAN(kan_input_dim, num_inner_units, num_outer_units),
    "Mamba Only": Mamba(mamba_input_dim, mamba_state_dim, sequence_length),
    "Mamba + KAN": HybridModel(
        kan_input_dim=kan_input_dim,
        kan_inner_units=num_inner_units,
        kan_outer_units=num_outer_units,
        mamba_input_dim=mamba_input_dim,
        mamba_state_dim=mamba_state_dim,
        sequence_length=sequence_length
    )
}

# ------------------
# Train & Evaluate
# ------------------
x_kan_train, x_kan_test = x_kan_train.to(device), x_kan_test.to(device)
x_mamba_train, x_mamba_test = x_mamba_train.to(device), x_mamba_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for name, model in models.items():
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    early_stopping = EarlyStoppingWithMAPE(patience=3, min_mape=3.0)
    print(f"\nTraining {name} with Early Stopping...\n")

    for epoch in range(70):
        model.train()
        optimizer.zero_grad()

        if name == "KAN Only":
            y_pred = model(x_kan_train)
        elif name == "Mamba Only":
            y_pred = model(x_mamba_train)
        else:
            y_pred = model(x_kan_train, x_mamba_train)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if name == "KAN Only":
                test_pred = model(x_kan_test)
            elif name == "Mamba Only":
                test_pred = model(x_mamba_test)
            else:
                test_pred = model(x_kan_test, x_mamba_test)

            test_mse = mean_squared_error(y_test.cpu(), test_pred.cpu())
            test_mae = mean_absolute_error(y_test.cpu(), test_pred.cpu())
            test_mape = evaluate_last_five_points(model, x_kan_test, x_mamba_test, y_test)

        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, "
            f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, "
            f"MAPE (Last 5 Points): {test_mape:.2f}%"
        )

        early_stopping(test_mape)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

# ------------------
# ARMA Results
# ------------------
arma_forecast = train_arma(y_train, y_test)