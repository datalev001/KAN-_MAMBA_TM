import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import torch.nn.functional as F

# ------------------
# Data Configuration
# ------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_cols, target_cols, lag=5):
        self.lag = lag
        self.inputs = df[input_cols].values
        self.targets = df[target_cols].values

    def __len__(self):
        return len(self.inputs) - self.lag

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx: idx + self.lag], dtype=torch.float32),
            torch.tensor(self.targets[idx + self.lag], dtype=torch.float32),
        )

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])

# Mamba Only Model
class MambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MambaModel, self).__init__()
        # State transition (GRU approximates A and B matrices dynamically)
        self.state_transition = nn.GRU(input_size, hidden_size, batch_first=True)
        # Emission function (approximates C matrix)
        self.emission = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Capture state transitions
        _, state = self.state_transition(x)  # state: (1, batch_size, hidden_size)
        # Map hidden state to observations
        output = self.emission(state[-1])  # Use the last hidden state
        return output

# KAN Model with Dynamic Weights
class KANModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KANModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer to transform input features
        self.dynamic_weights = nn.Parameter(torch.rand(hidden_size))  # Learnable dynamic weights
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer to produce predictions

    def forward(self, x):
        # Flatten the input along the time axis
        x_flattened = x.view(x.size(0), -1)  # (batch_size, time_steps * input_size)
        # Apply the first fully connected layer
        out = torch.relu(self.fc1(x_flattened))  # (batch_size, hidden_size)
        # Apply dynamic weights
        out = out * self.dynamic_weights  # Element-wise multiplication
        # Produce final output
        out = self.fc2(out)  # (batch_size, output_size)
        return out

# GNN Model (Fixed)
class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, graph_size):
        super(GNNModel, self).__init__()
        # Learnable adjacency matrix
        self.graph_weights = nn.Parameter(torch.rand(graph_size, graph_size))
        # Linear transformation layers
        self.fc_transform = nn.Linear(input_size, graph_size)
        self.fc_hidden = nn.Linear(graph_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten time-series input into node features
        batch_size, time_steps, feature_size = x.size()  # (batch_size, time_steps, input_size)
        node_features = x.mean(dim=1)  # Aggregate across time steps to create node embeddings (batch_size, input_size)

        # Linear transformation to graph size
        graph_features = self.fc_transform(node_features)  # (batch_size, graph_size)

        # Graph interaction: graph_features @ graph_weights
        graph_interaction = torch.matmul(graph_features, self.graph_weights)  # (batch_size, graph_size)

        # Further transform the graph interaction output
        hidden_features = torch.relu(self.fc_hidden(graph_interaction))  # (batch_size, hidden_size)

        # Final output transformation
        output = self.fc_output(hidden_features)  # (batch_size, output_size)
        return output

# ARMA Model for Comparison
def train_arma(train_data, test_data):
    individual_mapes = []
    for i in range(train_data.shape[1]):  # Loop over each product series
        arma_model = ARIMA(train_data[:, i], order=(2, 0, 2))
        arma_result = arma_model.fit()
        forecast = arma_result.forecast(steps=test_data.shape[0])

        # Calculate MAPE for the current series
        mape = mean_absolute_percentage_error(test_data[:, i], forecast)
        individual_mapes.append(mape)

    overall_mape = np.mean(individual_mapes)  # Average MAPE across all series
    return individual_mapes, overall_mape

# Function to calculate MAPE for multiple time windows
def calculate_mape_for_windows(predictions, targets, windows):
    results = {}
    for window in windows:
        avg_preds = predictions[-window:].mean(axis=0)
        avg_targets = targets[-window:].mean(axis=0)
        mapes = [
            mean_absolute_percentage_error([avg_targets[i]], [avg_preds[i]]) for i in range(len(target_cols))
        ]
        overall_mape = np.mean(mapes)

        print(f"\nMAPE Results for the Last {window} Points:")
        for i, mape in enumerate(mapes):
            print(f"  product_{i+1}: {mape:.4f}")
        print(f"  Overall MAPE: {overall_mape:.4f}")

        results[window] = {
            "individual_mapes": mapes,
            "overall_mape": overall_mape
        }
    return results

# Function to calculate MAPE for ARMA with multiple window sizes
def calculate_arma_mape_for_windows(test_data, forecasts, windows):
    # Ensure forecasts is a NumPy array
    if isinstance(forecasts, tuple):
        forecasts = np.array(forecasts)

    results = {}
    for window in windows:
        avg_preds = forecasts[-window:].mean(axis=0)
        avg_targets = test_data[-window:].mean(axis=0)
        
        # Calculate MAPE for each series
        mapes = [
            mean_absolute_percentage_error([avg_targets[i]], [avg_preds[i]])
            for i in range(test_data.shape[1])
        ]
        results[f"last_{window}_points"] = {
            "individual_mapes": mapes,
            "overall_mape": np.mean(mapes),
        }
    return results

# ------------------
# Train and Evaluate Functions
# ------------------
def train_model(model, loader, optimizer, criterion, epochs=80):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            preds.append(pred.numpy())
            targets.append(y_batch.numpy())
    return np.vstack(preds), np.vstack(targets)

# ------------------
# Initialize and Train the Models
# ------------------
data = pd.read_csv('product_sales.csv')
data['date'] = pd.to_datetime(data['date'])

sequence_length = 10
input_cols = [f"product_{j+1}" for j in range(5)]
target_cols = input_cols
train_split = len(data) - 5

train_df = data.iloc[:train_split]
val_df = data.iloc[train_split - sequence_length:]

train_dataset = TimeSeriesDataset(train_df, input_cols, target_cols, sequence_length)
val_dataset = TimeSeriesDataset(val_df, input_cols, target_cols, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

input_size = len(input_cols)
hidden_size = 64
output_size = len(target_cols)
graph_size = len(input_cols)

######################Train and predict ########################

#Train GRU Model
gru_model = GRUModel(input_size, hidden_size, output_size)
gru_criterion = nn.MSELoss()
gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
print("Training GRU Model...")
train_model(gru_model, train_loader, gru_optimizer, gru_criterion, epochs=1000)

# Train Mamba Model
mamba_model = MambaModel(input_size, hidden_size, output_size)
mamba_optimizer = torch.optim.Adam(mamba_model.parameters(), lr=0.0002)
mamba_criterion = nn.MSELoss()
print("Training Mamba Model...")
train_model(mamba_model, train_loader, mamba_optimizer, mamba_criterion, epochs=1000)

# Train KAN Model
# Instantiate and Train KAN Model
kan_model = KANModel(input_size=len(input_cols) * sequence_length, hidden_size=64, output_size=len(target_cols))
kan_criterion = nn.MSELoss()
kan_optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.001)
print("Training KAN Model...")
train_model(kan_model, train_loader, kan_optimizer, kan_criterion, epochs=500)

# Train GNN Model
gnn_model = GNNModel(input_size, hidden_size, output_size, graph_size)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.0002)
gnn_criterion = nn.MSELoss()
print("Training GNN Model...")
train_model(gnn_model, train_loader, gnn_optimizer, gnn_criterion, epochs=1000)

# Evaluate Models
preds_gru, targets_gru = evaluate_model(gru_model, val_loader)
preds_mamba, targets_mamba = evaluate_model(mamba_model, val_loader)
preds_kan, targets_kan = evaluate_model(kan_model, val_loader)
preds_gnn, targets_gnn = evaluate_model(gnn_model, val_loader)

# Calculate MAPE for each model
print("Evaluating GRU Model...")
gru_results = calculate_mape_for_windows(preds_gru, targets_gru, [3, 5])

print("Evaluating Mamba Model...")
mamba_results = calculate_mape_for_windows(preds_mamba, targets_mamba, [3, 5])

print("Evaluating KAN Model...")
kan_results = calculate_mape_for_windows(preds_kan, targets_kan, [3, 5])

print("Evaluating GNN Model...")
gnn_results = calculate_mape_for_windows(preds_gnn, targets_gnn, [3, 5])

# Benchmark: ARMA Model
arma_train_data = train_df.drop(columns=['date']).values
arma_test_data = val_df.drop(columns=['date']).values
arma_individual_mapes, arma_overall_mape = train_arma(arma_train_data, arma_test_data)
arma_train_data = train_df.drop(columns=['date']).values
arma_test_data = val_df.drop(columns=['date']).values

# Perform ARMA forecasting
arma_forecasts = []
for i in range(arma_train_data.shape[1]):  # Loop over each series
    arma_model = ARIMA(arma_train_data[:, i], order=(2, 0, 2))
    arma_result = arma_model.fit()
    forecast = arma_result.forecast(steps=arma_test_data.shape[0])
    arma_forecasts.append(forecast)

# Transpose to align with test data shape
arma_forecasts = np.array(arma_forecasts).T  

# Calculate MAPEs for ARMA
arma_results = calculate_arma_mape_for_windows(arma_test_data, arma_forecasts, [3, 5, 10])

# Print ARMA Results
for window, result in arma_results.items():
    print(f"ARMA MAPE for {window}:")
    for i, mape in enumerate(result["individual_mapes"]):
        print(f"  product_{i+1}: {mape:.4f}")
    print(f"  Overall MAPE: {result['overall_mape']:.4f}")