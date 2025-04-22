#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Regressor for NVDA Stock Prediction (Regression)
This script builds and tunes a feed-forward neural network using PyTorch to
predict the next day's closing price of NVDA stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
from tqdm import tqdm

rcParams['figure.figsize'] = (16, 8)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# -------------------- Data Loading and Preprocessing -------------------- #
def load_and_preprocess_data(filepath):
    col_names = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(filepath, skiprows=3, header=None, names=col_names)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    df["Target_Reg"] = df["Close"].shift(-1)
    df = df.iloc[:-1]
    return df

def prepare_features_targets(df):
    features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "Return"]
    X = df[features].values
    y = df["Target_Reg"].values
    return X, y

# -------------------- DNN Regressor Definition -------------------- #
class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# -------------------- Run Single Experiment -------------------- #
def run_experiment(params, X_train_tensor, y_train_tensor, X_test_tensor, y_test, targ_scaler):
    model = DNNRegressor(
        input_dim=X_train_tensor.shape[1],
        hidden_sizes=params["hidden_sizes"],
        dropout_rate=params["dropout"]
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer_cls = optim.Adam if params["optimizer"] == "adam" else optim.RMSprop
    optimizer = optimizer_cls(model.parameters(), lr=params["lr"])

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    model.train()
    for epoch in range(params["epochs"]):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    pred_dnn = targ_scaler.inverse_transform(predictions)
    mse = mean_squared_error(y_test, pred_dnn)
    r2 = r2_score(y_test, pred_dnn)
    return {"params": params, "mse": mse, "r2": r2}

# -------------------- Plotting Function -------------------- #
def plot_results(results, metric="mse", top_n=10):
    sorted_results = sorted(results, key=lambda x: x[metric] if metric == "mse" else -x[metric])
    best_results = sorted_results[:top_n]

    labels = [
        f'{r["params"]["optimizer"]}, lr={r["params"]["lr"]}, bs={r["params"]["batch_size"]}, hs={r["params"]["hidden_sizes"]}, do={r["params"]["dropout"]}'
        for r in best_results
    ]
    values = [r[metric] for r in best_results]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(top_n), values, color="skyblue")
    plt.yticks(range(top_n), labels)
    plt.gca().invert_yaxis()
    plt.xlabel(metric.upper())
    plt.title(f"Top {top_n} DNN Models by {metric.upper()}")

    for bar, val in zip(bars, values):
        plt.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va='center')

    plt.tight_layout()
    plt.show()

# -------------------- Main Pipeline -------------------- #
def main():
    filepath = "NVIDIA_STOCK.csv"
    df = load_and_preprocess_data(filepath)
    X, y = prepare_features_targets(df)

    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled = feat_scaler.transform(X_test)

    targ_scaler = MinMaxScaler()
    y_train_scaled = targ_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = targ_scaler.transform(y_test.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    # Hyperparameter grid
    param_grid = {
        "optimizer": ["adam", "rmsprop"],
        "lr": [0.0001, 0.001, 0.005],
        "batch_size": [32, 64],
        "hidden_sizes": [(64, 32), (128, 64)],
        "dropout": [0.2, 0.3],
        "epochs": [100]
    }

    param_combos = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []

    print(f"Running {len(param_combos)} DNNR experiments...")
    for combo in tqdm(param_combos):
        params = dict(zip(param_names, combo))
        result = run_experiment(params, X_train_tensor, y_train_tensor, X_test_tensor, y_test, targ_scaler)
        results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame([{
        **r["params"],
        "mse": r["mse"],
        "r2": r["r2"]
    } for r in results])

    results_df.to_csv("dnnr_hyperparameter_results.csv", index=False)
    print("\nAll DNNR results saved to 'dnnr_hyperparameter_results.csv'")

    # Print top 10 results by MSE and R² in terminal
    top_mse_results = sorted(results, key=lambda r: r["mse"])[:10]
    print("\nTop 10 DNN Experiments by MSE:")
    for i, res in enumerate(top_mse_results, 1):
        print(f"#{i}: Params: {res['params']} | MSE: {res['mse']:.4f} | R²: {res['r2']:.4f}")

    top_r2_results = sorted(results, key=lambda r: -r["r2"])[:10]
    print("\nTop 10 DNN Experiments by R²:")
    for i, res in enumerate(top_r2_results, 1):
        print(f"#{i}: Params: {res['params']} | R²: {res['r2']:.4f} | MSE: {res['mse']:.4f}")

    # Plotting
    plot_results(results, metric="mse", top_n=10)
    plot_results(results, metric="r2", top_n=10)

if __name__ == "__main__":
    main()
