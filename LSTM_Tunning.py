#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project: NVDA Stock Price Prediction using LSTM (PyTorch Version)
Description: Builds and tunes an LSTM model using PyTorch to predict NVDA stock's next-day closing price.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

rcParams['figure.figsize'] = (20, 10)
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

# -------------------- Sequence Data Preparation -------------------- #
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# -------------------- LSTM Model Definition -------------------- #
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# -------------------- Single Experiment -------------------- #
def run_experiment(params, X_train_seq, y_train_seq, X_test_seq, y_test_seq, targ_scaler, device):
    input_size = X_train_seq.shape[2]
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = {
        "adam": optim.Adam(model.parameters(), lr=params["lr"]),
        "rmsprop": optim.RMSprop(model.parameters(), lr=params["lr"])
    }[params["optimizer"]]

    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32).to(device),
                                  torch.tensor(y_train_seq, dtype=torch.float32).to(device))
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
        predictions = model(torch.tensor(X_test_seq, dtype=torch.float32).to(device)).cpu().numpy()

    predicted = targ_scaler.inverse_transform(predictions)
    actual = targ_scaler.inverse_transform(y_test_seq)

    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {
        "params": params,
        "mse": mse,
        "r2": r2
    }

# -------------------- Plotting Function -------------------- #
def plot_results(results, metric="mse", top_n=10):
    sorted_results = sorted(results, key=lambda x: x[metric] if metric == "mse" else -x[metric])
    best_results = sorted_results[:top_n]

    labels = [
        f'{r["params"]["optimizer"]}, lr={r["params"]["lr"]}, bs={r["params"]["batch_size"]}, hs={r["params"]["hidden_size"]}, nl={r["params"]["num_layers"]}'
        for r in best_results
    ]
    values = [r[metric] for r in best_results]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(top_n), values, color="skyblue")
    plt.yticks(range(top_n), labels)
    plt.gca().invert_yaxis()
    plt.xlabel(metric.upper())
    plt.title(f"Top {top_n} LSTM Models by {metric.upper()}")

    for bar, val in zip(bars, values):
        plt.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va='center')

    plt.tight_layout()
    plt.show()

# -------------------- Main Pipeline -------------------- #
def main():
    df = load_and_preprocess_data("NVIDIA_STOCK.csv")

    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return']
    X = df[feature_columns].values
    y = df['Target_Reg'].values

    feat_scaler = MinMaxScaler()
    X_scaled = feat_scaler.fit_transform(X)

    targ_scaler = MinMaxScaler()
    y_scaled = targ_scaler.fit_transform(y.reshape(-1, 1))

    train_size = int(len(df) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y_scaled[:train_size]
    y_test = y_scaled[train_size:]

    time_steps = 60
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    param_grid = {
        "optimizer": ["adam", "rmsprop"],
        "lr": [0.0001, 0.001, 0.005],
        "batch_size": [32, 64],
        "hidden_size": [64, 100],
        "num_layers": [1, 2],
        "epochs": [20]
    }

    param_combos = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []

    print(f"Running {len(param_combos)} experiments...")
    for combo in tqdm(param_combos):
        params = dict(zip(param_names, combo))
        result = run_experiment(params, X_train_seq, y_train_seq, X_test_seq, y_test_seq, targ_scaler, DEVICE)
        results.append(result)

    results_df = pd.DataFrame([{
        **r["params"],
        "mse": r["mse"],
        "r2": r["r2"]
    } for r in results])

    results_df.to_csv("lstm_hyperparameter_results.csv", index=False)
    print("\nAll results saved to 'lstm_hyperparameter_results.csv'")

    top_mse_results = sorted(results, key=lambda r: r["mse"])[:10]
    print("\nTop 10 LSTM Experiments by MSE:")
    for i, res in enumerate(top_mse_results, 1):
        print(f"#{i}: Params: {res['params']} | MSE: {res['mse']:.4f} | R²: {res['r2']:.4f}")

    top_r2_results = sorted(results, key=lambda r: -r["r2"])[:10]
    print("\nTop 10 LSTM Experiments by R²:")
    for i, res in enumerate(top_r2_results, 1):
        print(f"#{i}: Params: {res['params']} | R²: {res['r2']:.4f} | MSE: {res['mse']:.4f}")

    plot_results(results, metric="mse", top_n=10)
    plot_results(results, metric="r2", top_n=10)

if __name__ == "__main__":
    main()
