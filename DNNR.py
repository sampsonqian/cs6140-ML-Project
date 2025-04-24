#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Regressor for NVDA Stock Prediction (Regression)

This script builds and trains a feed-forward neural network using PyTorch to
predict the next day's closing price of NVDA stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

rcParams['figure.figsize'] = (16, 8)

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Data Loading and Preprocessing -------------------- #
def load_and_preprocess_data(filepath):
    """
    Loads NVDA stock data from a CSV file with extra header rows,
    computes technical indicators, and creates the regression target
    (next day's closing price).
    """
    col_names = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(filepath, skiprows=3, header=None, names=col_names)

    print("Data after reading CSV (first 5 rows):")
    print(df.head())

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
    """
    Returns the feature matrix X and the regression target y.
    Features: Open, High, Low, Close, Volume, MA10, MA50, Return.
    Target: Next day's closing price.
    """
    features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "Return"]
    X = df[features].values
    y = df["Target_Reg"].values
    return X, y


# -------------------- PyTorch DNN Regressor Model Definition -------------------- #
class DNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# -------------------- Main Pipeline for DNN Regressor -------------------- #
def main():
    filepath = "NVIDIA_STOCK.csv"  # Update path as needed
    df = load_and_preprocess_data(filepath)

    X, y = prepare_features_targets(df)

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale features using StandardScaler
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled = feat_scaler.transform(X_test)

    # Scale target using MinMaxScaler
    targ_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = targ_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = targ_scaler.transform(y_test.reshape(-1, 1))

    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = X_train_scaled.shape[1]
    model = DNNRegressor(input_dim).to(DEVICE)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_losses):.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    pred_dnn = targ_scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_test, pred_dnn)
    r2 = r2_score(y_test, pred_dnn)
    print("DNN Regressor: MSE =", mse, "R2 =", r2)

    plt.figure(figsize=(16, 8))
    plt.plot(y_test, label="Actual", marker="o")
    plt.plot(pred_dnn, label="Predicted", marker="x")
    plt.title("DNN Regressor: Actual vs Predicted Closing Price")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
