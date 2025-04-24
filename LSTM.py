#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project: NVDA Stock Price Prediction using LSTM (PyTorch Version)
Description: Builds an improved LSTM model using PyTorch to predict NVDA stock's next-day closing price.
Author: Your Name
Date: YYYY-MM-DD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

rcParams['figure.figsize'] = (20, 10)

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Data Loading and Preprocessing -------------------- #
def load_and_preprocess_data(filepath):
    """
    Loads NVDA stock data from a CSV file with extra header rows.
    Skips the first three rows, assigns column names, selects:
      Date, Open, High, Low, Close, Volume,
    computes technical indicators (MA10, MA50, Return),
    and creates the regression target (next day's closing price).
    """
    col_names = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(filepath, skiprows=3, header=None, names=col_names)

    print("Data after reading CSV (first 5 rows):")
    print(df.head())

    # Convert Date to datetime and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # Select required columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute technical indicators
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Return"] = df["Close"].pct_change()

    # Drop rows with NaN values (from rolling calculations)
    df.dropna(inplace=True)

    # Regression target: next day's closing price
    df["Target_Reg"] = df["Close"].shift(-1)
    df = df.iloc[:-1]  # Remove last row (no next day value)

    return df


# -------------------- Sequence Data Preparation for LSTM -------------------- #
def create_sequences(X, y, time_steps=60):
    """
    Converts 2D array X and target y into 3D sequences for LSTM.
    Each sequence consists of 'time_steps' consecutive rows from X,
    with the corresponding target y being the value immediately after the sequence.
    """
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
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        # Get output from the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# -------------------- Visualization Function -------------------- #
def plot_regression_results(actual, predicted, title="Actual vs Predicted Closing Price"):
    plt.figure(figsize=(20, 10))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(predicted, label="Predicted", marker="x")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Closing Price")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------- Main Pipeline -------------------- #
def main():
    data_filepath = "NVIDIA_STOCK.csv"
    df = load_and_preprocess_data(data_filepath)

    # Use the features: Open, High, Low, Close, Volume, MA10, MA50, Return
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return']
    X = df[feature_columns].values
    y = df['Target_Reg'].values

    # Scale features and target separately using MinMaxScaler
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = feat_scaler.fit_transform(X)

    targ_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = targ_scaler.fit_transform(y.reshape(-1, 1))

    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y_scaled[:train_size]
    y_test = y_scaled[train_size:]

    # Create sequences for LSTM using a look-back window of 60 days
    time_steps = 60
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    print("LSTM train sequence shape:", X_train_seq.shape)
    print("LSTM test sequence shape:", X_test_seq.shape)

    # Convert sequences to torch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_size = X_train_seq.shape[2]  # number of features per time step
    hidden_size = 100
    num_layers = 2
    output_size = 1

    # Build the LSTM regressor model
    model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(DEVICE)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_losses):.6f}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    # Inverse transform predictions and actual values
    predicted = targ_scaler.inverse_transform(predictions)
    actual = targ_scaler.inverse_transform(y_test_seq)

    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print("LSTM Regressor: MSE =", mse, "R2 =", r2)

    plot_regression_results(actual, predicted,
                            title="NVDA LSTM Regression (PyTorch): Actual vs Predicted Closing Price")


if __name__ == '__main__':
    main()