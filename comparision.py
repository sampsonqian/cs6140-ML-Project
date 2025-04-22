import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# DNN Model
class DNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# LSTM Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, 100, 1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def load_and_preprocess(filepath):
    col_names = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
    df = pd.read_csv(filepath, skiprows=3, header=None, names=col_names)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    df["Target_Reg"] = df["Close"].shift(-1)
    df = df.iloc[:-1]
    return df

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def report(name, true, pred):
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")
    return mse, r2

def main():
    df = load_and_preprocess("NVIDIA_STOCK.csv")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'Return']
    X = df[features].values
    y = df['Target_Reg'].values

    feat_scaler = StandardScaler()
    X_scaled = feat_scaler.fit_transform(X)

    targ_scaler = MinMaxScaler()
    y_scaled = targ_scaler.fit_transform(y.reshape(-1, 1))

    train_size = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    y_test_actual = y[train_size:]

    # -------------------- DNN --------------------
    dnn_model = DNNRegressor(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.005)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(DEVICE),
                                  torch.tensor(y_train, dtype=torch.float32).to(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    dnn_model.train()
    for epoch in range(100):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = dnn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    dnn_model.eval()
    with torch.no_grad():
        dnn_pred = dnn_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    dnn_pred = targ_scaler.inverse_transform(dnn_pred)

    # -------------------- LSTM --------------------
    time_steps = 60
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    y_test_actual_seq = y[train_size + time_steps:]

    lstm_model = LSTMRegressor(input_size=X_train_seq.shape[2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE),
                                  torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    lstm_model.train()
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_pred = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    lstm_pred = targ_scaler.inverse_transform(lstm_pred)

    # -------------------- PLR --------------------
    poly = PolynomialFeatures(degree=1)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)
    plr_pred_scaled = lin_reg.predict(X_test_poly)
    plr_pred = targ_scaler.inverse_transform(plr_pred_scaled.reshape(-1, 1))

    # -------------------- Results --------------------
    print("\n--- Final Model Comparison ---")
    report("DNNR", y_test_actual, dnn_pred)
    report("LSTM", y_test_actual_seq, lstm_pred)
    report("PLR", y_test_actual, plr_pred)

    # -------------------- Plot --------------------
    plt.figure(figsize=(18, 8))
    plt.plot(y_test_actual, label="Actual", color="black", linestyle="--")
    plt.plot(dnn_pred, label="DNNR", alpha=0.8)
    plt.plot(np.arange(time_steps, len(y_test_actual)), lstm_pred, label="LSTM", alpha=0.8)
    plt.plot(plr_pred, label="PLR", alpha=0.8)
    plt.title("Model Predictions vs Actual Closing Prices")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
