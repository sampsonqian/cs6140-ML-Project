#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polynomial Linear Regression for NVDA Stock Prediction (Regression)
This script uses polynomial expansion and linear regression
with hyperparameter tuning to predict NVDA stock's next-day closing price.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
from tqdm import tqdm

rcParams['figure.figsize'] = (16, 8)

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

# -------------------- Plotting Function -------------------- #
def plot_results(results, metric="mse", top_n=10):
    sorted_results = sorted(results, key=lambda x: x[metric] if metric == "mse" else -x[metric])
    top_n = min(top_n, len(sorted_results))  # 🔧 Adjust if fewer results
    best_results = sorted_results[:top_n]

    labels = [f'Degree = {r["degree"]}' for r in best_results]
    values = [r[metric] for r in best_results]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(top_n), values, color="skyblue")
    plt.yticks(range(top_n), labels)
    plt.gca().invert_yaxis()
    plt.xlabel(metric.upper())
    plt.title(f"Top Polynomial Models by {metric.upper()}")

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

    degrees = [1, 2, 3, 4]
    results = []

    print(f"Running {len(degrees)} polynomial regression experiments...")
    for degree in tqdm(degrees):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        model = LinearRegression()
        model.fit(X_train_poly, y_train_scaled)
        pred_scaled = model.predict(X_test_poly)
        pred = targ_scaler.inverse_transform(pred_scaled)

        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results.append({"degree": degree, "mse": mse, "r2": r2})

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("plr_results.csv", index=False)
    print("\nAll PLR results saved to 'plr_results.csv'")

    # Print top results by MSE and R²
    top_mse = sorted(results, key=lambda r: r["mse"])[:10]
    print("\nTop Polynomial Models by MSE:")
    for i, r in enumerate(top_mse, 1):
        print(f"#{i}: Degree = {r['degree']} | MSE = {r['mse']:.4f} | R² = {r['r2']:.4f}")

    top_r2 = sorted(results, key=lambda r: -r["r2"])[:10]
    print("\nTop Polynomial Models by R²:")
    for i, r in enumerate(top_r2, 1):
        print(f"#{i}: Degree = {r['degree']} | R² = {r['r2']:.4f} | MSE = {r['mse']:.4f}")

    plot_results(results, metric="mse", top_n=10)
    plot_results(results, metric="r2", top_n=10)

if __name__ == "__main__":
    main()
