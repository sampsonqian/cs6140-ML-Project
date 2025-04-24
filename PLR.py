#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polynomial Linear Regression for NVDA Stock Prediction (Regression)

This script uses a degree-2 polynomial expansion along with Linear Regression
to predict the next day's closing price of NVDA stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

rcParams['figure.figsize'] = (16, 8)


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

    # Convert Date to datetime and sort
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

    df.dropna(inplace=True)

    # Regression target: next day's closing price
    df["Target_Reg"] = df["Close"].shift(-1)
    df = df.iloc[:-1]  # Remove last row (no next day value)

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


# -------------------- Main Pipeline -------------------- #
def main():
    filepath = "NVIDIA_STOCK.csv"  # Update this path if necessary
    df = load_and_preprocess_data(filepath)
    print("Dataset loaded. Shape:", df.shape)

    X, y = prepare_features_targets(df)

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale features using StandardScaler
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled = feat_scaler.transform(X_test)

    # Scale target using MinMaxScaler (for easier inversion later)
    targ_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = targ_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = targ_scaler.transform(y_test.reshape(-1, 1))

    # Apply polynomial transformation (degree 2)
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Train Linear Regression model on polynomial features
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train_scaled)
    pred_poly_scaled = lin_reg.predict(X_test_poly)
    pred_poly = targ_scaler.inverse_transform(pred_poly_scaled)

    mse = mean_squared_error(y_test, pred_poly)
    r2 = r2_score(y_test, pred_poly)
    print("Polynomial Linear Regression: MSE =", mse, "R2 =", r2)

    # Plot actual vs. predicted closing prices
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, label="Actual", marker="o")
    plt.plot(pred_poly, label="Predicted", marker="x")
    plt.title("Polynomial Regression: Actual vs Predicted Closing Price")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
