# NVDA Stock Price Prediction

This project implements three regression models to predict **NVIDIA (NVDA)** next-day closing prices using historical stock data:

1. Polynomial Linear Regression
2. Feed-Forward Deep Neural Network (DNN)
3. Long Short-Term Memory (LSTM) Network

Each model captures different aspects of stock price behavior, from basic nonlinear patterns to complex feature interactions and temporal dependencies.

---

## Project Structure

```
nvda-stock-prediction/
├── NVIDIA_STOCK.csv      # Historical NVDA stock data (required)
├── PLR.py                # Polynomial Linear Regression model
├── PLR_Tunning.py        # PLR model tuning
├── DNNR.py               # Feed-Forward DNN model (PyTorch)
├── DNNR_Tunning.py       # DNNR model tuning
├── LSTM.py               # LSTM Sequence model (PyTorch)
├── LSTM_Tunning.py       # LSTM model tuning
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sampsonqian/cs6140-ML-Project.git
   cd cs6140-ML-Project(or your customer folder name)
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**  
   Place `NVIDIA_STOCK.csv` inside the folder.

---

## How to Run

### Polynomial Linear Regression

```bash
python PLR.py
```

### Feed-Forward DNN Regressor

```bash
python DNNR.py
```

### LSTM Sequence Model

```bash
python LSTM.py
```

Each script will output evaluation metrics (MSE, R²) and display a plot of actual vs. predicted prices.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- torch

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Notes

- Uses an 80/20 train-test split.
- LSTM and DNN benefit from GPU acceleration if available.
- Hyperparameters can be adjusted directly within each script.

---
