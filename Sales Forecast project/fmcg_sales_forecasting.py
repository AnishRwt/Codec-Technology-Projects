import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)


def load_and_preprocess(filepath="fmcg_sales.csv"):
    """Load raw transaction logs and aggregate to daily sales."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['revenue'] = df['units_sold'] * df['price_unit']

    daily = df.groupby('date').agg({
        'units_sold': 'sum',
        'revenue': 'sum',
        'promotion_flag': lambda x: 1 if (x == 1).any() else 0,
        'price_unit': 'mean'
    }).reset_index()

    daily = daily.sort_values('date').set_index('date').asfreq('D')
    daily['units_sold'] = daily['units_sold'].fillna(0)
    daily['revenue'] = daily['revenue'].fillna(0)
    daily['promotion_flag'] = daily['promotion_flag'].fillna(0)

    return daily


def train_sarimax(train_df, test_df):
    """Train SARIMAX (ARIMA with seasonal order and promo exogenous flag)."""
    model = SARIMAX(
        train_df['units_sold'],
        exog=train_df[['promotion_flag']],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    preds = model.predict(
        start=test_df.index[0],
        end=test_df.index[-1],
        exog=test_df[['promotion_flag']]
    )
    return preds


def train_prophet(train_df, test_df):
    """Train Facebook Prophet with multiplicative seasonality and promo regressor."""
    p_train = train_df.reset_index()[['date', 'units_sold', 'promotion_flag']].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )

    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_regressor('promotion_flag', mode='multiplicative')
    model.fit(p_train)

    p_test = test_df.reset_index()[['date', 'promotion_flag']].rename(columns={'date': 'ds'})
    forecast = model.predict(p_test)
    return forecast['yhat'].values


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(train_df, test_df, seq_len=14, epochs=50):
    """Train a simple PyTorch LSTM on sales and promotional flag."""
    scaler_y = MinMaxScaler()
    scaler_p = MinMaxScaler()

    train_y = scaler_y.fit_transform(train_df[['units_sold']])
    test_y = scaler_y.transform(test_df[['units_sold']])
    train_p = scaler_p.fit_transform(train_df[['promotion_flag']])
    test_p = scaler_p.transform(test_df[['promotion_flag']])

    train_data = np.hstack([train_y, train_p])
    full_data = np.vstack([train_data, np.hstack([test_y, test_p])])

    X_train, y_train = [], []
    for i in range(len(train_data) - seq_len):
        X_train.append(train_data[i:i+seq_len])
        y_train.append(train_data[i+seq_len, 0])

    X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    test_input = full_data[len(train_data) - seq_len:]
    preds = []
    with torch.no_grad():
        for i in range(len(test_df)):
            inp = torch.tensor(test_input[i:i+seq_len], dtype=torch.float32).unsqueeze(0)
            preds.append(model(inp).item())

    return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def print_metrics(y_true, predictions_dict):
    print("\n--- Model Evaluation (Test Horizon) ---")
    for name, pred in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        mape = mean_absolute_percentage_error(y_true, pred) * 100
        print(f"{name:15s} | RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | MAPE: {mape:5.2f}%")


if __name__ == "__main__":
    print("Loading data...")
    daily = load_and_preprocess("fmcg_sales.csv")

    test_days = 30
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    print(f"Training on {len(train)} days, testing on {len(test)} days.")

    print("Running ARIMA (SARIMAX)...")
    sarimax_preds = train_sarimax(train, test)

    print("Running Facebook Prophet...")
    prophet_preds = train_prophet(train, test)

    print("Running LSTM...")
    lstm_preds = train_lstm(train, test)

    print_metrics(test['units_sold'], {
        "ARIMA (SARIMAX)": sarimax_preds,
        "Prophet": prophet_preds,
        "LSTM": lstm_preds
    })
