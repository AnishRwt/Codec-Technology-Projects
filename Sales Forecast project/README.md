# Retail FMCG Sales Forecasting

A predictive data analysis project to forecast daily sales for retail Fast-Moving Consumer Goods (FMCG) products using time-series modeling.

The goal of this project is to analyze historical sales patterns and predict future demand while accounting for:
- Long-term trends
- Weekly and seasonal shopping cycles
- Promotional events and price discounts

---

## Dataset Overview

The dataset contains daily transaction logs for retail FMCG products with the following features:
- `date`: Transaction date (DD-MM-YYYY)
- `units_sold`: Target variable (daily units purchased)
- `price_unit`: Selling price per unit
- `promotion_flag`: Binary indicator (1 if a promotion was active, 0 otherwise)
- Categorical features: `channel` (Retail, Discount, E-commerce), `region`, `pack_type`, and `category` (Milk/Dairy)

For forecasting, data was aggregated to a continuous daily time series of total units sold and active promotions.

---

## Exploratory Data Analysis & Key Insights

1. **Promotional Lift**:
   - Average daily sales without promotion: **192 units**
   - Average daily sales with promotion: **3,538 units**
   - Active promotions drive an average uplift of **+1,742%**, making promotional scheduling the most significant factor in demand spikes.

2. **Weekly Seasonality**:
   - Demand consistently peaks on **Mondays (+3.1%)** and **Thursdays (+1.5%)**, corresponding to distributor and store restocking schedules.
   - Weekend and midweek sales remain relatively flat.

3. **Annual Cycles**:
   - Strong category seasonality with peak demand during late spring/summer (**May to July**) and lower volumes during winter (**November to January**).

---

## Models Implemented

Three different time-series modeling approaches were built and evaluated on an out-of-sample 30-day test horizon (December 2024):

1. **ARIMA (SARIMAX)**:
   - Configured with seasonal parameters `(1, 1, 1) x (1, 1, 1, 7)` to capture weekly autocorrelation.
   - Included `promotion_flag` as an exogenous variable (`exog`) to account for promotional surges.

2. **Facebook Prophet**:
   - Configured with `seasonality_mode='multiplicative'` to handle large promotional spikes.
   - Incorporated weekly seasonality, annual seasonality, and an external binary regressor for promotions.

3. **LSTM Neural Network (PyTorch)**:
   - Built a sequence-to-one recurrent neural network using a 14-day sliding window.
   - Fed normalized historical sales and promotional flags to predict next-day sales.

---

## Results & Performance Comparison

The models were evaluated against actual sales on the test dataset using Root Mean Squared Error (RMSE):

| Model | Test RMSE (Units) | Key Observations |
| :--- | :---: | :--- |
| **Facebook Prophet** | **173.6** | **Best overall**. Multiplicative regressor adapted smoothly to the heavy December promotional period. |
| **ARIMA (SARIMAX)** | **175.6** | Performed closely to Prophet. Captures weekly autocorrelation and promotional shifts reliably. |
| **LSTM Neural Network** | **292.7** | Captured overall direction, but had higher variance across the 30-day horizon due to limited training sequence length. |

### Forecast Dashboard
![Executive Dashboard](assets/fmcg_executive_dashboard.png)

---

## Business & Supply Chain Takeaways

- **Inventory Coordination**: Because promotions increase demand by over 10x, promotional calendars must be shared with warehouse teams at least one week in advance to prevent stockouts.
- **Delivery Scheduling**: Logistics and restocking dispatches should prioritize Sunday evenings and Wednesday afternoons ahead of the Monday and Thursday order peaks.
- **Seasonality Planning**: Inventory buffers should be scaled down in Q4 and gradually built up starting in April for the summer peak.

---

## Project Structure

```text
├── assets/
│   ├── fmcg_executive_dashboard.png   # Full summary dashboard
│   ├── forecast_vs_actual.png         # Forecast comparison plot
│   └── prophet_components.png         # Trend and seasonality decomposition
├── FMCG_Sales_Forecasting.ipynb       # Google Colab / Jupyter notebook
├── fmcg_sales_forecasting.py          # Python pipeline script
├── README.md                          # Project documentation
└── .gitignore
```

---

## How to Run

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `FMCG_Sales_Forecasting.ipynb`.
3. Upload your dataset file as `fmcg_sales.csv`.
4. Click **Runtime > Run all**.

### Option 2: Local Python Environment
Ensure dependencies are installed:
```bash
pip install numpy pandas matplotlib seaborn statsmodels prophet torch scikit-learn
```

Run the pipeline:
```bash
python fmcg_sales_forecasting.py
```
