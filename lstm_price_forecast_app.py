import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 LSTM Price Forecasting App")
st.markdown("""
Upload a time series dataset with `Date` and `Close` columns.
This app will train a simple LSTM neural network and forecast the next few days of prices.
""")

st.header("1. Upload CSV or Excel File")
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(uploaded_file, parse_dates=["Date"])

    st.subheader("2. 📄 Raw Data Preview")
    st.dataframe(df.tail())

    st.subheader("3. 📉 Closing Price Chart")
    fig, ax = plt.subplots()
    df.set_index("Date")["Close"].plot(ax=ax, figsize=(10, 5))
    st.pyplot(fig)

    st.subheader("4. 🧪 Train LSTM and Forecast")
    df_lstm = df[["Close"]].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)

    def create_sequences(data, seq_len=5):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    sequence_length = 5
    X, y = create_sequences(scaled_data, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=1, verbose=0)
    st.success("✅ LSTM Model Trained")

    st.subheader("5. 🔮 Forecasted Prices")
    input_seq = scaled_data[-sequence_length:]
    predictions = []

    for _ in range(5):
        pred_input = input_seq[-sequence_length:].reshape(1, sequence_length, 1)
        pred = model.predict(pred_input, verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq, pred, axis=0)

    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=5)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Close": forecast_prices})
    st.dataframe(forecast_df)

    st.subheader("6. 💾 Save Forecast")
    today = date.today().isoformat()
    os.makedirs("forecasts", exist_ok=True)
    file_path = f"forecasts/lstm_price_forecast_{today}.csv"
    forecast_df.to_csv(file_path, index=False)
    st.success(f"📁 Forecast saved to {file_path}")
else:
    st.warning("👆 Please upload a file to begin.")
