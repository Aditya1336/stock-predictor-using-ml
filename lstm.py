# This branch makes use of lstm instead of linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Load and set page configuration
im = Image.open('icon.jpg')
st.set_page_config(page_title="Stock Predictor", page_icon=im, layout="wide")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

def scrape_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

st.image('icon.jpg', width=400)
st.title('Stock Trend Prediction Using LSTM')
st.markdown("[Y Finance Reference](https://finance.yahoo.com)")

if __name__ == "__main__":
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.text_input('Enter Start Date', 'YYYY-MM-DD')
    end_date = st.text_input('Enter End Date', 'YYYY-MM-DD')
    df = scrape_stock_data(user_input, start_date, end_date)

st.subheader('Data from 2014 - 2024')
st.write(df.describe())

# Data preparation for LSTM
data_close = df['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close.values.reshape(-1, 1))

# Create sequences
window_size = 60
x_data, y_data = [], []
for i in range(window_size, len(scaled_data)):
    x_data.append(scaled_data[i-window_size:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# Split the data
train_size = int(len(x_data) * 0.8)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predictions
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualizing results
st.subheader("Closing Price vs Predictions")
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(data_close, label="Actual Price", color='blue')
plt.plot(range(window_size, train_size + window_size), train_predictions, label="Train Predictions", color='green')
plt.plot(range(train_size + window_size, len(data_close)), test_predictions, label="Test Predictions", color='red')
plt.legend()
st.pyplot(fig)

# Real-time prediction
st.sidebar.title("Input Values for Prediction")
u1 = st.sidebar.number_input("Enter Open Value", value=0.0, format="%.2f")
u2 = st.sidebar.number_input("Enter High Value", value=0.0, format="%.2f")
u3 = st.sidebar.number_input("Enter Low Value", value=0.0, format="%.2f")
u4 = st.sidebar.number_input("Enter Volume", value=0.0, format="%.2f")

if st.button("Calculate"):
    recent_data = scaled_data[-window_size:]  # Last 'window_size' days
    recent_data = np.append(recent_data, [[u1], [u2], [u3], [u4]])
    recent_data = recent_data[-window_size:]
    recent_data = recent_data.reshape(1, window_size, 1)
    prediction = model.predict(recent_data)
    predicted_price = scaler.inverse_transform(prediction)
    st.markdown(f"### The predicted closing value is: {predicted_price[0][0]:.2f}")
