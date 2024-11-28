import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from PIL import Image
im = Image.open('icon.jpg')
st.set_page_config(page_title="Stock Predictor", page_icon=im, layout="wide")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

import yfinance as yf

page = st.selectbox("Choose your page", ["Stock Prediction", "Stock Symbols"]) 
if page == "Stock Prediction":
    def scrape_stock_data(ticker, start_date, end_date):
        # Scrape data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data

    st.image('icon.jpg', width=400)
    st.title('Stock Trend Prediction')

    if __name__ == "__main__":
        user_input = st.text_input('Enter Stock Ticker', 'AAPL')
        start_date = '2014-01-01'
        end_date = '2024-01-01'

        df = scrape_stock_data(user_input, start_date, end_date)

    st.subheader('Data from 2014 - 2024')
    st.write(df.describe()) 

   
    col1, col2 = st.columns(2)


    with col1:
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close)
        st.pyplot(fig)


    with col2:
        st.subheader('Closing Price and Moving Averages vs Time Chart')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, 'r', label='MA100')
        plt.plot(df.Close, 'g', label='Close Price')
        plt.plot(ma200, 'b', label='MA200')
        plt.legend()
        st.pyplot(fig)
        st.subheader("*According to expert Finance Analysts, when MA100( Moving average of 100) is above MA200( Moving average of 200), we see the trend of rising stock prices. In the case of vice versa, the trend is of falling prices.*")

    df = df.drop('Adj Close', axis=1)

    x = df.drop('Close', axis=1)
    y = df['Close']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    x_sc = ss.fit_transform(x_train)
    x_test_sc = ss.transform(x_test)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_sc, y_train)
    pred = lr.predict(x_test_sc)

    dc = {'Actual Values': y_test, 'Predicted Values': pred}
    new = pd.DataFrame(dc)
    new.sort_values(by='Actual Values', inplace=True)
    snew = new.sort_index()

    st.subheader("Actual vs Predicted")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(snew['Actual Values'], 'b', label='Original Price')
    plt.plot(snew['Predicted Values'], 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)


    st.sidebar.title("Input Values")
    u1 = st.sidebar.number_input("Enter Open Value", value=0.0, format="%.2f")
    u2 = st.sidebar.number_input("Enter High Value", value=0.0, format="%.2f")
    u3 = st.sidebar.number_input("Enter Low Value", value=0.0, format="%.2f")
    u4 = st.sidebar.number_input("Enter Volume", value=0.0, format="%.2f")


    input_data = pd.DataFrame({
        'Parameter': ['Open Value', 'High Value', 'Low Value', 'Volume'],
        'Value': [u1, u2, u3, u4]
    })
    st.subheader("Input Values")
    st.table(input_data)


    if st.button("Calculate"):
        v = [[u1, u2, u3, u4]]
        v = ss.transform(v)
        w = lr.predict(v)
        st.markdown(f"### The predicted closing value is: {w}")
    
elif page == "Stock Symbols":
    
    def get_all_tickers():
        all_tickers = yf.Tickers().tickers
        tickers_list = []
        company_names_list = []
        for ticker in all_tickers:
            tickers_list.append(ticker.ticker)
            company_names_list.append(ticker.info['longName'])
        return tickers_list, company_names_list
    
    st.title('Yahoo Finance Symbol and Company Name Table')
    tickers_list, company_names_list = get_all_tickers()
    data = pd.DataFrame({'Symbol': tickers_list, 'Company Name': company_names_list})
    st.write(data)