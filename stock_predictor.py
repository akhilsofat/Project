#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:21:21 2025

@author: akhil
"""

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime
import streamlit as st

# Define model and predictors
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Function to fetch and prepare data
@st.cache
def load_data(stock_ticker, start_date, end_date):
    df = yf.Ticker(stock_ticker).history(start=start_date, end=end_date)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df = df.dropna()
    return df

# Function to train the model
def train_model(data):
    train = data.iloc[:-100]
    model.fit(train[predictors], train["Target"])
    return model

# Function to make predictions
def predict_next_day(data, model):
    latest_data = data.iloc[-1]
    features = latest_data[predictors].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Increase" if prediction == 1 else "Decrease"

# Streamlit UI
st.title("Stock Price Predictor")
st.write("Predict whether a stock's price will increase or decrease tomorrow.")

# User inputs
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, ^GSPC):", value="AAPL")
start_date = st.date_input("Start Date:", value=datetime(2000, 1, 1))
end_date = st.date_input("End Date:", value=datetime.today())

# Load data and train model
if st.button("Predict"):
    with st.spinner("Loading data and training model..."):
        df = load_data(stock_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        trained_model = train_model(df)
        prediction = predict_next_day(df, trained_model)
        st.success(f"The model predicts the stock price will: **{prediction}** tomorrow.")

    # Display data and model accuracy
    st.subheader("Data Preview")
    st.write(df.tail())

    # Model accuracy
    test = df.iloc[-100:]
    preds = trained_model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    st.write(f"Model Precision Score: {precision:.2f}")
