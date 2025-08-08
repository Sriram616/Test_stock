import streamlit as st
import numpy as np
import yfinance as yf
from stable_baselines3 import DQN

# Load RL model once when app starts
@st.cache(allow_output_mutation=True)
def load_model():
    model = DQN.load("dqn_stock_model.zip")
    return model

model = load_model()

st.title("Stock Trading RL Agent")

# Option 1: Manual price input
price_input = st.number_input("Enter current stock price manually", min_value=0.0, value=250.0)

# Option 2: Fetch live price from yfinance
symbol = st.text_input("Or enter stock ticker symbol to fetch live price", value="MSFT")
if symbol:
    data = yf.download(symbol, period="1d", interval="1m")
    if not data.empty:
        live_price = data['Close'][-1]
        st.write(f"Latest {symbol} price: ${live_price:.2f}")
    else:
        live_price = None
        st.write("Could not fetch live price.")
else:
    live_price = None

# Use manual price or live price if available
current_price = live_price if live_price is not None else price_input

# Prepare observation for model
obs = np.array([current_price], dtype=np.float32)

# Get model action
action, _ = model.predict(obs, deterministic=True)

actions = {0: "Hold", 1: "Buy", 2: "Sell"}
st.write(f"RL Agent recommends to: **{actions[int(action)]}** at price ${current_price:.2f}")