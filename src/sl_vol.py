import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from datetime import datetime
from ivol import fetch_options_data,plot_vol_surface

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
st.set_page_config(page_title="IV Explorer", layout="wide")
st.title("Options Implied Volatility Surface")
ticker_input = st.sidebar.text_input("Enter Ticker", value="AAPL").upper()

if ticker_input:
    csv_path = os.path.join(DATA_DIR, f"{ticker_input}_options.csv")
    if not os.path.exists(csv_path):
        st.warning(f"Data for {ticker_input} not found. Fetching options data...")
        df, out = fetch_options_data(ticker_input,risk_free_rate=0.05)
        st.success(f"Data fetched and saved to {out} with risk-free rate 0.05")
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Data loaded from {csv_path}")
        hist = yf.Ticker(ticker_input).history(period="7d")
        spot_price = hist["Close"].dropna().iloc[-1]
        spot_date = hist.index[-1].strftime("%Y-%m-%d")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div style="background-color:#f1f3f6; padding: 1.5rem; border-radius: 1rem; text-align: center;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;">
                    <h3 style="margin-bottom: 0.5rem;">Spot Price for <span style="color:#3366cc;">{ticker_input}</span></h3>
                    <p style="font-size: 2rem; margin: 0; color:#222;"><strong>${spot_price:.2f}</strong></p>
                    <p style="margin-top: 0.5rem; color:#666;">Last updated: {spot_date}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Loading error : {e}")
        st.stop()

    st.sidebar.header("Filter Options Data")
    avalaible_dates = sorted(df["Expiration"].unique(),key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    expiration_choice = st.sidebar.selectbox("Select Expiration Date", avalaible_dates)
    type_choice = st.sidebar.radio("Select Option Type", ["call", "put"])
    avalaible_strikes = sorted(df["Strike"].unique())
    strike_choice = st.sidebar.selectbox("Select Strike Price", avalaible_strikes)
    df_skew = df[(df["Expiration"] == expiration_choice)&(df["Type"] == type_choice)]
    tab1, tab2 = st.tabs(["📊 Skew & Term Structure", "🌐 Volatility Surface"])
    with tab1:
        st.subheader(f"Skew - {ticker_input} - {type_choice} @ {expiration_choice}")
        fig_skew = px.line(df_skew,x="Strike", y="IV",markers=True, title=f"Implied Volatility Skew in function of Strike Price",labels={"IV": "Implied Volatility", "Strike": "Strike Price"})
        st.plotly_chart(fig_skew, use_container_width=True)

        df_term = df[(df["Strike"] == strike_choice) & (df["Type"] == type_choice)]
        st.subheader(f"Term Structure - {ticker_input} - Strike {strike_choice}")
        fig_term = px.line(df_term, x="DaysToExpiry", y="IV", markers=True,
                        title="Term Structure of Implied Volatility in function of Days to Expiration",
                        labels={"IV": "Implied Volatility", "DaysToExpiry": "Days to Expiration"})
        fig_term.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_term, use_container_width=True)
        with st.expander("Full Options Data"):
            st.dataframe(df.head(200))
    with tab2:
        st.subheader(f"Implied volatility surface for {ticker_input}")

        try:
            surface_fig = plot_vol_surface(ticker_input)
            st.plotly_chart(surface_fig)
            
        except Exception as e:
            st.error(f"Error plotting surface: {e}")

else:
    st.info("Please enter a ticker symbol to fetch options data.")