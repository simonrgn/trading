import os
import yfinance as yf
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
from typing import Optional,cast,Tuple
import logging
import typer
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import plotly.io as pio
from scipy.interpolate import griddata
logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def black_scholes_call(S:float, K:float, T:float, r:float, sigma:float) -> float:
    """Calculate the Black-Scholes call option price."""
    d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return cast(float,S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)) 

def black_scholes_put(S:float, K:float, T:float, r:float, sigma:float) -> float:
    """Calculate the Black-Scholes put option price."""
    d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return cast(float,K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))


def implied_vol(S:float, K:float, T:float, r:float, market_price:float,*,option_type:str)-> Optional[float]:
    """Retourne la volatilité implicite par inversion de Black-Scholes"""
    try:
        if option_type.lower() == "call":
            f = lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price
        else:
            f = lambda sigma: black_scholes_put(S, K, T, r, sigma) - market_price
        
        return cast(float,brentq(f, 1e-6, 5.0)) 
    except Exception:
        return None

def compute_greeks(S:float, K:float, T:float, r:float, sigma:float,*,option_call:str) -> Tuple[float, float]:
    """Compute the Greeks for a given option."""
    d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    
    if option_call.lower() == "call":
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r*T) * norm.cdf(d2))
        vega = S * norm.pdf(d1) * sqrt(T)
    else:
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r*T) * norm.cdf(-d2))
        vega = S * norm.pdf(d1) * sqrt(T)

    return delta, gamma, theta, vega #type:ignore

def fetch_options_data(ticker:str, risk_free_rate:float)-> Tuple[pd.DataFrame, str]:
    """Fetch options data for a given ticker and save it to a CSV file."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="7d")  # pour être sûr d'avoir des données récentes
    if hist.empty:
        raise ValueError(f"No historical data found for {ticker}")
    spot = hist["Close"].dropna().iloc[-1]
    expirations = tk.options
    if not expirations:
        raise ValueError(f"No options data found for ticker: {ticker}")
    rows = []
    for exp in expirations: 
        opt_chain = tk.option_chain(exp)
        days_to_expiration = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        T = max(days_to_expiration / 365, 1e-6)
        for _, row in opt_chain.calls.iterrows():
            K = row["strike"]
            price = row["lastPrice"]
            iv = implied_vol(spot, K, T, risk_free_rate, price,option_type="call")
            rows.append([ticker, "call", exp, days_to_expiration,K, price, iv])

        for _, row in opt_chain.puts.iterrows():
            K = row["strike"]
            price = row["lastPrice"]
            iv = implied_vol(spot, K, T, risk_free_rate, price,option_type="put")
            rows.append([ticker, "put", exp,days_to_expiration, K, price, iv])

    df = pd.DataFrame(rows, columns=["Ticker", "Type", "Expiration","DaysToExpiry", "Strike", "MarketPrice", "IV"])
    
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"{ticker}_options.csv")
    df.to_csv(out, index=False)
    logger.info(f"Options data for {ticker} saved to {out}")
    return df, out

def plot_vol_surface(
    ticker: str,
    csv_path: Optional[str] = None,
    iv_cap: float = 5.0,
) -> Figure:
    """Plot the implied volatility surface for a given ticker using data from a CSV file."""

    if csv_path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}_options.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV unfoundable: {csv_path}. Launch fetch_options_data(ticker) first.")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["IV"])
    if "DaysToExpiry" not in df.columns:
        raise ValueError("CSV must contain 'DaysToExpiry' column.")

    df["T"] = df["DaysToExpiry"] / 365.0
    spot = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    df["Moneyness"] = df["Strike"] / float(spot)

    df = df[(df["IV"] > 0) & (df["IV"] < iv_cap)]
    if df.empty:
        raise ValueError("DataFrame is empty after filtering. Check your data or adjust the iv_cap parameter.")

    m_min, m_max = df["Moneyness"].quantile([0.05, 0.95])
    t_min, t_max = df["T"].min(), df["T"].max()
    m_grid = np.linspace(m_min, m_max, 60)
    t_grid = np.linspace(t_min, t_max, 60)
    M, T = np.meshgrid(m_grid, t_grid)

    points = df[["Moneyness", "T"]].to_numpy()
    values = df["IV"].to_numpy()
    IV_grid = griddata(points, values, (M, T), method="linear")
    fig = go.Figure(
        data=[
            go.Surface(
                x=M, y=T * 365.0, z=IV_grid,
                colorbar_title="IV", showscale=True,
                colorscale="Viridis",  # ou autre
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"Vol Surface — {ticker}",
            x=0.5,
            xanchor="center"
        ),
        autosize=False,
        width=1000,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(
                title="Moneyness (K/S)",
                tickmode='linear',
                tick0=0,
                dtick=0.1,
            ),
            yaxis=dict(
                title="Days to Expiry",
                autorange="reversed"
            ),
            zaxis=dict(
                title="Implied Volatility",
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.7)
            )
        ),
        template="plotly_white",
    )

    return fig



@app.command("fetch")
def cli_fetch(ticker: str = typer.Option(..., "--ticker","-t",help="Ticker symbol of the stock (e.g., AAPL)"),
         risk_free_rate: float = typer.Option(0.05, help="Risk-free interest rate (default is 5%)"))-> None:
    """Main function to fetch options data for a given ticker."""
    fetch_options_data(ticker, risk_free_rate)

@app.command("plot-surface")
def cli_plot_surface(
    ticker: str = typer.Option(..., "-t", "--ticker", help="Ticker (e.g., AAPL)"),
    csv_path: Optional[str] = typer.Option(None, "--csv", help="CSV Path (else fetches data)"),
    iv_cap: float = typer.Option(5.0, "--iv-cap", help="Filter abnormal values (>cap)"),
) -> None:
    """Shows volatility surface for a given ticker."""
    plot_vol_surface(ticker=ticker, csv_path=csv_path, iv_cap=iv_cap)

@app.command("fetch-and-plot")
def cli_fetch_and_plot(
    ticker: str = typer.Option(..., "-t", "--ticker", help="Ticker (e.g., AAPL)"),
    rate: float = typer.Option(0.05, "-r", "--rate", help="Risk-free interest rate (default is 5%)"),
    iv_cap: float = typer.Option(5.0, "--iv-cap", help="Filter abnormal values (>cap)"),
) -> None:
    """Fetch options data and plot the volatility surface for a given ticker."""
    _, csv_path = fetch_options_data(ticker, rate)
    plot_vol_surface(ticker=ticker, csv_path=csv_path, iv_cap=iv_cap)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    app()
