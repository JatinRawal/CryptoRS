"""
Model Engine for Crypto Recommender System
Handles market data fetching from CoinGecko and AI-powered recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List


@st.cache_data(ttl=300)  # Cache for 5 minutes to prevent rate limiting
def fetch_market_data(vs_currency: str = "usd", per_page: int = 50) -> pd.DataFrame:
    """
    Fetch live cryptocurrency market data from CoinGecko API.
    
    Args:
        vs_currency: The target currency for prices
        per_page: Number of coins to fetch
    
    Returns:
        DataFrame with market data
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": True,
        "price_change_percentage": "24h,7d"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return _get_fallback_data()
        
        df = pd.DataFrame(data)
        
        # Calculate volatility from sparkline if available, otherwise estimate from price change
        if 'sparkline_in_7d' in df.columns:
            df['volatility'] = df['sparkline_in_7d'].apply(
                lambda x: np.std(x['price']) if x and isinstance(x, dict) and 'price' in x and len(x['price']) > 0 else 0
            )
        else:
            # Estimate volatility from 24h price change when sparkline not available
            df['volatility'] = abs(df['price_change_percentage_24h'].fillna(0)) / 100 * df['current_price'].fillna(1)
        
        # Normalize volatility
        if df['volatility'].max() > 0:
            df['volatility_normalized'] = df['volatility'] / df['volatility'].max()
        else:
            df['volatility_normalized'] = 0
        
        # Calculate risk ratio (higher = riskier)
        df['risk_ratio'] = (
            df['volatility_normalized'] * 0.5 +
            (1 - df['market_cap'].rank(pct=True)) * 0.3 +
            abs(df['price_change_percentage_24h'].fillna(0)) / 100 * 0.2
        )
        
        return df
        
    except (requests.RequestException, KeyError, ValueError) as e:
        st.warning(f"Could not fetch live data. Using cached/fallback data.")
        return _get_fallback_data()


def _get_fallback_data() -> pd.DataFrame:
    """Provide fallback data when API is unavailable."""
    fallback_coins = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "current_price": 43250, 
         "market_cap": 850000000000, "price_change_percentage_24h": 1.5, "volatility": 0.02, 
         "volatility_normalized": 0.2, "risk_ratio": 0.15, "image": "https://assets.coingecko.com/coins/images/1/small/bitcoin.png"},
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "current_price": 2280, 
         "market_cap": 275000000000, "price_change_percentage_24h": 2.1, "volatility": 0.03, 
         "volatility_normalized": 0.3, "risk_ratio": 0.25, "image": "https://assets.coingecko.com/coins/images/279/small/ethereum.png"},
        {"id": "tether", "symbol": "usdt", "name": "Tether", "current_price": 1.0, 
         "market_cap": 95000000000, "price_change_percentage_24h": 0.01, "volatility": 0.001, 
         "volatility_normalized": 0.01, "risk_ratio": 0.05, "image": "https://assets.coingecko.com/coins/images/325/small/Tether.png"},
        {"id": "binancecoin", "symbol": "bnb", "name": "BNB", "current_price": 315, 
         "market_cap": 47000000000, "price_change_percentage_24h": 1.8, "volatility": 0.035, 
         "volatility_normalized": 0.35, "risk_ratio": 0.28, "image": "https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png"},
        {"id": "solana", "symbol": "sol", "name": "Solana", "current_price": 98, 
         "market_cap": 42000000000, "price_change_percentage_24h": 4.5, "volatility": 0.06, 
         "volatility_normalized": 0.6, "risk_ratio": 0.45, "image": "https://assets.coingecko.com/coins/images/4128/small/solana.png"},
        {"id": "ripple", "symbol": "xrp", "name": "XRP", "current_price": 0.62, 
         "market_cap": 34000000000, "price_change_percentage_24h": 1.2, "volatility": 0.04, 
         "volatility_normalized": 0.4, "risk_ratio": 0.32, "image": "https://assets.coingecko.com/coins/images/44/small/xrp-symbol-white-128.png"},
        {"id": "cardano", "symbol": "ada", "name": "Cardano", "current_price": 0.52, 
         "market_cap": 18500000000, "price_change_percentage_24h": 2.8, "volatility": 0.045, 
         "volatility_normalized": 0.45, "risk_ratio": 0.38, "image": "https://assets.coingecko.com/coins/images/975/small/cardano.png"},
        {"id": "avalanche-2", "symbol": "avax", "name": "Avalanche", "current_price": 35, 
         "market_cap": 13000000000, "price_change_percentage_24h": 5.2, "volatility": 0.07, 
         "volatility_normalized": 0.7, "risk_ratio": 0.55, "image": "https://assets.coingecko.com/coins/images/12559/small/Avalanche_Circle_RedWhite_Trans.png"},
        {"id": "polkadot", "symbol": "dot", "name": "Polkadot", "current_price": 7.2, 
         "market_cap": 9500000000, "price_change_percentage_24h": 3.1, "volatility": 0.055, 
         "volatility_normalized": 0.55, "risk_ratio": 0.42, "image": "https://assets.coingecko.com/coins/images/12171/small/polkadot.png"},
        {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin", "current_price": 0.082, 
         "market_cap": 11700000000, "price_change_percentage_24h": 6.5, "volatility": 0.08, 
         "volatility_normalized": 0.8, "risk_ratio": 0.65, "image": "https://assets.coingecko.com/coins/images/5/small/dogecoin.png"},
    ]
    return pd.DataFrame(fallback_coins)


def get_ai_recommendation(
    risk_level: str,
    risk_score: int,
    capital: float
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Generate AI-powered cryptocurrency recommendations based on user's risk profile.
    
    Args:
        risk_level: "Conservative", "Moderate", or "Aggressive"
        risk_score: Numerical risk score (1-100)
        capital: Amount of capital to deploy in USD
    
    Returns:
        Tuple of (portfolio_df, chart_colors, explanation)
    """
    df = fetch_market_data()
    
    if df.empty:
        return pd.DataFrame(), [], "Unable to generate recommendations. Please try again later."
    
    # Define allocation strategies based on risk level
    if risk_level == "Conservative":
        portfolio, colors, explanation = _conservative_portfolio(df, capital, risk_score)
    elif risk_level == "Moderate":
        portfolio, colors, explanation = _moderate_portfolio(df, capital, risk_score)
    else:  # Aggressive
        portfolio, colors, explanation = _aggressive_portfolio(df, capital, risk_score)
    
    return portfolio, colors, explanation


def _conservative_portfolio(
    df: pd.DataFrame, 
    capital: float, 
    risk_score: int
) -> Tuple[pd.DataFrame, List[str], str]:
    """Generate conservative portfolio: Top market cap + lowest volatility."""
    
    # Filter for stable coins (stablecoins + top market cap with low volatility)
    df_sorted = df.nsmallest(10, 'risk_ratio')
    
    # Select top 4-5 coins
    selected = df_sorted.head(5)
    
    # Weights: Heavy on stablecoins and BTC
    weights = [0.35, 0.30, 0.20, 0.10, 0.05][:len(selected)]
    
    portfolio = _create_portfolio_df(selected, weights, capital)
    
    colors = ["#00FF94", "#00D4AA", "#00B4BE", "#0099CC", "#007ACC"]
    
    explanation = f"""
### 🛡️ Conservative Portfolio Strategy

**Risk Score:** {risk_score}/100 (Low Risk Tolerance)

**Why This Portfolio:**
- **Heavy allocation to Bitcoin and stablecoins** for capital preservation
- Selected coins with the **lowest volatility** in the market
- Focus on **established, high market-cap** cryptocurrencies
- Prioritized assets with **proven track records** and institutional backing

**Key Characteristics:**
- Lower potential returns, but significantly **reduced downside risk**
- Stablecoin allocation provides **liquidity buffer**
- Bitcoin serves as **digital gold** hedge against traditional markets

**Recommendation:** Hold for 6-12 months minimum. Consider dollar-cost averaging.
"""
    
    return portfolio, colors, explanation


def _moderate_portfolio(
    df: pd.DataFrame, 
    capital: float, 
    risk_score: int
) -> Tuple[pd.DataFrame, List[str], str]:
    """Generate moderate portfolio: Mix of stable and growth coins."""
    
    # Get a mix: some stable, some growth
    stable_coins = df.nsmallest(3, 'risk_ratio')
    growth_coins = df[~df['id'].isin(stable_coins['id'])].nlargest(3, 'price_change_percentage_24h')
    
    selected = pd.concat([stable_coins.head(3), growth_coins.head(3)]).head(6)
    
    # Balanced weights
    weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10][:len(selected)]
    
    portfolio = _create_portfolio_df(selected, weights, capital)
    
    colors = ["#00FF94", "#00D4AA", "#FFD700", "#FF8C00", "#FF6B6B", "#CC5599"]
    
    explanation = f"""
### ⚖️ Moderate Portfolio Strategy

**Risk Score:** {risk_score}/100 (Balanced Risk Tolerance)

**Why This Portfolio:**
- **Balanced mix** of stability and growth potential
- Core holdings in **established cryptocurrencies** (BTC, ETH)
- Allocation to **high-momentum altcoins** for growth
- Diversification across **different blockchain ecosystems**

**Key Characteristics:**
- **Moderate volatility** with growth potential
- 50% in stable assets, 50% in growth assets
- Exposure to **smart contract platforms** and **emerging sectors**

**Recommendation:** Quarterly rebalancing recommended. Monitor market conditions.
"""
    
    return portfolio, colors, explanation


def _aggressive_portfolio(
    df: pd.DataFrame, 
    capital: float, 
    risk_score: int
) -> Tuple[pd.DataFrame, List[str], str]:
    """Generate aggressive portfolio: High momentum coins."""
    
    # Focus on high momentum and growth potential
    # Select coins with highest 24h change (momentum indicators)
    momentum_coins = df.nlargest(8, 'price_change_percentage_24h')
    
    # Also consider some high volatility (potential gains)
    selected = momentum_coins.head(6)
    
    # Aggressive weights - more concentrated
    weights = [0.30, 0.25, 0.18, 0.12, 0.10, 0.05][:len(selected)]
    
    portfolio = _create_portfolio_df(selected, weights, capital)
    
    colors = ["#FF6B6B", "#FF8E53", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6"]
    
    explanation = f"""
### 🚀 Aggressive Portfolio Strategy

**Risk Score:** {risk_score}/100 (High Risk Tolerance)

**Why This Portfolio:**
- Focus on **high-momentum assets** with strong recent performance
- Selected coins showing **bullish price action** (24h gains)
- Concentrated positions for **maximum upside potential**
- Exposure to **emerging altcoins** and trending protocols

**Key Characteristics:**
- **High volatility** - expect significant price swings
- Potential for **substantial gains** (and losses)
- Requires **active monitoring** and quick decision-making

**⚠️ Risk Warning:** This portfolio is suitable only for investors who can afford to lose their entire investment.

**Recommendation:** Set stop-losses. Consider taking profits at 20-30% gains.
"""
    
    return portfolio, colors, explanation


def _create_portfolio_df(
    selected: pd.DataFrame, 
    weights: List[float], 
    capital: float
) -> pd.DataFrame:
    """Create the final portfolio DataFrame."""
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights[:len(selected)]]
    
    portfolio_data = []
    for i, (_, coin) in enumerate(selected.iterrows()):
        if i >= len(weights):
            break
        
        weight = weights[i]
        allocation = capital * weight
        
        portfolio_data.append({
            "Asset": coin.get('symbol', 'N/A').upper(),
            "Name": coin.get('name', 'Unknown'),
            "Price": f"${coin.get('current_price', 0):,.2f}",
            "24h Change": f"{coin.get('price_change_percentage_24h', 0):+.2f}%",
            "Weight": f"{weight * 100:.1f}%",
            "Allocated ($)": f"${allocation:,.2f}"
        })
    
    return pd.DataFrame(portfolio_data)


def get_portfolio_for_chart(portfolio_df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """Extract data for Plotly chart from portfolio DataFrame."""
    if portfolio_df.empty:
        return [], []
    
    labels = portfolio_df["Asset"].tolist()
    
    # Parse weights from string format
    values = []
    for w in portfolio_df["Weight"]:
        try:
            values.append(float(w.replace("%", "")))
        except:
            values.append(0)
    
    return labels, values
