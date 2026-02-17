"""
Model Engine for Crypto Recommender System
Implements 3 recommendation models from the research notebook:
    1. KNN (K-Nearest Neighbors) — feature similarity baseline
    2. SVD (Truncated SVD)       — latent factor model
    3. GCN (Graph Convolutional Network) — trained with self-supervised
       adjacency-reconstruction loss

All models implemented in pure NumPy/scikit-learn (no PyTorch dependency).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, f1_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_market_data(vs_currency: str = "usd", per_page: int = 50) -> pd.DataFrame:
    """Fetch live cryptocurrency market data from CoinGecko API."""
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
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _get_fallback_data()
        return pd.DataFrame(data)
    except (requests.RequestException, ValueError):
        st.warning("Could not fetch live data — using fallback dataset.")
        return _get_fallback_data()


def _get_fallback_data() -> pd.DataFrame:
    """Hardcoded top-10 crypto for offline use."""
    coins = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin",
         "current_price": 43250, "market_cap": 850e9, "total_volume": 28e9,
         "price_change_percentage_24h": 1.5, "sparkline_in_7d": {"price": list(np.random.normal(43000, 500, 168))}},
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum",
         "current_price": 2280, "market_cap": 275e9, "total_volume": 15e9,
         "price_change_percentage_24h": 2.1, "sparkline_in_7d": {"price": list(np.random.normal(2250, 60, 168))}},
        {"id": "tether", "symbol": "usdt", "name": "Tether",
         "current_price": 1.0, "market_cap": 95e9, "total_volume": 55e9,
         "price_change_percentage_24h": 0.01, "sparkline_in_7d": {"price": list(np.random.normal(1.0, 0.002, 168))}},
        {"id": "binancecoin", "symbol": "bnb", "name": "BNB",
         "current_price": 315, "market_cap": 47e9, "total_volume": 1.2e9,
         "price_change_percentage_24h": 1.8, "sparkline_in_7d": {"price": list(np.random.normal(312, 8, 168))}},
        {"id": "solana", "symbol": "sol", "name": "Solana",
         "current_price": 98, "market_cap": 42e9, "total_volume": 2.5e9,
         "price_change_percentage_24h": 4.5, "sparkline_in_7d": {"price": list(np.random.normal(95, 7, 168))}},
        {"id": "ripple", "symbol": "xrp", "name": "XRP",
         "current_price": 0.62, "market_cap": 34e9, "total_volume": 1.1e9,
         "price_change_percentage_24h": 1.2, "sparkline_in_7d": {"price": list(np.random.normal(0.61, 0.02, 168))}},
        {"id": "cardano", "symbol": "ada", "name": "Cardano",
         "current_price": 0.52, "market_cap": 18.5e9, "total_volume": 0.5e9,
         "price_change_percentage_24h": 2.8, "sparkline_in_7d": {"price": list(np.random.normal(0.51, 0.02, 168))}},
        {"id": "avalanche-2", "symbol": "avax", "name": "Avalanche",
         "current_price": 35, "market_cap": 13e9, "total_volume": 0.6e9,
         "price_change_percentage_24h": 5.2, "sparkline_in_7d": {"price": list(np.random.normal(34, 3, 168))}},
        {"id": "polkadot", "symbol": "dot", "name": "Polkadot",
         "current_price": 7.2, "market_cap": 9.5e9, "total_volume": 0.3e9,
         "price_change_percentage_24h": 3.1, "sparkline_in_7d": {"price": list(np.random.normal(7.0, 0.5, 168))}},
        {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin",
         "current_price": 0.082, "market_cap": 11.7e9, "total_volume": 0.8e9,
         "price_change_percentage_24h": 6.5, "sparkline_in_7d": {"price": list(np.random.normal(0.08, 0.008, 168))}},
    ]
    return pd.DataFrame(coins)


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (matching notebook)
# ══════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Extract and normalise features exactly as the notebook does:
        volatility_7d, momentum, risk_ratio, log_market_cap
    """
    # Volatility from 7-day sparkline (handle missing column)
    df = df.copy()
    if "sparkline_in_7d" in df.columns:
        df["volatility_7d"] = df["sparkline_in_7d"].apply(
            lambda x: np.std(x["price"]) if isinstance(x, dict) and "price" in x and len(x["price"]) > 0 else 0.0
        )
    else:
        # Fallback: use absolute 24h change * price as volatility proxy
        df["volatility_7d"] = (
            abs(df["price_change_percentage_24h"].fillna(0)) / 100
            * df["current_price"].fillna(1)
        )

    # Momentum (weighted 24h + 7d change)
    col_7d = "price_change_percentage_7d_in_currency"
    if col_7d not in df.columns:
        col_7d_fallback = df["price_change_percentage_24h"].fillna(0) * 1.5
        df["momentum"] = 0.4 * df["price_change_percentage_24h"].fillna(0) + 0.6 * col_7d_fallback
    else:
        df["momentum"] = (
            0.4 * df["price_change_percentage_24h"].fillna(0)
            + 0.6 * df[col_7d].fillna(0)
        )

    # Risk ratio
    df["risk_ratio"] = df.apply(
        lambda r: r["volatility_7d"] / r["current_price"] if r["current_price"] > 0 else 0,
        axis=1,
    )

    # Log market cap
    df["log_market_cap"] = np.log1p(df["market_cap"].fillna(0))

    features = ["volatility_7d", "momentum", "risk_ratio", "log_market_cap"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].fillna(0))

    # Risk categories for evaluation
    try:
        df["risk_cat"] = pd.qcut(df["risk_ratio"], 3, labels=["Low", "Medium", "High"])
    except ValueError:
        df["risk_cat"] = pd.cut(df["risk_ratio"], 3, labels=["Low", "Medium", "High"])

    return df, X_scaled, features


# ══════════════════════════════════════════════════════════════
# 3. GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def _build_graph(X: np.ndarray, threshold: float = 0.5):
    """
    Build adjacency matrix from feature cosine similarity,
    then normalise as D^{-0.5} A D^{-0.5}.
    Returns (normalised_adj, raw_adj, nx_graph).
    """
    sim = cosine_similarity(X)
    raw_adj = (sim > threshold).astype(float)
    np.fill_diagonal(raw_adj, 1.0)  # self-loops

    # Normalise
    D = raw_adj.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-12))
    norm_adj = D_inv_sqrt @ raw_adj @ D_inv_sqrt

    # Build NetworkX graph for metrics / visualisation
    G = nx.Graph()
    n = X.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if raw_adj[i, j] > 0 and i != j:
                G.add_edge(i, j, weight=float(sim[i, j]))

    return norm_adj, raw_adj, G


# ══════════════════════════════════════════════════════════════
# 4. GCN MODEL (pure NumPy — matches notebook's SimpleGCN)
# ══════════════════════════════════════════════════════════════

class GCNLayer:
    """Single GCN layer: H' = ReLU(A_hat @ H @ W)"""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, (in_dim, out_dim))

    def forward(self, A: np.ndarray, H: np.ndarray, apply_relu: bool = True) -> np.ndarray:
        Z = A @ H @ self.W
        if apply_relu:
            Z = np.maximum(Z, 0)
        return Z


class CryptoGCN:
    """
    2-layer GCN matching notebook's SimpleGCN.
    Trained via self-supervised adjacency reconstruction loss.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.layer1 = GCNLayer(in_dim, hidden_dim, rng)
        self.layer2 = GCNLayer(hidden_dim, out_dim, rng)
        self.lr = 0.01
        self.losses: list = []

    def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._H1_input = X
        self._H1 = self.layer1.forward(A, X, apply_relu=True)
        self._H2 = self.layer2.forward(A, self._H1, apply_relu=False)
        return self._H2

    def train(self, A_norm: np.ndarray, A_raw: np.ndarray, X: np.ndarray, epochs: int = 200):
        """
        Self-supervised training: learn to reconstruct the adjacency matrix
        from embeddings via  A_rec = sigmoid(Z @ Z^T).
        Loss = MSE(A_rec, A_raw).
        """
        self.losses = []
        for _ in range(epochs):
            # Forward
            Z = self.forward(A_norm, X)

            # Reconstruct adjacency
            logits = Z @ Z.T
            A_rec = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

            # MSE loss
            diff = A_rec - A_raw
            loss = np.mean(diff ** 2)
            self.losses.append(loss)

            # ── Backprop ──
            n = X.shape[0]
            dL_dArec = 2.0 * diff / (n * n)
            dArec_dlogits = A_rec * (1.0 - A_rec)
            dL_dlogits = dL_dArec * dArec_dlogits

            # dL/dZ  (from Z @ Z^T)
            dL_dZ = (dL_dlogits + dL_dlogits.T) @ Z

            # Layer 2 (no ReLU)
            dL_dH1 = A_norm @ dL_dZ @ self.layer2.W.T
            dL_dW2 = (A_norm @ self._H1).T @ dL_dZ

            # ReLU mask for layer 1
            relu_mask = (self._H1 > 0).astype(float)
            dL_dH1 *= relu_mask

            # Layer 1
            dL_dW1 = (A_norm @ X).T @ dL_dH1

            # Update weights (gradient descent)
            self.layer2.W -= self.lr * dL_dW2
            self.layer1.W -= self.lr * dL_dW1

        return self.losses


# ══════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════

def _get_risk_consistency(embeddings: np.ndarray, risk_labels) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour risk consistency (from notebook)."""
    nbrs = NearestNeighbors(n_neighbors=2, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    y_pred = np.array([risk_labels.iloc[i] for i in indices[:, 1]])
    y_true = risk_labels.values
    return y_true, y_pred


@st.cache_data(ttl=300)
def run_training_pipeline() -> Dict:
    """
    Full training pipeline returning all model artefacts.
    Cached for 5 minutes to avoid re-training on every rerun.
    """
    df_raw = fetch_market_data()
    if df_raw.empty:
        return {"error": True}

    df, X_scaled, feature_names = _engineer_features(df_raw)
    n = len(df)

    # ------ KNN ------
    k = min(8, n - 1)
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(X_scaled)

    # ------ SVD ------
    n_comps = min(2, X_scaled.shape[1])
    svd = TruncatedSVD(n_components=n_comps)
    X_svd = svd.fit_transform(X_scaled)
    X_svd_rec = svd.inverse_transform(X_svd)

    # ------ GCN (trained) ------
    A_norm, A_raw, G = _build_graph(X_scaled, threshold=0.5)
    gcn = CryptoGCN(in_dim=X_scaled.shape[1], hidden_dim=16, out_dim=8, seed=42)
    loss_history = gcn.train(A_norm, A_raw, X_scaled, epochs=200)
    X_gnn = gcn.forward(A_norm, X_scaled)

    # ------ Evaluation Metrics ------
    metrics = {}

    # KNN
    y_true_k, y_pred_k = _get_risk_consistency(X_scaled, df["risk_cat"])
    f1_knn = f1_score(y_true_k, y_pred_k, average="macro")
    sil_knn = silhouette_score(X_scaled, df["risk_cat"])
    metrics["KNN"] = {"MSE": "N/A", "Silhouette": round(sil_knn, 4), "F1": round(f1_knn, 4), "Type": "Baseline"}

    # SVD
    mse_svd = mean_squared_error(X_scaled, X_svd_rec)
    y_true_s, y_pred_s = _get_risk_consistency(X_svd, df["risk_cat"])
    f1_svd = f1_score(y_true_s, y_pred_s, average="macro")
    sil_svd = silhouette_score(X_svd, df["risk_cat"])
    metrics["SVD"] = {"MSE": round(mse_svd, 4), "Silhouette": round(sil_svd, 4), "F1": round(f1_svd, 4), "Type": "Latent"}

    # GCN
    y_true_g, y_pred_g = _get_risk_consistency(X_gnn, df["risk_cat"])
    f1_gnn = f1_score(y_true_g, y_pred_g, average="macro")
    sil_gnn = silhouette_score(X_gnn, df["risk_cat"])
    metrics["GCN"] = {"MSE": round(loss_history[-1], 4), "Silhouette": round(sil_gnn, 4), "F1": round(f1_gnn, 4), "Type": "Proposed"}

    return {
        "error": False,
        "df": df,
        "X_scaled": X_scaled,
        "X_svd": X_svd,
        "X_gnn": X_gnn,
        "knn": knn,
        "svd": svd,
        "gcn": gcn,
        "graph": G,
        "A_norm": A_norm,
        "metrics": metrics,
        "loss_history": loss_history,
        "feature_names": feature_names,
    }


# ══════════════════════════════════════════════════════════════
# 6. RECOMMENDATION GENERATION
# ══════════════════════════════════════════════════════════════

def _score_for_risk(df: pd.DataFrame, embeddings: np.ndarray, risk_level: str) -> np.ndarray:
    """
    Score each coin in the embedding space based on user risk level.
    Uses a risk-profile preference vector projected into the embedding space.
    """
    n, d = embeddings.shape

    # Build a risk-profile target from the data itself
    risk_map = {"Conservative": "Low", "Moderate": "Medium", "Aggressive": "High"}
    target_cat = risk_map.get(risk_level, "Medium")
    mask = df["risk_cat"] == target_cat

    if mask.sum() > 0:
        # Centroid of matching risk category in embedding space
        centroid = embeddings[mask.values].mean(axis=0).reshape(1, -1)
    else:
        centroid = embeddings.mean(axis=0).reshape(1, -1)

    # Cosine similarity to centroid
    sims = cosine_similarity(embeddings, centroid).flatten()

    # For Aggressive: also boost high-momentum coins
    if risk_level == "Aggressive":
        momentum = df["momentum"].fillna(0).values
        momentum_norm = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-12)
        sims = 0.6 * sims + 0.4 * momentum_norm
    # For Conservative: boost low-volatility coins
    elif risk_level == "Conservative":
        vol = df["volatility_7d"].fillna(0).values
        inv_vol = 1.0 - (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
        sims = 0.6 * sims + 0.4 * inv_vol

    return sims


def _compute_shap_values(
    pipeline: Dict, risk_level: str, top_indices: np.ndarray, feature_names: list
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute SHAP values using a surrogate RandomForest that approximates the
    ensemble scoring.  TreeExplainer is fast and avoids the GCN dimension
    mismatch that KernelExplainer would trigger.
    """
    from sklearn.ensemble import RandomForestRegressor

    X_scaled = pipeline["X_scaled"]
    df = pipeline["df"].copy()

    # Compute ensemble scores for ALL coins (these are the 'targets' for the surrogate)
    scores_knn = _score_for_risk(df, X_scaled, risk_level)
    scores_svd = _score_for_risk(df, pipeline["X_svd"], risk_level)
    scores_gnn = _score_for_risk(df, pipeline["X_gnn"], risk_level)
    ensemble_scores = 0.50 * scores_gnn + 0.25 * scores_knn + 0.25 * scores_svd

    # Train a surrogate model:  raw features -> ensemble score
    surrogate = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    surrogate.fit(X_scaled, ensemble_scores)

    # SHAP TreeExplainer (very fast on RandomForest)
    explainer = shap.TreeExplainer(surrogate)
    top_features = X_scaled[top_indices]
    shap_values = explainer.shap_values(top_features)

    return np.asarray(shap_values), top_features, feature_names


def get_ai_recommendation(
    risk_level: str,
    risk_score: int,
    capital: float,
) -> Tuple[pd.DataFrame, List[str], str, Dict]:
    """
    Generate recommendations using the ensemble of all 3 models.
    Returns (portfolio_df, colors, explanation, shap_data).
    """
    pipeline = run_training_pipeline()
    if pipeline.get("error"):
        return pd.DataFrame(), [], "Unable to generate recommendations.", {}

    df = pipeline["df"]
    X_scaled = pipeline["X_scaled"]
    X_svd = pipeline["X_svd"]
    X_gnn = pipeline["X_gnn"]

    # Score with each model
    scores_knn = _score_for_risk(df, X_scaled, risk_level)
    scores_svd = _score_for_risk(df, X_svd, risk_level)
    scores_gnn = _score_for_risk(df, X_gnn, risk_level)

    # Ensemble weights: GCN 50%, KNN 25%, SVD 25%
    ensemble = 0.50 * scores_gnn + 0.25 * scores_knn + 0.25 * scores_svd
    df["ensemble_score"] = ensemble

    # Select top coins
    num_coins = {"Conservative": 5, "Moderate": 6, "Aggressive": 6}.get(risk_level, 5)
    top = df.nlargest(num_coins, "ensemble_score")
    top_indices = top.index.values
    top = top.reset_index(drop=True)

    # Softmax weights
    s = top["ensemble_score"].values.astype(float)
    exp_s = np.exp(s - s.max())
    weights = (exp_s / exp_s.sum()).tolist()

    portfolio = _create_portfolio_df(top, weights, capital)
    colors = _get_colors(risk_level, len(top))
    explanation = _build_explanation(risk_level, risk_score, pipeline, top)

    # SHAP explainability
    shap_data = {}
    try:
        sv, feat_data, feat_names = _compute_shap_values(
            pipeline, risk_level, top_indices, pipeline["feature_names"]
        )
        shap_data = {
            "shap_values": sv,
            "feature_data": feat_data,
            "feature_names": feat_names,
            "coin_names": top["name"].tolist() if "name" in top.columns else [],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()

    return portfolio, colors, explanation, shap_data


# ══════════════════════════════════════════════════════════════
# 7. HELPERS
# ══════════════════════════════════════════════════════════════

def _create_portfolio_df(selected: pd.DataFrame, weights: List[float], capital: float) -> pd.DataFrame:
    rows = []
    for i, (_, coin) in enumerate(selected.iterrows()):
        if i >= len(weights):
            break
        w = weights[i]
        rows.append({
            "Asset": coin.get("symbol", "?").upper(),
            "Name": coin.get("name", "Unknown"),
            "Price": f"${coin.get('current_price', 0):,.2f}",
            "24h Change": f"{coin.get('price_change_percentage_24h', 0):+.2f}%",
            "Weight": f"{w * 100:.1f}%",
            "Allocated ($)": f"${capital * w:,.2f}",
        })
    return pd.DataFrame(rows)


def _get_colors(risk_level: str, n: int) -> List[str]:
    palettes = {
        "Conservative": ["#00FF94", "#00D4AA", "#00B4BE", "#0099CC", "#007ACC", "#005FAA"],
        "Moderate":     ["#00FF94", "#00D4AA", "#FFD700", "#FF8C00", "#FF6B6B", "#CC5599"],
        "Aggressive":   ["#FF6B6B", "#FF8E53", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6"],
    }
    return palettes.get(risk_level, palettes["Moderate"])[:n]


def _build_explanation(risk_level: str, risk_score: int, pipeline: Dict, top: pd.DataFrame) -> str:
    """User-facing explainable AI -- shown to all users."""
    top_coins = ", ".join(top["name"].tolist()) if "name" in top.columns else "N/A"
    n = len(pipeline["df"])

    strategy = {
        "Conservative": ("Conservative", "Low-volatility, high market-cap assets",
                         "Capital preservation with steady growth",
                         "Hold 6-12 months minimum. Dollar-cost average."),
        "Moderate": ("Moderate", "Balanced mix of stability and growth",
                     "Moderate risk with growth exposure",
                     "Quarterly rebalancing recommended."),
        "Aggressive": ("Aggressive", "High-momentum, high-volatility assets",
                       "Maximum upside via concentrated positions",
                       "Set stop-losses. Take profits at 20-30% gains."),
    }
    label, focus, goal, advice = strategy.get(risk_level, strategy["Moderate"])

    return f"""
### {label} Portfolio Strategy

**Risk Score:** {risk_score}/100

**How it works:** Our AI analyses {n} cryptocurrencies using a Graph Neural
Network that learns relationships between assets based on their market
behaviour -- price, volatility, momentum, and trading volume. Assets are
scored against your risk profile and the top picks are selected for your
portfolio.

**Focus:** {focus}
**Goal:** {goal}
**Selected:** {top_coins}
**Advice:** {advice}
"""


def _build_technical_explanation(pipeline: Dict) -> str:
    """Technical metrics and architecture -- shown to master admin only."""
    G = pipeline["graph"]
    metrics = pipeline["metrics"]
    n = len(pipeline["df"])
    num_edges = G.number_of_edges()
    avg_deg = sum(dict(G.degree()).values()) / n if n else 0
    final_loss = pipeline["loss_history"][-1] if pipeline["loss_history"] else "N/A"

    m_knn = metrics["KNN"]
    m_svd = metrics["SVD"]
    m_gcn = metrics["GCN"]

    return f"""
### Multi-Model Ensemble Pipeline

This portfolio was generated using an ensemble of **3 trained models**:

| Model | Silhouette | F1-Score | Info |
|-------|-----------|----------|------|
| KNN (Baseline) | {m_knn['Silhouette']} | {m_knn['F1']} | k=8, cosine distance |
| SVD (Latent) | {m_svd['Silhouette']} | {m_svd['F1']} | MSE: {m_svd['MSE']} |
| **GCN (Proposed)** | **{m_gcn['Silhouette']}** | **{m_gcn['F1']}** | Loss: {final_loss:.4f} |

**GCN Architecture:**
- Graph: {n} nodes, {num_edges} edges (avg degree {avg_deg:.1f})
- 2-layer GCN (4 -> 16 -> 8) trained for 200 epochs
- Self-supervised loss: adjacency matrix reconstruction

**Ensemble:** GCN 50% + KNN 25% + SVD 25%
"""


def get_portfolio_for_chart(portfolio_df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    if portfolio_df.empty:
        return [], []
    labels = portfolio_df["Asset"].tolist()
    values = []
    for w in portfolio_df["Weight"]:
        try:
            values.append(float(w.replace("%", "")))
        except (ValueError, AttributeError):
            values.append(0)
    return labels, values


def get_model_metrics() -> Dict:
    """
    Return model evaluation metrics for display in the dashboard.
    Triggers the training pipeline if not already cached.
    """
    pipeline = run_training_pipeline()
    if pipeline.get("error"):
        return {}
    return pipeline["metrics"]


def get_loss_history() -> List[float]:
    """Return GCN training loss history for plotting."""
    pipeline = run_training_pipeline()
    if pipeline.get("error"):
        return []
    return pipeline["loss_history"]


def get_technical_explanation() -> str:
    """Return the technical multi-model explanation for master admin."""
    pipeline = run_training_pipeline()
    if pipeline.get("error"):
        return ""
    return _build_technical_explanation(pipeline)
