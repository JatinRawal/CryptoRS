# Cloud-Based Crypto Recommender System

A comprehensive **Cryptocurrency Recommendation Engine** designed to solve the "Cold Start Problem" for new investors. This application uses a rule-based AI engine to analyze user demographics and financial goals, generating personalized investment portfolios with explainable insights.

##  Features

### 1. Secure Authentication System
- **User Registration & Login:** Secure account creation with password hashing (bcrypt).
- **Role-Based Access Control (RBAC):** Distinct dashboards for **Users** and **Admins**.
- **Session Management:** Persists user state across page reloads.

### 2. "Cold Start" Problem Solver
- **Interactive Onboarding:** New users complete a psychometric questionnaire (Age, Income, Experience, Goals).
- **Risk Scoring Algorithm:** Automatically calculates a numerical **Risk Score (0-100)** and categorizes users into:
  -  **Conservative**
  -  **Moderate**
  -  **Aggressive**

### 3. AI Recommendation Engine
- **Live Market Data:** Fetches real-time prices, market caps, and volatility from the **CoinGecko API**.
- **Dynamic Portfolio Generation:**
  - **Conservative:** Focuses on high-cap assets (BTC, ETH) and stablecoins.
  - **Moderate:** Balanced mix of stability and growth assets.
  - **Aggressive:** High-momentum altcoins and volatility plays.
- **Explainable AI (XAI):** Provides a "Why This Portfolio?" card explaining the logic behind the suggestions.

### 4. Interactive Dashboards
- **User Dashboard:** View risk profile, monthly capacity, and generate custom investment amounts.
- **Visual Analytics:** Interactive **Plotly Pie Charts** showing portfolio allocation.
- **Admin Panel:** View all registered users, monitor system-wide risk distribution, and manage user accounts.

---

## Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/) (with custom CSS for Glassmorphism UI)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly Express / Graph Objects
- **Authentication:** Bcrypt, JSON-based local database
- **API:** CoinGecko Public API


