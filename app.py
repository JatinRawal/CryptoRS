"""
Cloud-Based Crypto Recommender System
Main Streamlit Application with Authentication, Onboarding, and AI Recommendations
"""

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from auth_utils import (
    register_user, login_user, save_user_profile, 
    get_user_profile, get_all_users, calculate_risk_score, delete_user,
    seed_master_admin, is_master_admin, get_pending_admins, approve_admin, reject_admin
)
from model_engine import get_ai_recommendation, get_portfolio_for_chart, get_model_metrics, get_loss_history, get_technical_explanation

# Seed the master admin account on app startup
seed_master_admin()

# Page Configuration
st.set_page_config(
    page_title="CrypTop - Cryptocurrency Recommender System",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism and Neon Effects
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(38, 39, 48, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .glass-card-glow {
        background: rgba(38, 39, 48, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(0, 255, 148, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 25px rgba(0, 255, 148, 0.1);
    }
    
    /* Neon Button Style */
    .stButton > button {
        background: linear-gradient(135deg, #00FF94 0%, #00D4AA 100%);
        color: #0E1117;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 148, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 255, 148, 0.5);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #00FF94 0%, #00D4AA 50%, #00B4BE 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #8B8B8B;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Risk Badge Styling */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .risk-conservative {
        background: linear-gradient(135deg, #00FF94, #00D4AA);
        color: #0E1117;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #0E1117;
    }
    
    .risk-aggressive {
        background: linear-gradient(135deg, #FF6B6B, #FF4757);
        color: #FFFFFF;
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(38, 39, 48, 0.35);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00FF94;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8B8B8B;
        margin-top: 0.3rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(38, 39, 48, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #FAFAFA;
    }
    
    .stSelectbox > div > div {
        background: rgba(38, 39, 48, 0.8);
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(14, 17, 23, 0.95);
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_role" not in st.session_state:
    st.session_state.user_role = "user"
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "is_master" not in st.session_state:
    st.session_state.is_master = False


def show_login_page():
    """Display the login/signup page."""
    st.markdown('<h1 class="main-header">CryptoAI Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Cryptocurrency Portfolio Recommendations</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # st.markdown('<div class="glass-card animated">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            login_username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            if st.button("Login", key="login_btn", use_container_width=True):
                success, message, user_data = login_user(login_username, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = user_data["username"]
                    st.session_state.user_role = user_data["role"]
                    st.session_state.user_profile = user_data.get("profile")
                    st.session_state.is_master = user_data.get("is_master", False)
                    if user_data["role"] == "pending_admin":
                        st.warning(message)
                    else:
                        st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        with tab2:
            st.markdown("### Create Account")
            reg_username = st.text_input("Choose Username", key="reg_user", placeholder="Enter username")
            reg_password = st.text_input("Choose Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Confirm password")
            
            # Admin registration option
            is_admin = st.checkbox("Request Admin Access", key="is_admin",
                                   help="Admin requests require approval from the master admin")
            
            if st.button("Create Account", key="reg_btn", use_container_width=True):
                if reg_password != reg_password2:
                    st.error("Passwords do not match!")
                else:
                    role = "admin" if is_admin else "user"
                    success, message = register_user(reg_username, reg_password, role)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_onboarding():
    """Display the onboarding form to solve cold start problem."""
    st.markdown('<h1 class="main-header">Complete Your Profile</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Help us understand your investment preferences</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-card-glow animated">', unsafe_allow_html=True)
        
        st.markdown("### Personal Information")
        
        age = st.slider("What is your age?", 18, 80, 30, help="Your age helps us determine appropriate risk levels")
        
        income = st.selectbox(
            "Annual Income Range",
            [
                "< $30,000",
                "$30,000 - $50,000",
                "$50,000 - $75,000",
                "$75,000 - $100,000",
                "$100,000 - $150,000",
                "> $150,000"
            ],
            help="Higher income may allow for more risk tolerance"
        )
        
        monthly_capacity = st.number_input(
            "Monthly Investment Capacity ($)",
            min_value=50,
            max_value=100000,
            value=500,
            step=50,
            help="How much can you invest each month?"
        )
        
        st.markdown("### Investment Experience")
        
        experience = st.selectbox(
            "Crypto Investment Experience",
            [
                "None - Complete Beginner",
                "Beginner - Less than 1 year",
                "Intermediate - 1-3 years",
                "Advanced - 3+ years"
            ]
        )
        
        goal = st.selectbox(
            "Investment Goal",
            [
                "Capital Preservation",
                "Steady Income",
                "Balanced Growth",
                "Aggressive Growth",
                "Maximum Returns (High Risk)"
            ]
        )
        
        st.markdown("---")
        
        if st.button("Calculate My Risk Profile", use_container_width=True):
            risk_score, risk_level = calculate_risk_score(age, income, experience, goal)
            
            # Save profile
            profile_data = {
                "age": age,
                "income": income,
                "monthly_capacity": monthly_capacity,
                "experience": experience,
                "goal": goal,
                "risk_score": risk_score,
                "risk_level": risk_level
            }
            
            success, message = save_user_profile(st.session_state.username, profile_data)
            
            if success:
                st.session_state.user_profile = profile_data
                
                # Show result
                st.markdown("---")
                st.markdown("### Your Risk Profile")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{risk_score}</div>
                        <div class="metric-label">Risk Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    badge_class = f"risk-{risk_level.lower()}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value"><span class="risk-badge {badge_class}">{risk_level}</span></div>
                        <div class="metric-label">Risk Level</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("Profile saved! Click 'Dashboard' in the sidebar to get personalized recommendations.")
                st.balloons()
            else:
                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_user_dashboard():
    """Display the main user dashboard with recommendations."""
    profile = st.session_state.user_profile
    
    st.markdown('<h1 class="main-header">Your Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Welcome back, {st.session_state.username}!</p>', unsafe_allow_html=True)
    
    # Profile Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card animated">
            <div class="metric-value">{profile.get('risk_score', 'N/A')}</div>
            <div class="metric-label">Risk Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level = profile.get('risk_level', 'N/A')
        badge_class = f"risk-{risk_level.lower()}" if risk_level != 'N/A' else ""
        st.markdown(f"""
        <div class="metric-card animated">
            <div class="metric-value"><span class="risk-badge {badge_class}">{risk_level}</span></div>
            <div class="metric-label">Risk Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card animated">
            <div class="metric-value">${profile.get('monthly_capacity', 0):,}</div>
            <div class="metric-label">Monthly Capacity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card animated">
            <div class="metric-value">{profile.get('experience', 'N/A').split(' - ')[0]}</div>
            <div class="metric-label">Experience</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendation Section
    st.markdown("### Generate Portfolio Recommendation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        capital = st.number_input(
            "Capital to Deploy ($)",
            min_value=100,
            max_value=1000000,
            value=min(profile.get('monthly_capacity', 500), 10000),
            step=100,
            help="Enter the amount you want to invest"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Recommendation", use_container_width=True)
    
    if generate_btn:
        with st.spinner("Analyzing market data and generating recommendations..."):
            portfolio_df, colors, explanation, shap_data = get_ai_recommendation(
                risk_level=profile.get('risk_level', 'Moderate'),
                risk_score=profile.get('risk_score', 50),
                capital=capital
            )
        
        if not portfolio_df.empty:
            st.markdown("---")
            
            # Two column layout for chart and table
            col_chart, col_table = st.columns([1, 1])
            
            with col_chart:
                st.markdown("### Portfolio Allocation")
                
                labels, values = get_portfolio_for_chart(portfolio_df)
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    marker=dict(colors=colors[:len(labels)]),
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white'),
                    hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>"
                )])
                
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(color='white')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=30, b=80, l=30, r=30),
                    height=400,
                    annotations=[
                        dict(
                            text=f'${capital:,.0f}',
                            x=0.5, y=0.5,
                            font=dict(size=24, color='#00FF94', family='Inter'),
                            showarrow=False
                        )
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_table:
                st.markdown("### Allocation Details")
                
                # Style the dataframe
                st.dataframe(
                    portfolio_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Asset": st.column_config.TextColumn("Asset", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "24h Change": st.column_config.TextColumn("24h", width="small"),
                        "Weight": st.column_config.TextColumn("Weight", width="small"),
                        "Allocated ($)": st.column_config.TextColumn("Amount", width="small"),
                    }
                )
            
            # Explainable AI Card
            st.markdown("---")
            st.markdown("### Why This Portfolio?")
            st.markdown(explanation)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Technical Details + SHAP (Master Admin Only)
            if st.session_state.get("is_master", False):
                # SHAP Feature Importance Chart
                if shap_data and "shap_values" in shap_data:
                    st.markdown("---")
                    st.markdown("### Feature Impact Analysis (SHAP)")
                    st.caption("Shows how each feature contributed to selecting these assets")
                    
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    sv = shap_data["shap_values"]
                    feat_names = shap_data["feature_names"]
                    coin_names = shap_data.get("coin_names", [])
                    
                    # Mean absolute SHAP values per feature
                    mean_shap = np.abs(sv).mean(axis=0)
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    
                    bars = ax.barh(
                        feat_names, mean_shap,
                        color=['#00FF94', '#FFD700', '#FF6B6B', '#4D96FF'][:len(feat_names)],
                        edgecolor='none', height=0.6
                    )
                    ax.set_xlabel('Mean |SHAP Value|', color='white', fontsize=11)
                    ax.set_title('Feature Importance (Global)', color='white', fontsize=13)
                    ax.tick_params(colors='white', labelsize=10)
                    for spine in ax.spines.values():
                        spine.set_color('#333')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Per-coin SHAP breakdown
                    if len(coin_names) > 0 and sv.shape[0] == len(coin_names):
                        st.markdown("#### Per-Asset Feature Breakdown")
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        fig2.patch.set_facecolor('#0e1117')
                        ax2.set_facecolor('#0e1117')
                        
                        x = np.arange(len(feat_names))
                        width = 0.8 / len(coin_names)
                        palette = ['#00FF94', '#FFD700', '#FF6B6B', '#4D96FF', '#9B59B6', '#FF8E53']
                        
                        for i, name in enumerate(coin_names):
                            ax2.bar(
                                x + i * width, sv[i],
                                width=width, label=name,
                                color=palette[i % len(palette)], alpha=0.85
                            )
                        
                        ax2.set_xticks(x + width * len(coin_names) / 2)
                        ax2.set_xticklabels(feat_names, color='white', fontsize=10)
                        ax2.set_ylabel('SHAP Value', color='white', fontsize=11)
                        ax2.set_title('Feature Contribution per Asset', color='white', fontsize=13)
                        ax2.tick_params(colors='white', labelsize=9)
                        ax2.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
                        ax2.axhline(y=0, color='#555', linewidth=0.5)
                        for spine in ax2.spines.values():
                            spine.set_color('#333')
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)

                tech_explanation = get_technical_explanation()
                if tech_explanation:
                    st.markdown("---")
                    st.markdown('<div class="glass-card-glow animated">', unsafe_allow_html=True)
                    st.markdown(tech_explanation)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Model Performance")
                
                metrics = get_model_metrics()
                loss_history = get_loss_history()
                
                if metrics:
                    col_metrics, col_loss = st.columns([1, 1])
                    
                    with col_metrics:
                        st.markdown("#### Comparative Metrics")
                        metrics_rows = []
                        for model_name, m in metrics.items():
                            metrics_rows.append({
                                "Model": model_name,
                                "MSE": str(m.get('MSE', 'N/A')),
                                "Silhouette": m.get('Silhouette', 0),
                                "F1-Score": m.get('F1', 0),
                            })
                        st.dataframe(
                            pd.DataFrame(metrics_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    
                    with col_loss:
                        st.markdown("#### GCN Training Loss")
                        if loss_history:
                            fig_loss = go.Figure()
                            fig_loss.add_trace(go.Scatter(
                                y=loss_history, mode='lines',
                                line=dict(color='#00FF94', width=2),
                                name='Reconstruction Loss'
                            ))
                            fig_loss.update_layout(
                                xaxis_title="Epoch",
                                yaxis_title="Loss (MSE)",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=300,
                                margin=dict(l=40, r=20, t=20, b=40),
                            )
                            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.error("Unable to generate recommendations. Please try again later.")


def show_admin_dashboard():
    """Display admin dashboard with user management."""
    st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">User Management & Analytics</p>', unsafe_allow_html=True)
    
    # Get all users
    users = get_all_users()
    
    if users:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card animated">
                <div class="metric-value">{len(users)}</div>
                <div class="metric-label">Total Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conservative = sum(1 for u in users if u.get('Risk Level') == 'Conservative')
            st.markdown(f"""
            <div class="metric-card animated">
                <div class="metric-value">{conservative}</div>
                <div class="metric-label">Conservative</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            moderate = sum(1 for u in users if u.get('Risk Level') == 'Moderate')
            st.markdown(f"""
            <div class="metric-card animated">
                <div class="metric-value">{moderate}</div>
                <div class="metric-label">Moderate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            aggressive = sum(1 for u in users if u.get('Risk Level') == 'Aggressive')
            st.markdown(f"""
            <div class="metric-card animated">
                <div class="metric-value">{aggressive}</div>
                <div class="metric-label">Aggressive</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User table
        st.markdown("### Registered Users")
        
        df = pd.DataFrame(users)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Username": st.column_config.TextColumn("Username", width="medium"),
                "Role": st.column_config.TextColumn("Role", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="medium"),
                "Monthly Capacity": st.column_config.TextColumn("Capacity", width="small"),
                "Created": st.column_config.TextColumn("Joined", width="small"),
            }
        )
        
        # Delete User Section
        st.markdown("---")
        st.markdown("### Delete User")
        
        # Get list of deletable users (exclude current admin)
        deletable_users = [u["Username"] for u in users if u["Username"].lower() != st.session_state.username.lower()]
        
        if deletable_users:
            col_select, col_btn = st.columns([3, 1])
            
            with col_select:
                user_to_delete = st.selectbox(
                    "Select user to delete",
                    options=deletable_users,
                    key="delete_user_select"
                )
            
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Delete", key="delete_btn", type="secondary"):
                    st.session_state.confirm_delete = user_to_delete
            
            # Confirmation dialog
            if "confirm_delete" in st.session_state and st.session_state.confirm_delete:
                st.warning(f"Are you sure you want to delete user **{st.session_state.confirm_delete}**? This action cannot be undone.")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, Delete", use_container_width=True, type="primary"):
                        success, message = delete_user(st.session_state.confirm_delete, st.session_state.username)
                        if success:
                            st.success(message)
                            st.session_state.confirm_delete = None
                            st.rerun()
                        else:
                            st.error(message)
                with col_no:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()
        else:
            st.info("No other users to delete.")
        
        # Risk distribution chart
        st.markdown("---")
        st.markdown("### Risk Distribution")
        
        risk_counts = {"Conservative": conservative, "Moderate": moderate, "Aggressive": aggressive}
        not_set = sum(1 for u in users if u.get('Risk Level') == 'Not Set')
        if not_set > 0:
            risk_counts["Not Set"] = not_set
        
        fig = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            color_discrete_sequence=["#00FF94", "#FFD700", "#FF6B6B", "#666666"],
            hole=0.5
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Admin Approval Section (Master Admin Only)
        if st.session_state.is_master:
            st.markdown("---")
            st.markdown("### Pending Admin Approvals")
            
            pending = get_pending_admins()
            
            if pending:
                for pending_user in pending:
                    col_name, col_approve, col_reject = st.columns([3, 1, 1])
                    with col_name:
                        st.markdown(f"**{pending_user}** — requesting admin access")
                    with col_approve:
                        if st.button("Approve", key=f"approve_{pending_user}", type="primary"):
                            ok, msg = approve_admin(pending_user)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_reject:
                        if st.button("Reject", key=f"reject_{pending_user}"):
                            ok, msg = reject_admin(pending_user)
                            if ok:
                                st.info(msg)
                                st.rerun()
                            else:
                                st.error(msg)
            else:
                st.info("No pending admin requests.")
    else:
        st.info("No users registered yet.")


def show_footer():
    """Display the disclaimer footer."""
    st.markdown("""
    <div class="footer">
        <strong>⚠️ Disclaimer:</strong> This is a prototype application for educational purposes only. 
        It does not constitute financial advice. Cryptocurrency investments carry significant risks, 
        including the potential loss of principal. Always conduct your own research and consult with 
        a qualified financial advisor before making investment decisions.
        <br><br>
        🔐 CryptoAI Recommender 
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    if not st.session_state.logged_in:
        show_login_page()
        show_footer()
        return
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"### {st.session_state.username}")
        role_label = st.session_state.user_role.capitalize()
        if st.session_state.is_master:
            role_label = "Master Admin"
        elif st.session_state.user_role == "pending_admin":
            role_label = "Pending Admin"
        st.markdown(f"*Role: {role_label}*")
        st.markdown("---")
        
        # Build menu options based on role and profile status
        menu_options = ["Dashboard"]
        menu_icons = ["speedometer2"]
        
        if not st.session_state.user_profile:
            menu_options = ["Complete Profile"] + menu_options
            menu_icons = ["person-badge"] + menu_icons
        
        if st.session_state.user_role == "admin":
            menu_options.append("Admin Panel")
            menu_icons.append("shield-lock")
        
        menu_options.append("Logout")
        menu_icons.append("box-arrow-right")
        
        selected = option_menu(
            menu_title="Navigation",
            options=menu_options,
            icons=menu_icons,
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#0E1117"},
                "icon": {"color": "#00FF94", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "5px",
                    "padding": "10px",
                    "border-radius": "8px",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 255, 148, 0.2)",
                    "color": "#00FF94",
                },
            },
        )
    
    # Route to selected page
    if selected == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_role = "user"
        st.session_state.user_profile = None
        st.session_state.is_master = False
        st.rerun()
    
    elif selected == "Complete Profile":
        show_onboarding()
    
    elif selected == "Dashboard":
        if st.session_state.user_profile:
            show_user_dashboard()
        else:
            st.warning("Please complete your profile first to get personalized recommendations.")
            show_onboarding()
    
    elif selected == "Admin Panel":
        if st.session_state.user_role == "admin":
            show_admin_dashboard()
        else:
            st.error("Access denied. Admin privileges required.")
    
    show_footer()


if __name__ == "__main__":
    main()
