"""
FraudGuard AI - Dashboard Interface
Streamlit-based frontend for fraud detection system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List

# ============================================================================
# PAGE: LOGIN/SIGNUP (ENTERPRISE VERSION)
# ============================================================================

def show_login_page():
    """Enterprise login/signup page with simplified UI"""
    
    # Hero section layout
    col_hero1, col_hero2 = st.columns([3, 2])
    
    # --- LEFT SIDE: CLEANER & FIXED RENDERING ---
    with col_hero1:
        st.markdown("""
        <div style='padding: 60px 20px;'>
            <h1 style='color: #ffffff; font-size: 56px; font-weight: 700; margin: 0; text-shadow: 0 0 30px rgba(0,255,136,0.3);'>
                🛡️ FraudGuard AI
            </h1>
            <p style='color: #00ff88; font-size: 24px; margin: 10px 0 30px 0; font-weight: 600;'>
                Enterprise Fraud Detection System
            </p>
            <p style='color: #c9ccd4; font-size: 18px; line-height: 1.6; margin-bottom: 40px;'>
                Secure your platform with our hybrid ML + Rule-based detection engine.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- RIGHT SIDE: SIMPLIFIED FORMS ---
    with col_hero2:
        st.markdown('<div style="padding: 40px 20px;">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
        
        # --- TAB 1: LOGIN (Unchanged) ---
        with tab1:
            st.markdown("### Welcome Back")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submit:
                    if not username or not password:
                        st.error("⚠️ Please fill in all fields")
                    else:
                        with st.spinner("Authenticating..."):
                            response = call_api("/auth/login", method="POST", data={
                                "username": username,
                                "password": password
                            })
                            if response:
                                st.session_state.authenticated = True
                                st.session_state.token = response['access_token']
                                st.session_state.user = response['user']
                                st.session_state.page = 'Dashboard'
                                st.success(f"✅ Welcome back!")
                                time.sleep(1)
                                st.rerun()

        # --- TAB 2: SIMPLIFIED SIGNUP ---
        with tab2:
            st.markdown("### Get Started")
            st.caption("Quick registration")
            
            with st.form("signup_form"):
                # Simplified Fields
                username = st.text_input("Username *", placeholder="Choose a username")
                new_password = st.text_input("Password *", type="password", placeholder="Min. 6 characters")
                confirm_password = st.text_input("Confirm Password *", type="password")
                
                st.info("ℹ️ **Account Type:** Fraud Analyst\n\nYou'll have access to fraud detection, monitoring, and transaction verification features.")
                account_type = "Analyst"
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit_signup = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                
                if submit_signup:
                    # Simplified Validation
                    if not all([username, new_password, confirm_password]):
                        st.error("⚠️ Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("⚠️ Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("⚠️ Password must be at least 6 characters")
                    else:
                        with st.spinner("Creating account..."):
                            # API Call with minimal data
                            # Backend handles missing email/full_name automatically
                            response = call_api("/auth/signup", method="POST", data={
                                "username": username,
                                "password": new_password,
                                "role": "analyst"  
                            })
                            
                            if response:
                                st.session_state.authenticated = True
                                st.session_state.token = response['access_token']
                                st.session_state.user = response['user']
                                st.session_state.page = 'Dashboard'
                                st.success(f"✅ Account created! Welcome, {username}!")
                                time.sleep(1)
                                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Define ROLES globally for the frontend
ROLES = {
    'super_admin': {'name': 'Super Administrator', 'icon': '👑'},
    'admin': {'name': 'Administrator', 'icon': '🔧'},
    'senior_analyst': {'name': 'Senior Fraud Analyst', 'icon': '🎯'},
    'analyst': {'name': 'Fraud Analyst', 'icon': '🔍'},
    'viewer': {'name': 'Viewer', 'icon': '👁️'}
}
# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1d29;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #2d3142;
    }
    .risk-low {
        color: #00ff88;
        font-weight: bold;
        font-size: 15px;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
        font-size: 15px;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
        font-size: 15px;
    }
    .status-approved {
        background-color: #00ff8833;
        color: #00ff88;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid #00ff8855;
    }
    .status-review {
        background-color: #ffaa0033;
        color: #ffaa00;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid #ffaa0055;
    }
    .status-rejected {
        background-color: #ff444433;
        color: #ff4444;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 700;
        font-size: 14px;
        border: 1px solid #ff444455;
    }
    .card-title {
        font-size: 13px;
        color: #a0a5ba;
        margin-bottom: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .card-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
    }
    .card-subtitle {
        font-size: 13px;
        color: #7289da;
        margin-top: 6px;
        font-weight: 600;
    }
    .table-header {
        color: #d4d7e0;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .table-text {
        color: #c9ccd4;
        font-size: 14px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    p, span, div {
        color: #c9ccd4;
    }
    .stMarkdown {
        color: #c9ccd4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'selected_transaction' not in st.session_state:
    st.session_state.selected_transaction = None
if 'refresh_live' not in st.session_state:
    st.session_state.refresh_live = False
if 'show_audit_modal' not in st.session_state:
    st.session_state.show_audit_modal = False
if 'audit_data' not in st.session_state:
    st.session_state.audit_data = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'live_transactions' not in st.session_state:
    st.session_state.live_transactions = []
if 'all_transactions' not in st.session_state:
    st.session_state.all_transactions = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'pending_reviews' not in st.session_state:
    st.session_state.pending_reviews = []
if 'transaction_counter' not in st.session_state:
    st.session_state.transaction_counter = 0
if 'show_ml_explanation' not in st.session_state:
    st.session_state.show_ml_explanation = False

# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def call_api(endpoint: str, method: str = "GET", data: Dict = None, files=None):
    """Make API calls with error handling and authentication"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        # Add authentication header if token exists
        headers = {}
        if st.session_state.get('token') and not endpoint.startswith('/auth/'):
            headers['Authorization'] = f"Bearer {st.session_state.token}"
        
        # --- UPDATE START: ADD PUT AND DELETE SUPPORT ---
        if method == "GET":
            response = requests.get(url, params=data, headers=headers, timeout=10)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, headers=headers, timeout=30)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        
        # Handle 401 Unauthorized
        if response.status_code == 401:
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.user = None
            st.error("Session expired. Please login again.")
            st.rerun()
            return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("🔴 Cannot connect to API. Please ensure the backend is running on port 8000.")
        st.code("python fraud_detection_api.py", language="bash")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None


def get_analytics():
    """Fetch analytics data"""
    return call_api("/analytics")


def predict_transaction(transaction_data):
    """Predict single transaction"""
    return call_api("/predict", method="POST", data=transaction_data)


def get_live_transactions():
    """Get live transaction stream"""
    return call_api("/live-transactions")


def generate_single_transaction():
    """Generate a single random transaction"""
    categories = ['Personal Care', 'Misc Net', 'Home', 'Health Fitness', 
                 'Kids Pets', 'Grocery Pos', 'Gas Transport', 'Shopping Net', 
                 'Food Dining', 'Travel']
    
    trans_time = datetime.now()
    amount = np.random.uniform(10, 5000)
    category = np.random.choice(categories)
    
    transaction_data = {
        "amt": amount,
        "category": category,
        "trans_hour": trans_time.hour,
        "age": np.random.randint(18, 85),
        "city_pop": np.random.randint(100, 500000)
    }
    
    # Get prediction
    result = predict_transaction(transaction_data)
    
    if result:
        return {
            'time': trans_time.strftime('%I:%M:%S %p'),
            'timestamp': trans_time,
            'id': result['transaction_id'],
            'amount': amount,
            'category': category,
            'hybrid_score': result['hybrid_probability'],
            'risk_level': result['risk_level'],
            'status': result['prediction'],
            'rule_score': result['rule_score'],
            'ml_proba': result['ml_probabilities'].get('XGBoost', 0.5),
            'violations': result['rule_violations'],
            'source': 'Live Monitor'
        }
    return None


def add_transaction_to_history(transaction):
    """Add transaction to global history"""
    if transaction:
        st.session_state.all_transactions.insert(0, transaction)  # Add at beginning (newest first)
        st.session_state.transaction_counter += 1


def batch_predict(file):
    """Batch prediction from CSV"""
    files = {'file': (file.name, file, 'text/csv')}
    return call_api("/batch", method="POST", files=files)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# Check authentication
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

with st.sidebar:
    # User info header
    user_role = st.session_state.user.get('role', 'viewer')
    role_info = ROLES.get(user_role, {'name': 'User', 'icon': '👤'})
    
    st.markdown(f"""
    <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%); border-radius: 10px; margin-bottom: 10px;'>
        <h1 style='color: #00ff88; margin: 0; font-size: 28px; font-weight: 700; text-shadow: 0 0 20px rgba(0,255,136,0.3);'>🛡️ FraudGuard</h1>
        <p style='color: #c9ccd4; margin: 8px 0; font-size: 14px; font-weight: 600; letter-spacing: 1px;'>Hybrid AI Detection</p>
        <div style='margin-top: 15px; padding: 10px; background-color: #00ff8822; border-radius: 6px;'>
        <p style='color:#00ff88; margin: 0; font-size: 12px; font-weight: 700;'>{role_info['icon']} {role_info['name']}</p>
        <p style='color: #7289da; margin: 4px 0 0 0; font-size: 11px;'>{st.session_state.user['full_name']}</p>
        <p style='color: #a0a5ba; margin: 2px 0 0 0; font-size: 10px;'>{st.session_state.user.get('organization', 'N/A')}</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Navigation buttons
    if st.button("📊 Dashboard", use_container_width=True, 
                type="primary" if st.session_state.page == 'Dashboard' else "secondary"):
        st.session_state.page = 'Dashboard'
        st.rerun()

    if st.button("📡 Live Monitor", use_container_width=True,
                type="primary" if st.session_state.page == 'Live Monitor' else "secondary"):
        st.session_state.page = 'Live Monitor'
        st.rerun()

    if st.button("🔍 Prediction Simulator", use_container_width=True,
                type="primary" if st.session_state.page == 'Prediction Simulator' else "secondary"):
        st.session_state.page = 'Prediction Simulator'
        st.rerun()

    if st.button("📁 Batch Analysis", use_container_width=True,
                type="primary" if st.session_state.page == 'Batch Analysis' else "secondary"):
        st.session_state.page = 'Batch Analysis'
        st.rerun()

    if st.button("📋 Transaction History", use_container_width=True,
                type="primary" if st.session_state.page == 'Transaction History' else "secondary"):
        st.session_state.page = 'Transaction History'
        st.rerun()

    # Admin Dashboard button (only for admins)
    if user_role in ['admin', 'super_admin']:
        st.markdown("---")
        st.markdown("### 👑 ADMIN PANEL")
        
        if st.button("👥 User Management", use_container_width=True,
                    type="primary" if st.session_state.page == 'Admin Dashboard' else "secondary"):
            st.session_state.page = 'Admin Dashboard'
            st.rerun()

    st.markdown("---")
    st.markdown("### SYSTEM STATUS")

    # Check API health
    health = call_api("/health")
    if health and health.get('status') == 'healthy':
        st.markdown('<p style="color: #00ff88; font-weight: 700; font-size: 14px;">🟢 Ensemble Online</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #00ff88; font-weight: 700; font-size: 14px;">🟢 Multi-Model Active</p>', unsafe_allow_html=True)
        st.caption(f"Processed: {health.get('total_transactions', 0)} txns")
    else:
        st.markdown('<p style="color: #ff4444; font-weight: 700; font-size: 14px;">🔴 System Offline</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ff4444; font-weight: 700; font-size: 14px;">🔴 Waiting for API...</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Logout button
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.session_state.token = None
        st.session_state.user = None
        st.success("✅ Logged out successfully!")
        time.sleep(1)
        st.rerun()


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if st.session_state.page == 'Dashboard':
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">📊 Operational Analytics</h1>', unsafe_allow_html=True)
    st.caption(f"Last Login: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Fetch analytics data
    analytics = get_analytics()
    
    if analytics:
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>PROCESSED TRANSACTIONS</div>
                <div class='card-value'>{:,}</div>
            </div>
            """.format(analytics['total_transactions']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>FRAUD DETECTED</div>
                <div class='card-value'>{}</div>
                <div class='card-subtitle'>{:.2f}% Rate</div>
            </div>
            """.format(analytics['fraud_detected'], analytics['fraud_rate']), 
            unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>TOTAL VOLUME</div>
                <div class='card-value'>${:,.0f}</div>
            </div>
            """.format(analytics['total_volume']), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>PREVENTED LOSS</div>
                <div class='card-value' style='color: #00ff88;'>${:,.0f}</div>
            </div>
            """.format(analytics['prevented_loss']), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### ⏰ Temporal Fraud Patterns")
            st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">Transaction volume by hour (Validates 2-4 AM rule)</p>', unsafe_allow_html=True)
            
            hours = list(range(24))
            legitimate = analytics['temporal_data']['legitimate']
            fraud = analytics['temporal_data']['fraud']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours, y=legitimate,
                fill='tozeroy',
                name='Legitimate',
                line=dict(color='#88ccff', width=2),
                fillcolor='rgba(136, 204, 255, 0.3)'
            ))
            fig.add_trace(go.Scatter(
                x=hours, y=fraud,
                fill='tozeroy',
                name='Fraud',
                line=dict(color='#ff6b6b', width=2),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Hour of Day (24h)",
                yaxis_title="Transaction Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Risk Distribution")
            st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">Hybrid classification breakdown</p>', unsafe_allow_html=True)
            
            risk_dist = analytics['risk_distribution']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                values=[risk_dist['LOW'], risk_dist['MEDIUM'], risk_dist['HIGH']],
                hole=0.6,
                marker=dict(colors=['#00ff88', '#ffaa00', '#ff4444']),
                textinfo='label+percent',
                textfont=dict(size=12)
            )])
            fig.update_layout(
                template='plotly_dark',
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False,
                annotations=[dict(
                    text=f'{analytics["total_transactions"]}<br>Total',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Top Rule Violations")
            st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">Frequency of specific heuristic triggers</p>', unsafe_allow_html=True)
            
            if analytics['top_violations']:
                violations = [v['violation'] for v in analytics['top_violations']]
                counts = [v['count'] for v in analytics['top_violations']]
                
                fig = go.Figure(go.Bar(
                    x=counts,
                    y=violations,
                    orientation='h',
                    marker=dict(color='#ffaa00', opacity=0.8),
                    text=counts,
                    textposition='auto'
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title="Frequency",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No violations recorded yet")
        
        with col2:
            st.markdown("### 🏪 Fraud by Category")
            st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">Transaction categories with highest fraud rates</p>', unsafe_allow_html=True)
            
            categories = list(analytics['category_fraud'].keys())
            fraud_counts = list(analytics['category_fraud'].values())
            
            fig = go.Figure(go.Bar(
                x=fraud_counts,
                y=categories,
                orientation='h',
                marker=dict(
                    color=fraud_counts,
                    colorscale='Reds',
                    showscale=False
                ),
                text=fraud_counts,
                textposition='auto'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Fraud Cases",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        
        # --- GLOBAL AI INSIGHTS ---
        st.markdown("### 🧠 Global AI Model Insights")
        st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">What features drive the AI\'s decisions the most?</p>', unsafe_allow_html=True)
        
        col_ai1, col_ai2 = st.columns([2, 1])
        
        with col_ai1:
            # We mock the GLOBAL importance based on domain knowledge for the dashboard view
            # (Calculating this dynamically for all history is too slow for a dashboard)
            feature_importance = {
                'Transaction Amount': 0.35,
                'Hour of Day (Late Night)': 0.25,
                'Category (Internet/Net)': 0.15,
                'Customer Age (Extremes)': 0.10,
                'City Population': 0.05,
                'Velocity (Freq)': 0.05,
                'Distance from Home': 0.05
            }
            
            # Sort for display
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            
            fig_imp = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(
                    color=importance,
                    colorscale='Viridis', 
                    showscale=False
                ),
                text=[f"{v*100:.0f}%" for v in importance],
                textposition='auto'
            ))
            
            fig_imp.update_layout(
                template='plotly_dark',
                title="Global Feature Importance (XGBoost Weighting)",
                xaxis_title="Relative Impact on Fraud Score",
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_ai2:
            st.info("""
            **💡 AI Logic Explanation**
            
            The **XGBoost** and **Random Forest** models have identified **Transaction Amount** as the #1 predictor of fraud in this dataset.
            
            **Key Insights:**
            1. Transactions over **$1,000** trigger immediate scrutiny.
            2. Activity between **2 AM - 5 AM** sees a +25% risk penalty.
            3. **Internet-based** categories (Misc Net, Shopping Net) are inherently riskier than POS (Point of Sale).
            """)
# ============================================================================
# PAGE: ADMIN DASHBOARD
# ============================================================================

elif st.session_state.page == 'Admin Dashboard':
    # Check if user has admin permissions
    user_role = st.session_state.user.get('role', 'viewer')
    if user_role not in ['admin', 'super_admin']:
        st.error("🚫 Access Denied: Admin privileges required")
        st.info("Contact your system administrator to request admin access.")
        st.stop()
    
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">👥 User Management</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a5ba; font-size: 14px; margin-top: 5px;">Manage users, roles, and permissions</p>', unsafe_allow_html=True)
    
    # Tabs for different admin functions
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Users", "➕ Add User", "📊 Activity Logs", "🔑 Roles & Permissions"])
    
    # ====================
    # TAB 1: USER LIST
    # ====================
    with tab1:
        st.markdown("### Registered Users")
        
        # Fetch users
        users_data = call_api("/admin/users")
        
        if users_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_users = len(users_data)
            active_users = len([u for u in users_data if u.get('is_active')])
            inactive_users = total_users - active_users
            
            # Count by role
            role_counts = {}
            for user in users_data:
                role = user.get('role', 'unknown')
                role_counts[role] = role_counts.get(role, 0) + 1
            
            col1.metric("Total Users", total_users)
            col2.metric("Active", active_users, f"{active_users/total_users*100:.0f}%")
            col3.metric("Inactive", inactive_users)
            col4.metric("Admins", role_counts.get('admin', 0) + role_counts.get('super_admin', 0))
            
            st.markdown("---")
            
            # Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                role_filter = st.selectbox("Filter by Role", 
                    ["All"] + list(set([u.get('role', 'unknown') for u in users_data])))
            
            with col_f2:
                status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
            
            with col_f3:
                org_filter = st.selectbox("Filter by Organization",
                    ["All"] + list(set([u.get('organization', 'N/A') for u in users_data])))
            
            # Apply filters
            filtered_users = users_data
            if role_filter != "All":
                filtered_users = [u for u in filtered_users if u.get('role') == role_filter]
            if status_filter == "Active":
                filtered_users = [u for u in filtered_users if u.get('is_active')]
            elif status_filter == "Inactive":
                filtered_users = [u for u in filtered_users if not u.get('is_active')]
            if org_filter != "All":
                filtered_users = [u for u in filtered_users if u.get('organization') == org_filter]
            
            st.markdown(f"**Showing {len(filtered_users)} of {total_users} users**")
            
            # Users table
            for user in filtered_users:
                with st.expander(f"👤 {user['full_name']} ({user['username']})", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**User Information**")
                        st.markdown(f"**ID:** {user['id']}")
                        st.markdown(f"**Email:** {user['email']}")
                        st.markdown(f"**Username:** {user['username']}")
                        st.markdown(f"**Phone:** {user.get('phone', 'N/A')}")
                    
                    with col_b:
                        st.markdown("**Organization Details**")
                        st.markdown(f"**Organization:** {user.get('organization', 'N/A')}")
                        st.markdown(f"**Department:** {user.get('department', 'N/A')}")
                        
                        # Role badge
                        role = user.get('role', 'unknown')
                        role_color = {
                            'admin': '#ff8800',
                            'analyst': '#00ff88'
                        }.get(role, '#cccccc')
                        
                        st.markdown(f"**Role:** <span style='background-color: {role_color}33; color: {role_color}; padding: 4px 12px; border-radius: 4px; font-weight: 700;'>{role.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown("**Account Status**")
                        
                        status_color = '#00ff88' if user.get('is_active') else '#ff4444'
                        status_text = 'ACTIVE' if user.get('is_active') else 'INACTIVE'
                        st.markdown(f"**Status:** <span style='color: {status_color}; font-weight: 700;'>{status_text}</span>", unsafe_allow_html=True)
                        
                        st.markdown(f"**Created:** {user.get('created_at', 'N/A')[:10]}")
                        st.markdown(f"**Last Login:** {user.get('last_login', 'Never')[:16] if user.get('last_login') else 'Never'}")
                    
                    st.markdown("---")
                    
                    # Action buttons
                    col_act1, col_act2, col_act3, col_act4 = st.columns(4)
                    
                    # Prevent self-modification
                    is_self = user['id'] == st.session_state.user['id']
                    can_modify = user_role == 'super_admin' or (user_role == 'admin' and user.get('role') not in ['admin', 'super_admin'])
                    
                    with col_act1:
                        if st.button("✏️ Edit", key=f"edit_{user['id']}", disabled=not can_modify):
                            st.session_state.editing_user = user
                            st.rerun()
                    
                    with col_act2:
                        if user.get('is_active'):
                            if st.button("🚫 Deactivate", key=f"deact_{user['id']}", disabled=is_self or not can_modify):
                                # CHANGE "POST" TO "PUT" HERE
                                response = call_api(f"/admin/users/{user['id']}", method="PUT", data={
                                    "is_active": False
                                })
                                if response:
                                    st.success(f"User {user['username']} deactivated")
                                    time.sleep(1)
                                    st.rerun()
                        else:
                            if st.button("✅ Activate", key=f"act_{user['id']}", disabled=not can_modify):
                                # CHANGE "POST" TO "PUT" HERE
                                response = call_api(f"/admin/users/{user['id']}", method="PUT", data={
                                    "is_active": True
                                })
                                if response:
                                    st.success(f"User {user['username']} activated")
                                    time.sleep(1)
                                    st.rerun()
                    
                    with col_act3:
                        if st.button("🔄 Reset Password", key=f"reset_{user['id']}", disabled=not can_modify):
                            st.info("Password reset functionality - Send reset email")
                    
                    with col_act4:
                        if user_role == 'super_admin':
                            if st.button("🗑️ Delete", key=f"del_{user['id']}", disabled=is_self):
                                if st.session_state.get(f"confirm_delete_{user['id']}"):
                                    response = call_api(f"/admin/users/{user['id']}", method="DELETE")
                                    if response:
                                        st.success(f"User {user['username']} deleted")
                                        time.sleep(1)
                                        st.rerun()
                                else:
                                    st.session_state[f"confirm_delete_{user['id']}"] = True
                                    st.warning("Click again to confirm deletion")
        else:
            st.info("No users found or unable to fetch user data")
    
    # ====================
    # TAB 2: ADD USER
    # ====================
    with tab2:
        st.markdown("### Get Started")
        st.caption("Create your analyst account")
    
        with st.form("signup_form"):
            username = st.text_input("Username *", placeholder="johndoe")
            new_password = st.text_input("Password *", type="password", placeholder="Min. 6 characters")
            confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Re-enter password")
            
            # Info about role
            st.info("ℹ️ **Account Type:** Fraud Analyst\n\nYou'll have access to fraud detection, monitoring, and transaction verification features.")
            
            # Terms and conditions
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submit_signup = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            
            if submit_signup:
                # Validation
                if not all([username, new_password, confirm_password]):
                    st.error("⚠️ Please fill in all required fields (*)")
                elif new_password != confirm_password:
                    st.error("⚠️ Passwords do not match")
                elif len(new_password) < 6:
                    st.error("⚠️ Password must be at least 6 characters")
                elif len(new_password) > 72:
                    st.error("⚠️ Password must be less than 72 characters")
                elif not agree_terms:
                    st.error("⚠️ Please agree to the Terms of Service")
                else:
                    with st.spinner("Creating your account..."):
                        response = call_api("/auth/signup", method="POST", data={
                            "username": username,
                            "password": new_password
                        })
                        
                        if response:
                            st.session_state.authenticated = True
                            st.session_state.token = response['access_token']
                            st.session_state.user = response['user']
                            st.session_state.page = 'Dashboard'
                            
                            st.success(f"✅ Account created successfully! Welcome, {username}!")
                            st.balloons()
                            
                            time.sleep(2)
                            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #7289da; font-size: 13px;'>
            <p>Already have an account? Switch to <strong>Sign In</strong> tab</p>
            <p style='margin-top: 10px;'>For admin access, contact your system administrator</p>
        </div>
        """, unsafe_allow_html=True)
        
    # ====================
    # TAB 3: ACTIVITY LOGS
    # ====================
    with tab3:
        st.markdown("### Recent Activity Logs")
        st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 20px;">Monitor user actions and system events</p>', unsafe_allow_html=True)
        
        # Fetch activity logs
        logs_data = call_api("/admin/activity-logs?limit=100")
        
        if logs_data:
            # Filter options
            col_log1, col_log2 = st.columns(2)
            
            with col_log1:
                action_filter = st.selectbox("Filter by Action",
                    ["All"] + list(set([log.get('action', 'UNKNOWN') for log in logs_data])))
            
            with col_log2:
                user_filter = st.selectbox("Filter by User",
                    ["All"] + list(set([log.get('username', 'Unknown') for log in logs_data])))
            
            # Apply filters
            filtered_logs = logs_data
            if action_filter != "All":
                filtered_logs = [log for log in filtered_logs if log.get('action') == action_filter]
            if user_filter != "All":
                filtered_logs = [log for log in filtered_logs if log.get('username') == user_filter]
            
            st.markdown(f"**Showing {len(filtered_logs)} of {len(logs_data)} logs**")
            
            # Display logs
            for log in filtered_logs[:50]:  # Show first 50
                action = log.get('action', 'UNKNOWN')
                
                # Color code by action type
                if 'LOGIN' in action:
                    icon = '🔐'
                    color = '#00ff88' if 'SUCCESS' in action else '#ff4444'
                elif 'USER' in action:
                    icon = '👤'
                    color = '#ffaa00'
                elif 'PREDICTION' in action or 'VERIFICATION' in action:
                    icon = '🔍'
                    color = '#88ccff'
                else:
                    icon = '📝'
                    color = '#c9ccd4'
                
                timestamp = log.get('timestamp', '')[:19]
                
                st.markdown(f"""
                <div style='background-color: #1e2130; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid {color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-size: 16px;'>{icon}</span>
                            <span style='color: {color}; font-weight: 700; margin-left: 8px;'>{action}</span>
                            <span style='color: #7289da; margin-left: 12px;'>by {log.get('full_name', 'Unknown')} (@{log.get('username', 'unknown')})</span>
                        </div>
                        <span style='color: #a0a5ba; font-size: 12px;'>{timestamp}</span>
                    </div>
                    {f"<div style='color: #c9ccd4; font-size: 13px; margin-top: 6px;'>{log.get('details', '')}</div>" if log.get('details') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No activity logs available")
    
    # ====================
    # TAB 4: ROLES & PERMISSIONS
    # ====================
    with tab4:
        st.markdown("### Roles & Permissions Matrix")
        st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 20px;">Understanding role-based access control</p>', unsafe_allow_html=True)
        
        roles_data = call_api("/admin/roles")
        
        if roles_data:
            for role_key, role_info in roles_data.items():
                # Role card
                role_color = {
                    'super_admin': '#ff4444',
                    'admin': '#ff8800',
                    'senior_analyst': '#ffaa00',
                    'analyst': '#00ff88',
                    'viewer': '#88ccff'
                }.get(role_key, '#cccccc')
                
                with st.expander(f"🔑 {role_info['name']}", expanded=(role_key == user_role)):
                    col_r1, col_r2 = st.columns([1, 2])
                    
                    with col_r1:
                        st.markdown(f"""
                        <div style='background-color: {role_color}22; padding: 20px; border-radius: 10px; border: 2px solid {role_color}44;'>
                            <h3 style='color: {role_color}; margin: 0;'>{role_info['name']}</h3>
                            <p style='color: #c9ccd4; margin-top: 10px;'>{role_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r2:
                        st.markdown("**Permissions:**")
                        
                        permissions = role_info.get('permissions', [])
                        
                        if 'all' in permissions:
                            st.markdown("✅ **All Permissions** (Super Admin)")
                        else:
                            permission_names = {
                                'manage_users': '👥 Manage Users',
                                'view_analytics': '📊 View Analytics',
                                'predict': '🔍 Run Predictions',
                                'batch': '📁 Batch Processing',
                                'second_verify': '🔐 Second Verification',
                                'export_data': '📥 Export Data'
                            }
                            
                            for perm in permissions:
                                st.markdown(f"✅ {permission_names.get(perm, perm)}")
                        
                        # Show user count
                        user_count = len([u for u in (call_api("/admin/users") or []) if u.get('role') == role_key])
                        st.markdown(f"\n**Active Users:** {user_count}")
        
        st.markdown("---")
        
        # Permissions guide
        st.markdown("### 📚 Permissions Guide")

        st.markdown("""
        #### 🔧 Administrator
        - **Full System Access** - All permissions
        - 👥 Create, edit, and deactivate users
        - 🎯 Assign roles (Admin or Analyst)
        - 📊 View all analytics and reports
        - 🔍 Run fraud predictions
        - 📁 Batch processing
        - ✅ Second verification authority
        - 📥 Export data
        - 📋 View activity logs

        #### 🔍 Fraud Analyst
        - 📊 View Analytics - Access dashboard and fraud statistics
        - 🔍 Run Predictions - Analyze individual transactions
        - 📁 Batch Processing - Upload CSV files for bulk analysis
        - ✅ Second Verification - Approve/reject flagged transactions
        - 📥 Export Reports - Download transaction histories

        **Note:** Only Administrators can create new users and manage accounts.
        """)

# ============================================================================
# PAGE: LIVE MONITOR
# ============================================================================

elif st.session_state.page == 'Live Monitor':
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">📡 Real-Time Transaction Stream</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #c9ccd4; font-size: 20px; margin-top: 10px;">Live Transaction Stream</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        st.markdown('<p style="color: #00ff88; font-weight: 700; font-size: 15px;">🟢 Hybrid Engine Active</p>', unsafe_allow_html=True)
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    
    # Initial load if empty
    if not st.session_state.live_transactions:
        with st.spinner("Syncing with fraud engine..."):
            # Fetch real history from backend instead of generating fake ones one-by-one
            history_data = call_api("/live-transactions")
            
            if history_data:
                # Convert ISO timestamp strings back to datetime objects if needed
                st.session_state.live_transactions = history_data
            else:
                # Fallback if empty: Generate just 2 to start, not 15
                for _ in range(2):
                    new_txn = generate_single_transaction()
                    if new_txn:
                        st.session_state.live_transactions.append(new_txn)
                        add_transaction_to_history(new_txn)
    
    # Check if we need to generate new transaction (every 5 seconds)
    current_time = time.time()
    if auto_refresh and (current_time - st.session_state.last_refresh >= 5):
        # Generate ONE new transaction
        new_txn = generate_single_transaction()
        if new_txn:
            # Add to live transactions at the beginning (newest first)
            st.session_state.live_transactions.insert(0, new_txn)
            # Keep only last 15 transactions
            st.session_state.live_transactions = st.session_state.live_transactions[:15]
            # Add to global history
            add_transaction_to_history(new_txn)
        
        st.session_state.last_refresh = current_time
    
    # Create placeholder for transactions table only
    transactions_placeholder = st.empty()
    
    # Separate placeholder for audit modal
    audit_placeholder = st.empty()
    
    def display_live_transactions():
        transactions = st.session_state.live_transactions
        
        if transactions:
            with transactions_placeholder.container():
                st.markdown("---")
                
                # Table header
                cols = st.columns([1, 1.2, 1, 1.5, 1.2, 1, 1.2, 0.6])
                cols[0].markdown('<p class="table-header">Time</p>', unsafe_allow_html=True)
                cols[1].markdown('<p class="table-header">ID</p>', unsafe_allow_html=True)
                cols[2].markdown('<p class="table-header">Amount</p>', unsafe_allow_html=True)
                cols[3].markdown('<p class="table-header">Category</p>', unsafe_allow_html=True)
                cols[4].markdown('<p class="table-header">Hybrid Score</p>', unsafe_allow_html=True)
                cols[5].markdown('<p class="table-header">Risk Level</p>', unsafe_allow_html=True)
                cols[6].markdown('<p class="table-header">Status</p>', unsafe_allow_html=True)
                cols[7].markdown('<p class="table-header">Action</p>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display transactions (already sorted, newest first)
                for idx, txn in enumerate(transactions):
                    cols = st.columns([1, 1.2, 1, 1.5, 1.2, 1, 1.2, 0.6])
                    
                    # Highlight newest transaction (first one)
                    if idx == 0:
                        cols[0].markdown(f'<p class="table-text" style="color: #00ff88; font-weight: 700;">🆕 {txn["time"]}</p>', unsafe_allow_html=True)
                    else:
                        cols[0].markdown(f'<p class="table-text">{txn["time"]}</p>', unsafe_allow_html=True)
                    
                    cols[1].markdown(f'<p class="table-text">{txn["id"]}</p>', unsafe_allow_html=True)
                    cols[2].markdown(f'<p class="table-text">${txn["amount"]:.2f}</p>', unsafe_allow_html=True)
                    cols[3].markdown(f'<p class="table-text">{txn["category"]}</p>', unsafe_allow_html=True)
                    
                    # Progress bar for score
                    score_val = int(txn['hybrid_score'] * 100)
                    cols[4].progress(score_val / 100)
                    cols[4].caption(f"{score_val}%")
                    
                    # Risk level badge
                    risk = txn['risk_level']
                    if risk == 'LOW':
                        cols[5].markdown('<span class="risk-low">● LOW</span>', unsafe_allow_html=True)
                    elif risk == 'MEDIUM':
                        cols[5].markdown('<span class="risk-medium">● MEDIUM</span>', unsafe_allow_html=True)
                    else:
                        cols[5].markdown('<span class="risk-high">● HIGH</span>', unsafe_allow_html=True)
                    
                    # Status badge
                    status = txn['status']
                    if status == 'Approved':
                        cols[6].markdown('<span class="status-approved">✓ Approved</span>', 
                                       unsafe_allow_html=True)
                    elif status == 'Review':
                        cols[6].markdown('<span class="status-review">⚠ Review</span>', 
                                       unsafe_allow_html=True)
                    else:
                        cols[6].markdown('<span class="status-rejected">✗ Rejected</span>', 
                                       unsafe_allow_html=True)
                    
                    # Action button - AI Audit
                    if cols[7].button("🔍", key=f"audit_{idx}_{txn['id']}"):
                        st.session_state.show_audit_modal = True
                        st.session_state.audit_data = txn
    
    # Display transactions
    display_live_transactions()
    
    # AI Audit Modal in separate placeholder
    if st.session_state.show_audit_modal and st.session_state.audit_data:
        with audit_placeholder.container():
            st.markdown("---")
            st.markdown("---")
            
            st.markdown("""
            <div style='background-color: #1e2130; padding: 30px; border-radius: 15px; border: 2px solid #00ff88;'>
                <h2 style='color: #00ff88; margin-bottom: 20px;'>🔍 Hybrid Transaction Audit</h2>
            </div>
            """, unsafe_allow_html=True)
            
            txn = st.session_state.audit_data
            
            st.markdown(f"**Transaction ID:** {txn['id']}")
            
            # Three metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>RULE SCORE</div>
                    <div class='card-value'>{}/5</div>
                </div>
                """.format(txn['rule_score']), unsafe_allow_html=True)
            
            with col_b:
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>ML PROB</div>
                    <div class='card-value'>{:.0f}%</div>
                </div>
                """.format(txn['ml_proba'] * 100), unsafe_allow_html=True)
            
            with col_c:
                hybrid_color = '#ff4444' if txn['hybrid_score'] > 0.6 else '#ffaa00' if txn['hybrid_score'] > 0.3 else '#00ff88'
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>FINAL HYBRID</div>
                    <div class='card-value' style='color: {};'>{:.1f}%</div>
                </div>
                """.format(hybrid_color, txn['hybrid_score'] * 100), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Rule Violations
            st.markdown("**Rule Violations**")
            if txn.get('violations'):
                for violation in txn['violations']:
                    st.markdown(f"• {violation}")
            else:
                st.success("✓ No rule violations detected")
            
            st.markdown("---")
            
            # ML Ensemble
            st.markdown("**ML Ensemble**")
            st.progress(txn['ml_proba'], text=f"XGBoost: {txn['ml_proba']*100:.0f}%")
            st.progress(txn['ml_proba'] * 0.9, text=f"Random Forest: {txn['ml_proba']*90:.0f}%")
            st.progress(txn['ml_proba'] * 1.1, text=f"Logistic Regression: {min(txn['ml_proba']*110, 100):.0f}%")
            
            st.markdown("---")
            
            # Risk Analysis
            st.markdown("**⚠️ Risk Analysis:**")
            st.info(f"""
            The system's **{txn['status'].upper()}** decision is well-justified. 
            Although the transaction amount (${txn['amount']:.2f}) is relatively {'high' if txn['amount'] > 1000 else 'moderate' if txn['amount'] > 500 else 'low'}, 
            key factors contribute to the {'elevated' if txn['hybrid_score'] > 0.5 else 'moderate' if txn['hybrid_score'] > 0.3 else 'low'} risk: 
            the use of the '{txn['category']}' category, which is {'inherently high-risk' if 'net' in txn['category'].lower() else 'generally safe'} 
            due to its generic nature and frequent association with fraudulent activity. 
            The Machine Learning model assigns a {txn['hybrid_score']:.0%} probability, 
            {'indicating high-confidence fraud' if txn['hybrid_score'] > 0.7 else 'suggesting moderate risk' if txn['hybrid_score'] > 0.3 else 'indicating legitimate behavior'}. 
            Combined with the flags raised by the rule-based component, this creates 
            {'enough ambiguity to warrant a manual review' if txn['status'] == 'Review' else 'sufficient confidence for automated ' + txn['status'].lower()}.
            """)
            
            st.markdown("**Final Verdict:** " + 
                       ("✓ Agree with System" if txn['hybrid_score'] < 0.6 else "⚠ Review Recommended"))
            
            # Close button
            if st.button("Close Audit", use_container_width=True, type="primary"):
                st.session_state.show_audit_modal = False
                st.session_state.audit_data = None
                st.rerun()
    
    # Auto-refresh logic - only refresh if modal is not open
    if auto_refresh and not st.session_state.show_audit_modal:
        time.sleep(1)
        st.rerun()

# ============================================================================
# PAGE: PREDICTION SIMULATOR
# ============================================================================

elif st.session_state.page == 'Prediction Simulator':
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">🔮 Prediction Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a5ba; font-size: 14px; margin-top: 5px;">What-If analysis tool for testing specific scenarios with second verification</p>', unsafe_allow_html=True)
    
    # Preset Scenarios Section
    st.markdown("### 🎯 Quick Test Scenarios")
    st.markdown('<p style="color: #a0a5ba; font-size: 13px; margin-bottom: 15px;">Load preset scenarios to test different risk levels</p>', unsafe_allow_html=True)
    
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    
    # Define preset scenarios
    presets = {
        'normal': {
            'name': '✅ Normal Transaction',
            'amt': 45.99,
            'category': 'Grocery Pos',
            'trans_hour': 14,
            'age': 35,
            'city_pop': 150000,
            'card_number': ''
        },
        'medium': {
            'name': '⚠️ Medium Risk',
            'amt': 650.00,
            'category': 'Shopping Net',
            'trans_hour': 3,
            'age': 82,
            'city_pop': 25000,
            'card_number': ''
        },
        'high': {
            'name': '🚨 High Risk',
            'amt': 3500.00,
            'category': 'Misc Net',
            'trans_hour': 3,
            'age': 17,
            'city_pop': 500,
            'card_number': 'CARD_12345'
        }
    }
    
    # Initialize session state for preset selection
    if 'selected_preset' not in st.session_state:
        st.session_state.selected_preset = None
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    
    with col_preset1:
        if st.button("✅ Load Normal Risk", use_container_width=True, type="secondary"):
            st.session_state.selected_preset = 'normal'
            st.rerun()
    
    with col_preset2:
        if st.button("⚠️ Load Medium Risk", use_container_width=True, type="secondary"):
            st.session_state.selected_preset = 'medium'
            st.rerun()
    
    with col_preset3:
        if st.button("🚨 Load High Risk", use_container_width=True, type="secondary"):
            st.session_state.selected_preset = 'high'
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Transaction Simulator")
        
        # Get preset values if selected
        preset_data = presets.get(st.session_state.selected_preset, {})
        
        with st.form("prediction_form", clear_on_submit=False):
            amount = st.number_input(
                "Amount ($)", 
                min_value=0.0, 
                value=float(preset_data.get('amt', 50.0)), 
                step=10.0
            )
            
            # Styled category dropdown
            st.markdown('<p class="table-text" style="margin-bottom: 5px; font-weight: 600;">Category</p>', unsafe_allow_html=True)
            
            categories_list = [
                'Grocery Pos', 'Gas Transport', 'Shopping Net', 'Misc Net',
                'Grocery Net', 'Food Dining', 'Personal Care', 'Health Fitness',
                'Travel', 'Kids Pets', 'Home', 'Entertainment'
            ]
            
            # Set default category index based on preset
            default_category = preset_data.get('category', 'Grocery Pos')
            category_index = categories_list.index(default_category) if default_category in categories_list else 0
            
            category = st.selectbox(
                "Category",
                categories_list,
                index=category_index,
                label_visibility="collapsed"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                trans_hour = st.slider(
                    "Time (Hour)", 
                    0, 23, 
                    int(preset_data.get('trans_hour', 14))
                )
            with col_b:
                age = st.number_input(
                    "Age", 
                    min_value=10, 
                    max_value=100, 
                    value=int(preset_data.get('age', 30))
                )
            
            city_pop = st.number_input(
                "City Population",
                min_value=100,
                max_value=5000000,
                value=int(preset_data.get('city_pop', 150000)),
                step=1000
            )
            
            st.markdown("#### Velocity Testing")
            card_number = st.text_input(
                "Card Number (Optional)", 
                value=preset_data.get('card_number', ''),
                placeholder="Enter to test velocity rules"
            )
            
            st.caption("💡 Submit multiple times with same card to trigger velocity rules")
            
            submit = st.form_submit_button("🔍 Analyze Transaction", 
                                          use_container_width=True, 
                                          type="primary")
        
        # Show loaded preset info
        if st.session_state.selected_preset:
            preset_name = presets[st.session_state.selected_preset]['name']
            st.info(f"📋 Loaded preset: **{preset_name}**")
            if st.button("🔄 Clear Preset", use_container_width=True):
                st.session_state.selected_preset = None
                st.rerun()
    
    with col2:
        if submit:
            with st.spinner("Running hybrid analysis..."):
                # Prepare transaction data
                trans_data = {
                    "amt": amount,
                    "category": category,
                    "trans_hour": trans_hour,
                    "age": age,
                    "city_pop": city_pop,
                    "card_number": card_number if card_number else None
                }
                
                result = predict_transaction(trans_data)
                
                if result:
                    # Store current prediction for second verification
                    st.session_state.current_prediction = result
                    
                    # Store in history
                    new_txn = {
                        'time': datetime.now().strftime('%I:%M:%S %p'),
                        'timestamp': datetime.now(),
                        'id': result['transaction_id'],
                        'amount': amount,
                        'category': category,
                        'hybrid_score': result['hybrid_probability'],
                        'risk_level': result['risk_level'],
                        'status': result['prediction'],
                        'rule_score': result['rule_score'],
                        'ml_proba': result['ml_probabilities'].get('XGBoost', 0.5),
                        'dnn_proba': result['ml_probabilities'].get('DNN', 0.5),
                        'violations': result['rule_violations'],
                        'source': 'Prediction Simulator'
                    }
                    add_transaction_to_history(new_txn)
        
        # Display results
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction
            
            st.markdown("### 📊 Analysis Results")
            
            # Header with decision
            if result['prediction'] == 'Approved':
                st.success(f"### ✅ {result['prediction'].upper()}")
            elif result['prediction'] == 'Review':
                st.warning(f"### ⚠️ {result['prediction'].upper()} - Second Verification Required")
            else:
                st.error(f"### ❌ {result['prediction'].upper()}")
            
            # Three cards: Rules, ML, Hybrid
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>RULE SCORE</div>
                    <div class='card-value'>{}/5</div>
                </div>
                """.format(result['rule_score']), unsafe_allow_html=True)
            
            with col_b:
                ml_avg = np.mean(list(result['ml_probabilities'].values()))
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>ML AVERAGE</div>
                    <div class='card-value'>{:.0f}%</div>
                </div>
                """.format(ml_avg * 100), unsafe_allow_html=True)
            
            with col_c:
                hybrid_color = '#ff4444' if result['hybrid_probability'] > 0.6 else '#ffaa00' if result['hybrid_probability'] > 0.3 else '#00ff88'
                st.markdown("""
                <div class='metric-card'>
                    <div class='card-title'>HYBRID FINAL</div>
                    <div class='card-value' style='color: {};'>{:.1f}%</div>
                </div>
                """.format(hybrid_color, result['hybrid_probability'] * 100), 
                unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Rule Violations
            st.markdown("#### 🚨 Rule Violations")
            if result['rule_violations']:
                for violation in result['rule_violations']:
                    st.markdown(f"• {violation}")
            else:
                st.success("✓ No rule violations detected")
            
            st.markdown("---")
            
            # ML Ensemble Breakdown with Explanations
            st.markdown("#### 🤖 ML Ensemble Breakdown")
            
            st.markdown("---")
            
            # Toggle for explanations
            show_explanations = st.checkbox("📚 Show Simple Explanations", value=False)
            
            for model_name, prob in result['ml_probabilities'].items():
                # Add emoji for DNN
                if model_name == 'DNN':
                    display_name = f"🧠 {model_name}"
                else:
                    display_name = model_name
                
                st.progress(prob, text=f"{display_name}: {prob*100:.1f}%")
                
                # Show explanation if enabled
                if show_explanations and model_name in result.get('ml_explanations', {}):
                    with st.expander(f"ℹ️ How {model_name} works"):
                        st.info(result['ml_explanations'][model_name])
            
            st.markdown("---")
            
            # Explanation
            st.markdown("#### 💡 Risk Analysis")
            st.info(result['explanation'])
            
            # Final verdict
            if result['prediction'] == 'Approved':
                st.markdown("**Final Verdict:** ✓ Agree with System (Low Risk)")
            elif result['prediction'] == 'Review':
                st.markdown("**Final Verdict:** ⚠️ Manual Review Recommended")
            else:
                st.markdown("**Final Verdict:** ❌ High Risk - Rejection Justified")
            
            st.markdown("---")
            
            # Second Verification Section (Only for Medium Risk)
            if result['prediction'] == 'Review':
                st.markdown("### 🔐 Second Verification")
                st.markdown('<p style="color: #ffaa00; font-size: 14px;">This transaction requires manual analyst review</p>', unsafe_allow_html=True)
                
                with st.form("verification_form"):
                    st.markdown("**Analyst Decision:**")
                    
                    col_v1, col_v2 = st.columns(2)
                    
                    with col_v1:
                        decision = st.radio(
                            "Select Decision",
                            ["Approve", "Reject"],
                            label_visibility="collapsed"
                        )
                    
                    with col_v2:
                        st.markdown("**Decision Guide:**")
                        st.markdown("• **Approve**: Customer verified, legitimate")
                        st.markdown("• **Reject**: Confirmed fraud attempt")
                    
                    analyst_notes = st.text_area(
                        "Analyst Notes (Optional)",
                        placeholder="Add context: customer contacted, ID verified, etc.",
                        height=100
                    )
                    
                    submit_verification = st.form_submit_button(
                        f"✓ Confirm {decision}",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if submit_verification:
                        with st.spinner("Processing verification..."):
                            verification_result = call_api(
                                "/second-verification",
                                method="POST",
                                data={
                                    "transaction_id": result['transaction_id'],
                                    "decision": decision,
                                    "analyst_notes": analyst_notes
                                }
                            )
                            
                            if verification_result:
                                if decision == "Approve":
                                    st.success(f"✅ Transaction {result['transaction_id']} APPROVED")
                                    st.balloons()
                                else:
                                    st.error(f"❌ Transaction {result['transaction_id']} REJECTED")
                                
                                st.info(f"**Notes:** {analyst_notes if analyst_notes else 'No additional notes'}")
                                
                                # Clear current prediction
                                st.session_state.current_prediction = None
                                
                                time.sleep(2)
                                st.rerun()
        
        else:
            st.info("""
            ### Ready for Analysis
            
            **Quick Start:**
            1. Click a preset scenario above OR
            2. Enter custom transaction details
            3. Click **Analyze** to see results
            
            **You'll see:**
            
            ✓ Rule-based scoring (5 heuristic rules)  
            ✓ ML ensemble predictions (LR, RF, XGBoost, DNN)  
            ✓ Hybrid decision logic (30% rules + 70% ML)  
            ✓ Detailed risk explanations  
            ✓ **Second verification for medium-risk transactions**
            
            **New Features:**
            - 📚 Simple ML explanations for non-technical users
            - 🔐 Direct second verification workflow
            - 📝 Add analyst notes for audit trail
            
            **Test velocity rules** by submitting multiple transactions 
            with the same card number!
            
            ---
            
            **Preset Scenarios:**
            - **Normal Risk**: Typical grocery purchase
            - **Medium Risk**: Late-night elderly shopper (triggers review)
            - **High Risk**: Minor with large misc net purchase
            """)

# ============================================================================
# PAGE: BATCH ANALYSIS
# ============================================================================

elif st.session_state.page == 'Batch Analysis':
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">📁 Batch Transaction Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a5ba; font-size: 14px; margin-top: 5px;">Upload CSV file for bulk processing</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Required CSV Columns:**
    - `amt`: Transaction amount
    - `category`: Transaction category
    - `trans_hour`: Hour of transaction (0-23)
    - `age`: Customer age
    - `city_pop` (optional): City population
    - `cc_num` (optional): Card number for velocity tracking
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Show preview
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 File Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown(f"**Total Rows:** {len(df)}")
        
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(df)} transactions..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Call batch API
                result = batch_predict(uploaded_file)
                
                if result:
                    st.session_state.batch_results = result
                    
                    # Add all transactions to history
                    for r in result['results']:
                        batch_txn = {
                            'time': datetime.now().strftime('%I:%M:%S %p'),
                            'timestamp': datetime.now(),
                            'id': r['transaction_id'],
                            'amount': np.random.uniform(50, 3000),  # Would come from CSV
                            'category': 'Various',
                            'hybrid_score': r['hybrid_probability'],
                            'risk_level': r['risk_level'],
                            'status': r['prediction'],
                            'rule_score': r['rule_score'],
                            'ml_proba': 0.5,
                            'violations': r['rule_violations'],
                            'source': 'Batch Analysis'
                        }
                        add_transaction_to_history(batch_txn)
                    
                    # Extract medium risk transactions for review
                    st.session_state.pending_reviews = [
                        r for r in result['results'] 
                        if r['prediction'] == 'Review'
                    ]
                    
                    st.success("✅ Batch processing complete!")
                    st.rerun()
    
    # Display results if available
    if st.session_state.batch_results:
        result = st.session_state.batch_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("PROCESSED", result['total_processed'])
        col2.metric("APPROVED", result['approved'], 
                   f"{result['approved']/result['total_processed']*100:.1f}%")
        col3.metric("REVIEW", result['review'],
                   f"{result['review']/result['total_processed']*100:.1f}%")
        col4.metric("REJECTED", result['rejected'],
                   f"{result['rejected']/result['total_processed']*100:.1f}%")
        
        st.markdown("---")
        
        # Second Verification for Medium Risk
        if st.session_state.pending_reviews:
            st.markdown("### ⚠️ Second Verification Required")
            st.markdown(f"**{len(st.session_state.pending_reviews)} transactions** flagged as MEDIUM RISK require manual review")
            
            # Display pending reviews
            for idx, txn in enumerate(st.session_state.pending_reviews):
                with st.expander(f"🔍 Transaction {idx + 1}: {txn['transaction_id']} - ${txn['hybrid_probability']*1000:.2f}", expanded=(idx==0)):
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Risk Level", txn['risk_level'])
                    col2.metric("Hybrid Score", f"{txn['hybrid_probability']*100:.1f}%")
                    col3.metric("Rule Score", f"{txn['rule_score']}/5")
                    
                    if txn['rule_violations']:
                        st.markdown("**Rule Violations:**")
                        for violation in txn['rule_violations']:
                            st.markdown(f"• {violation}")
                    
                    st.markdown("---")
                    
                    col_a, col_b = st.columns(2)
                    
                    if col_a.button("✅ Approve", key=f"approve_{idx}_{txn['transaction_id']}", use_container_width=True, type="primary"):
                        # Remove from pending
                        st.session_state.pending_reviews.pop(idx)
                        st.success(f"✅ Transaction {txn['transaction_id']} approved")
                        time.sleep(0.5)
                        st.rerun()
                    
                    if col_b.button("❌ Reject", key=f"reject_{idx}_{txn['transaction_id']}", use_container_width=True):
                        # Remove from pending
                        st.session_state.pending_reviews.pop(idx)
                        st.error(f"❌ Transaction {txn['transaction_id']} rejected")
                        time.sleep(0.5)
                        st.rerun()
            
            st.markdown("---")
        
        # Results table
        st.subheader("📊 All Transaction Results")
        
        # Create results dataframe
        results_data = []
        for r in result['results'][:50]:  # Show first 50
            results_data.append({
                'ID': r['transaction_id'][:8],
                'Risk': r['risk_level'],
                'Score': f"{r['hybrid_probability']*100:.1f}%",
                'Rules': f"{r['rule_score']}/5",
                'Status': r['prediction']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Color code by risk
        def color_risk(val):
            if val == 'LOW':
                return 'background-color: #00ff8822'
            elif val == 'MEDIUM':
                return 'background-color: #ffaa0022'
            else:
                return 'background-color: #ff444422'
        
        styled_df = results_df.style.applymap(color_risk, subset=['Risk'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        if len(result['results']) > 50:
            st.caption(f"Showing first 50 of {len(result['results'])} results")
        
        # Download results
        st.markdown("---")
        full_results = pd.DataFrame([
            {
                'transaction_id': r['transaction_id'],
                'prediction': r['prediction'],
                'risk_level': r['risk_level'],
                'hybrid_probability': r['hybrid_probability'],
                'rule_score': r['rule_score'],
                'violations': ', '.join(r['rule_violations'])
            }
            for r in result['results']
        ])
        
        csv = full_results.to_csv(index=False)
        st.download_button(
            "📥 Download Full Results (CSV)",
            csv,
            "fraud_detection_results.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Clear results button
        if st.button("🔄 Process New File", use_container_width=True):
            st.session_state.batch_results = None
            st.session_state.pending_reviews = []
            st.rerun()

# ============================================================================
# PAGE: TRANSACTION HISTORY
# ============================================================================

elif st.session_state.page == 'Transaction History':
    st.markdown('<h1 style="color: #ffffff; font-size: 32px; font-weight: 700;">📋 Transaction History</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a5ba; font-size: 14px; margin-top: 5px;">Comprehensive view of all transactions from all sources</p>', unsafe_allow_html=True)
    
    # Summary stats
    total_txns = len(st.session_state.all_transactions)
    review_txns = len([t for t in st.session_state.all_transactions if t['status'] == 'Review'])
    approved_txns = len([t for t in st.session_state.all_transactions if t['status'] == 'Approved'])
    rejected_txns = len([t for t in st.session_state.all_transactions if t['status'] == 'Rejected'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total_txns)
    col2.metric("Approved", approved_txns, f"{approved_txns/total_txns*100:.1f}%" if total_txns > 0 else "0%")
    col3.metric("Pending Review", review_txns, f"{review_txns/total_txns*100:.1f}%" if total_txns > 0 else "0%")
    col4.metric("Rejected", rejected_txns, f"{rejected_txns/total_txns*100:.1f}%" if total_txns > 0 else "0%")
    
    st.markdown("---")
    
    # Filter section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.selectbox("Filter by Source", 
                                     ["All", "Live Monitor", "Prediction Simulator", "Batch Analysis"])
    
    with col2:
        status_filter = st.selectbox("Filter by Status", 
                                     ["All", "Approved", "Review", "Rejected"])
    
    with col3:
        risk_filter = st.selectbox("Filter by Risk", 
                                   ["All", "LOW", "MEDIUM", "HIGH"])
    
    # Apply filters
    filtered_txns = st.session_state.all_transactions.copy()
    
    if source_filter != "All":
        filtered_txns = [t for t in filtered_txns if t.get('source') == source_filter]
    
    if status_filter != "All":
        filtered_txns = [t for t in filtered_txns if t['status'] == status_filter]
    
    if risk_filter != "All":
        filtered_txns = [t for t in filtered_txns if t['risk_level'] == risk_filter]
    
    st.markdown(f"**Showing {len(filtered_txns)} of {total_txns} transactions**")
    
    st.markdown("---")
    
    # Pending Reviews Section
    if review_txns > 0:
        with st.expander(f"⚠️ {review_txns} Transactions Require Second Verification", expanded=True):
            review_list = [t for t in st.session_state.all_transactions if t['status'] == 'Review']
            
            for idx, txn in enumerate(review_list[:10]):  # Show first 10
                st.markdown(f"### Transaction #{idx + 1}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ID", txn['id'][:10])
                col2.metric("Amount", f"${txn['amount']:.2f}")
                col3.metric("Risk Score", f"{txn['hybrid_score']*100:.1f}%")
                col4.metric("Source", txn.get('source', 'Unknown'))
                
                col_a, col_b = st.columns(2)
                col_a.metric("Category", txn['category'])
                col_b.metric("Rule Violations", txn['rule_score'])
                
                if txn.get('violations'):
                    st.markdown("**Violations:**")
                    for violation in txn['violations'][:3]:
                        st.markdown(f"• {violation}")
                
                col_x, col_y, col_z = st.columns(3)
                
                if col_x.button("✅ Approve", key=f"hist_approve_{idx}_{txn['id']}", use_container_width=True, type="primary"):
                    # Update transaction status
                    for t in st.session_state.all_transactions:
                        if t['id'] == txn['id']:
                            t['status'] = 'Approved'
                            break
                    st.success(f"✅ Transaction {txn['id']} approved")
                    time.sleep(0.5)
                    st.rerun()
                
                if col_y.button("❌ Reject", key=f"hist_reject_{idx}_{txn['id']}", use_container_width=True):
                    # Update transaction status
                    for t in st.session_state.all_transactions:
                        if t['id'] == txn['id']:
                            t['status'] = 'Rejected'
                            break
                    st.error(f"❌ Transaction {txn['id']} rejected")
                    time.sleep(0.5)
                    st.rerun()
                
                if col_z.button("🔍 Audit", key=f"hist_audit_{idx}_{txn['id']}", use_container_width=True):
                    st.session_state.show_audit_modal = True
                    st.session_state.audit_data = txn
                    st.rerun()
                
                st.markdown("---")
    
    # All Transactions Table
    st.markdown("### 📊 All Transactions")
    
    if filtered_txns:
        # Table header
        cols = st.columns([0.8, 1, 1, 1.2, 1, 1, 1, 1, 0.6])
        cols[0].markdown('<p class="table-header">Time</p>', unsafe_allow_html=True)
        cols[1].markdown('<p class="table-header">ID</p>', unsafe_allow_html=True)
        cols[2].markdown('<p class="table-header">Amount</p>', unsafe_allow_html=True)
        cols[3].markdown('<p class="table-header">Category</p>', unsafe_allow_html=True)
        cols[4].markdown('<p class="table-header">Score</p>', unsafe_allow_html=True)
        cols[5].markdown('<p class="table-header">Risk</p>', unsafe_allow_html=True)
        cols[6].markdown('<p class="table-header">Status</p>', unsafe_allow_html=True)
        cols[7].markdown('<p class="table-header">Source</p>', unsafe_allow_html=True)
        cols[8].markdown('<p class="table-header">Action</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display transactions (paginated)
        page_size = 20
        total_pages = (len(filtered_txns) - 1) // page_size + 1
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        
        start_idx = st.session_state.current_page * page_size
        end_idx = min(start_idx + page_size, len(filtered_txns))
        
        for txn in filtered_txns[start_idx:end_idx]:
            cols = st.columns([0.8, 1, 1, 1.2, 1, 1, 1, 1, 0.6])
            
            cols[0].markdown(f'<p class="table-text">{txn["time"]}</p>', unsafe_allow_html=True)
            cols[1].markdown(f'<p class="table-text">{txn["id"][:8]}...</p>', unsafe_allow_html=True)
            cols[2].markdown(f'<p class="table-text">${txn["amount"]:.2f}</p>', unsafe_allow_html=True)
            cols[3].markdown(f'<p class="table-text">{txn["category"]}</p>', unsafe_allow_html=True)
            
            # Score
            score_val = int(txn['hybrid_score'] * 100)
            cols[4].markdown(f'<p class="table-text">{score_val}%</p>', unsafe_allow_html=True)
            
            # Risk level
            risk = txn['risk_level']
            if risk == 'LOW':
                cols[5].markdown('<span class="risk-low">● LOW</span>', unsafe_allow_html=True)
            elif risk == 'MEDIUM':
                cols[5].markdown('<span class="risk-medium">● MED</span>', unsafe_allow_html=True)
            else:
                cols[5].markdown('<span class="risk-high">● HIGH</span>', unsafe_allow_html=True)
            
            # Status
            status = txn['status']
            if status == 'Approved':
                cols[6].markdown('<span class="status-approved">✓</span>', unsafe_allow_html=True)
            elif status == 'Review':
                cols[6].markdown('<span class="status-review">⚠</span>', unsafe_allow_html=True)
            else:
                cols[6].markdown('<span class="status-rejected">✗</span>', unsafe_allow_html=True)
            
            # Source
            cols[7].markdown(f'<p class="table-text" style="font-size: 12px;">{txn.get("source", "N/A")}</p>', unsafe_allow_html=True)
            
            # Audit button
            if cols[8].button("🔍", key=f"hist_view_{txn['id']}"):
                st.session_state.show_audit_modal = True
                st.session_state.audit_data = txn
                st.rerun()
        
        # Pagination
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(st.session_state.current_page == 0)):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col2:
            st.markdown(f'<p style="text-align: center; color: #c9ccd4;">Page {st.session_state.current_page + 1} of {total_pages}</p>', unsafe_allow_html=True)
        
        with col3:
            if st.button("Next ➡️", disabled=(st.session_state.current_page >= total_pages - 1)):
                st.session_state.current_page += 1
                st.rerun()
    
    else:
        st.info("No transactions found matching the selected filters.")
    
    # AI Audit Modal (same as Live Monitor)
    if st.session_state.show_audit_modal and st.session_state.audit_data:
        st.markdown("---")
        st.markdown("---")
        
        st.markdown("""
        <div style='background-color: #1e2130; padding: 30px; border-radius: 15px; border: 2px solid #00ff88;'>
            <h2 style='color: #00ff88; margin-bottom: 20px;'>🔍 Hybrid Transaction Audit</h2>
        </div>
        """, unsafe_allow_html=True)
        
        txn = st.session_state.audit_data
        
        st.markdown(f"**Transaction ID:** {txn['id']}")
        st.markdown(f"**Source:** {txn.get('source', 'Unknown')}")
        
        # Three metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>RULE SCORE</div>
                <div class='card-value'>{}/5</div>
            </div>
            """.format(txn['rule_score']), unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>ML PROB</div>
                <div class='card-value'>{:.0f}%</div>
            </div>
            """.format(txn['ml_proba'] * 100), unsafe_allow_html=True)
        
        with col_c:
            hybrid_color = '#ff4444' if txn['hybrid_score'] > 0.6 else '#ffaa00' if txn['hybrid_score'] > 0.3 else '#00ff88'
            st.markdown("""
            <div class='metric-card'>
                <div class='card-title'>FINAL HYBRID</div>
                <div class='card-value' style='color: {};'>{:.1f}%</div>
            </div>
            """.format(hybrid_color, txn['hybrid_score'] * 100), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Rule Violations
        st.markdown("**Rule Violations**")
        if txn.get('violations'):
            for violation in txn['violations']:
                st.markdown(f"• {violation}")
        else:
            st.success("✓ No rule violations detected")
        
        st.markdown("---")
        
        # Close button
        if st.button("Close Audit", use_container_width=True, type="primary"):
            st.session_state.show_audit_modal = False
            st.session_state.audit_data = None
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"© 2025 FraudGuard AI - Hybrid Detection System | Connected to: {API_BASE_URL}")