import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# =========================================================
# 1. UI Configuration
# =========================================================
st.set_page_config(
    layout="wide", 
    page_title="Pro Quant Terminal",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = 20
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = 60

# =========================================================
# 2. Advanced CSS Styling (Dark Professional Theme)
# =========================================================
st.markdown("""
    <style>
    /* Global App Style */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    /* Input Fields (Dark Mode) */
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input {
        background-color: #1C2128 !important;
        color: white !important;
        border: 1px solid #30363D !important;
        border-radius: 6px;
    }
    div[data-testid="stNumberInput"] > div {
        background-color: #1C2128 !important;
        border: none !important;
    }
    
    /* Metrics (KPI Cards) */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #30363D;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161B22;
        border-radius: 8px 8px 0px 0px;
        color: #8B949E;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        color: #58A6FF;
        border-top: 2px solid #58A6FF;
    }
    
    /* Primary Button */
    .stButton > button.primary {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Remove Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 3. Helper Functions
# =========================================================

def calculate_technical_indicators(df, short_w, long_w):
    # MA
    df['SMA_Short'] = df['Close'].rolling(window=short_w).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_w).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = rolling_mean + (2 * rolling_std)
    df['BB_Lower'] = rolling_mean - (2 * rolling_std)
    
    return df

def fetch_stock_data(symbol, period="2y"):
    symbol = symbol.upper().strip()
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Auto-fix for Taiwan stocks
    if df.empty and (symbol.endswith('.TW') or symbol.endswith('.TWO')):
        alt = symbol.replace('.TW', '.TWO') if symbol.endswith('.TW') else symbol.replace('.TWO', '.TW')
        df_alt = yf.download(alt, period=period, interval="1d", progress=False)
        if not df_alt.empty:
            if isinstance(df_alt.columns, pd.MultiIndex):
                df_alt.columns = df_alt.columns.get_level_values(0)
            return df_alt, alt
            
    return df, symbol

# FIX: Define a callback function for reliable session state updates
def update_optimizer_params(best_short, best_long):
    """Callback function to update session state parameters safely."""
    # Ensure conversion to standard Python int before setting state
    st.session_state['short_window'] = int(best_short)
    st.session_state['long_window'] = int(best_long)

def run_optimizer(symbol):
    short_opts = [5, 10, 20]
    long_opts = [20, 50, 60]
    results = []
    
    df, _ = fetch_stock_data(symbol, "1y")
    if df.empty: return None, "No Data"
    
    for s in short_opts:
        for l in long_opts:
            if s >= l: continue
            d = df.copy()
            d['S'] = d['Close'].rolling(s).mean()
            d['L'] = d['Close'].rolling(l).mean()
            d['Sig'] = np.where(d['S'] > d['L'], 1, 0)
            d['Ret'] = d['Sig'].shift(1) * d['Close'].pct_change()
            cum_ret = (d['Ret'] + 1).cumprod().iloc[-1] - 1
            results.append({'Short': s, 'Long': l, 'Return': cum_ret * 100})
            
    return pd.DataFrame(results).sort_values('Return', ascending=False), None

# =========================================================
# 4. Sidebar Controls
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", value="2330.TW", help="e.g. AAPL, TSLA, 2330.TW")
    capital = st.number_input("Capital ($)", value=10000.0)
    
    st.divider()
    
    st.subheader("Strategy (MA)")
    # Widget keys link directly to session state variables
    short_ma = st.number_input("Short MA", key="short_window", min_value=1)
    long_ma = st.number_input("Long MA", key="long_window", min_value=5)
    
    st.divider()
    
    with st.expander("🚀 AI Optimizer"):
        # The optimizer button is now defined here, but the result processing is separate
        run_optimize_button = st.button("Run Auto-Optimize")

# --- Optimization Processing Block ---
if run_optimize_button:
    with st.spinner("Optimizing..."):
        res, err = run_optimizer(ticker)
        
        if res is not None:
            best = res.iloc[0]
            best_short = best['Short']
            best_long = best['Long']
            best_ret = best['Return']
            
            # Use the callback function to update the state safely
            update_optimizer_params(best_short, best_long)

            # Display results in the expander (after state is updated)
            st.success(f"Best: {int(best_short)}/{int(best_long)} (Ret: {best_ret:.1f}%)")
            # st.dataframe(res.head(5).style.format("{:.2f}"))
            st.toast("Optimal parameters applied!", icon="✅")
            st.experimental_rerun() # Trigger rerun to update the main MA widgets
        elif err:
            st.error(err)


# =========================================================
# 5. Main Content
# =========================================================

# A. Header Section (Always Visible)
df, valid_ticker = fetch_stock_data(ticker)

if df.empty:
    st.error(f"⚠️ No data found for {ticker}")
    st.stop()

# Basic Calc
df = calculate_technical_indicators(df, short_ma, long_ma)
curr = df.iloc[-1]
prev = df.iloc[-2]
change = curr['Close'] - prev['Close']
pct_change = (change / prev['Close']) * 100
color = "green" if change >= 0 else "red"

col1, col2, col3 = st.columns([2, 4, 2])
with col1:
    st.title(valid_ticker)
    st.caption("Real-time Market Data")
with col3:
    st.metric("Current Price", f"${curr['Close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")

st.divider()

# B. Tabbed Layout
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🕯️ Technical Analysis", "🔮 AI Forecast"])

# --- TAB 1: DASHBOARD (Overview) ---
with tab1:
    # Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    # Strategy Return Calc
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    df['Pos'] = df['Signal'].diff()
    
    # Simple Backtest
    pos = 0; cash = capital
    for _, row in df.iterrows():
        if row['Pos'] == 1 and pos == 0:
            pos = cash / row['Close']; cash = 0
        elif row['Pos'] == -1 and pos > 0:
            cash = pos * row['Close']; pos = 0
    equity = cash if pos == 0 else pos * curr['Close']
    ret = (equity - capital) / capital * 100
    
    m1.metric("Strategy Return", f"{ret:.2f}%", delta_color="normal")
    m2.metric("RSI (14)", f"{curr['RSI']:.1f}")
    m3.metric("MACD", f"{curr['MACD']:.2f}")
    m4.metric("MA Trend", "Bullish" if curr['SMA_Short'] > curr['SMA_Long'] else "Bearish")
    
    # Simple Line Chart for quick view
    st.subheader("Price Trend")
    st.line_chart(df[['Close', 'SMA_Short', 'SMA_Long']].tail(180), color=["#FFFFFF", "#FFA500", "#00BFFF"])

# --- TAB 2: TECHNICALS (Detailed Chart) ---
with tab2:
    st.caption("Interactive Candlestick Chart with Buy/Sell Signals")
    
    # mplfinance plot
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True, facecolor='#0E1117', figcolor='#0E1117', gridcolor='#30363D')
    
    plots = [
        mpf.make_addplot(df['SMA_Short'].tail(120), color='#FFA726', width=1.5),
        mpf.make_addplot(df['SMA_Long'].tail(120), color='#29B6F6', width=1.5),
        mpf.make_addplot(df['BB_Upper'].tail(120), color='#80DEEA', alpha=0.3),
        mpf.make_addplot(df['BB_Lower'].tail(120), color='#80DEEA', alpha=0.3)
    ]
    
    # Signals
    subset = df.tail(120)
    buy_sig = np.where(subset['Pos'] == 1, subset['Low']*0.98, np.nan)
    sell_sig = np.where(subset['Pos'] == -1, subset['High']*1.02, np.nan)
    
    plots.append(mpf.make_addplot(buy_sig, type='scatter', markersize=80, marker='^', color='#00E676'))
    plots.append(mpf.make_addplot(sell_sig, type='scatter', markersize=80, marker='v', color='#FF1744'))
    
    fig, axlist = mpf.plot(subset, type='candle', style=s, addplot=plots, returnfig=True, figsize=(12, 6), volume=True)
    
    # Price Annotations
    ax = axlist[0]
    for i, (idx, row) in enumerate(subset.iterrows()):
        if row['Pos'] == 1:
            ax.annotate(f"{row['Close']:.0f}", xy=(i, row['Low']*0.98), xytext=(0, -15), textcoords='offset points', ha='center', color='#00E676', fontsize=8)
        elif row['Pos'] == -1:
            ax.annotate(f"{row['Close']:.0f}", xy=(i, row['High']*1.02), xytext=(0, 15), textcoords='offset points', ha='center', color='#FF1744', fontsize=8)
            
    st.pyplot(fig)

# --- TAB 3: FORECAST (AI Prediction) ---
with tab3:
    st.subheader("3-Day Linear Regression Forecast")
    
    # Prediction Logic
    recent = df['Close'].tail(15)
    x = np.arange(len(recent))
    y = recent.values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    future_prices = p(np.arange(len(recent), len(recent)+3))
    
    c1, c2, c3 = st.columns(3)
    days = ["Tomorrow", "+2 Days", "+3 Days"]
    
    for i, price in enumerate(future_prices):
        with [c1, c2, c3][i]:
            st.markdown(f"""
            <div style="background-color: #161B22; padding: 20px; border-radius: 8px; border: 1px solid #30363D; text-align: center;">
                <div style="color: #8B949E; font-size: 14px;">{days[i]}</div>
                <div style="color: #FAFAFA; font-size: 24px; font-weight: bold;">${price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
