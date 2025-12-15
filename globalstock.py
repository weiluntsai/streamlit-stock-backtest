import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# =========================================================
# 1. UI Configuration & Session State
# =========================================================
st.set_page_config(
    layout="wide", 
    page_title="Pro Quant Terminal",
    initial_sidebar_state="expanded"
)

# Initialize session state (default MA settings)
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = 20
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = 60

# =========================================================
# 2. Advanced CSS Styling (Dribbble Dark Theme)
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
    
    /* Auto-Optimize Button (Non-Primary) */
    .stButton > button:not(.primary) {
        background-color: #1A73E8; /* Navy Blue */
        color: white; /* White Text */
        border: 1px solid #1A73E8;
        border-radius: 6px; 
        font-weight: 600;
    }
    
    /* Remove Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Expander Style */
    .streamlit-expanderHeader {
        background-color: #1F242D;
        border-radius: 8px; 
        color: #E6EDF3;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 3. Helper Functions (Indicators)
# =========================================================

def calculate_rsi(series, period=14):
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence (MACD)"""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Bollinger Bands (BB)"""
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / rolling_mean
    return df

def calculate_kdj(df, period=9):
    """Stochastic Oscillator (KDJ)"""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def calculate_obv(df):
    """On-Balance Volume (OBV)"""
    obv_change = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 
                 np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = pd.Series(obv_change, index=df.index).cumsum()
    return df

def calculate_technical_indicators(df, short_w, long_w):
    # MA
    df['SMA_Short'] = df['Close'].rolling(window=short_w).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_w).mean()
    # Apply all other indicators
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_obv(df)
    df['RSI'] = calculate_rsi(df['Close'])
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

def run_optimizer(symbol):
    short_opts = [5, 10, 20]
    long_opts = [20, 50, 60]
    results = []
    
    # Use 1 year of data for optimization stability
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

def update_optimizer_params(best_short, best_long):
    """Callback function to update session state parameters safely."""
    st.session_state['short_window'] = int(best_short)
    st.session_state['long_window'] = int(best_long)

def predict_future_ma(df_historical, short_window, long_window, days_to_predict=3):
    recent_data = df_historical['Close'].tail(15)
    x = np.arange(len(recent_data))
    y = recent_data.values
    z = np.polyfit(x, y, 1) 
    p = np.poly1d(z)
    
    future_x = np.arange(len(recent_data), len(recent_data) + days_to_predict)
    future_prices = p(future_x)
    
    last_date = df_historical.index[-1]
    future_dates = []
    current_date = last_date
    while len(future_dates) < days_to_predict:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5: 
            future_dates.append(current_date)
            
    df_future = pd.DataFrame(index=future_dates)
    df_future['Close'] = future_prices
    
    df_combined = pd.concat([df_historical[['Close']], df_future[['Close']]])
    df_combined['SMA_Short'] = df_combined['Close'].rolling(window=short_window).mean()
    df_combined['SMA_Long'] = df_combined['Close'].rolling(window=long_window).mean()
    
    return df_combined.tail(days_to_predict)

# =========================================================
# 4. Sidebar Controls
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", value="2330.TW", help="e.g. AAPL, TSLA, 2330.TW or 8069.TWO")
    capital = st.number_input("Capital ($)", value=10000.0)
    
    st.divider()
    
    st.subheader("Strategy (MA)")
    short_ma = st.number_input("Short MA", key="short_window", min_value=1)
    long_ma = st.number_input("Long MA", key="long_window", min_value=5)
    
    st.markdown("##### 🛡️ Risk Filter")
    enable_rsi_filter = st.checkbox("Enable RSI Filter (>50 Buy, <50 Sell)", value=False, help="Reduces noise in choppy markets.")

    st.divider()
    
    st.subheader("Chart Options")
    show_bollinger = st.checkbox("Bollinger Bands (BB)", value=True)
    add_indicator = st.selectbox("Sub-Chart Panel", ["None", "MACD", "RSI", "KDJ", "OBV"], index=1)
    
    st.divider()
    
    with st.expander("🚀 AI Optimizer"):
        if st.button("Run Auto-Optimize"):
            with st.spinner("Simulating strategies..."):
                res, err = run_optimizer(ticker)
                
                if res is not None:
                    best = res.iloc[0]
                    best_short = best['Short']
                    best_long = best['Long']
                    
                    update_optimizer_params(best_short, best_long)
                    st.toast(f"Optimal: {int(best_short)}/{int(best_long)} applied!", icon="✅")
                    st.experimental_rerun()
                elif err:
                    st.error(f"Optimizer Error: {err}")


# =========================================================
# 5. Main Content (The Analysis Pipeline)
# =========================================================

if st.button("Run Analysis", type="primary"):
    
    if short_ma >= long_ma:
        st.error("MA Error: Short MA must be less than Long MA.")
        st.stop()
        
    with st.spinner(f"Fetching data for {ticker}..."):
        df_raw, valid_ticker = fetch_stock_data(ticker)
    
    if df_raw.empty or len(df_raw) < long_ma:
        st.error(f"⚠️ Not enough historical data (only {len(df_raw)} days). Need at least {long_ma} days.")
        st.stop()
        
    # Calculate all indicators
    df_raw = calculate_technical_indicators(df_raw, short_ma, long_ma)
    
    # Slice for Backtest period
    df = df_raw.tail(120).copy()
    
    # --- Advanced Signal Generation (With RSI Filter) ---
    # 1. Base Signal: MA Crossover
    df['MA_Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)
    
    # 2. Apply RSI Filter if enabled
    if enable_rsi_filter:
        # Buy only if MA Bullish AND RSI > 50 (Momentum is strong)
        # Sell only if MA Bearish AND RSI < 50 (Momentum is weak)
        # Otherwise, hold previous state (0 means do nothing/hold)
        conditions = [
            (df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] > 50), # Strong Buy
            (df['SMA_Short'] < df['SMA_Long']) & (df['RSI'] < 50)  # Strong Sell
        ]
        choices = [1, -1]
        df['Signal'] = np.select(conditions, choices, default=0)
        
        # Forward fill zeros to maintain position until signal changes
        # (Optional: depends if you want to exit immediately or hold)
        # Here we use a stricter logic: 0 means no *new* signal, but we calculate crossovers differently
    else:
        df['Signal'] = df['MA_Signal']

    # 3. Detect Entry/Exit points
    # We want to detect when the Signal *changes* to 1 or -1
    df['Pos'] = df['Signal'].diff()
    
    # --- Backtest Engine ---
    pos = 0; cash = capital
    for _, row in df.iterrows():
        # Logic: 
        # If Signal becomes 1 (Buy) and we are not in position -> BUY
        # If Signal becomes -1 (Sell) and we are in position -> SELL
        
        if row['Signal'] == 1 and pos == 0:
            pos = cash / row['Close']; cash = 0
        elif row['Signal'] == -1 and pos > 0:
            cash = pos * row['Close']; pos = 0
            
    curr = df_raw.iloc[-1]
    prev = df_raw.iloc[-2]
    equity = cash if pos == 0 else pos * curr['Close']
    ret = (equity - capital) / capital * 100
    
    buy_hold_roi = (curr['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close'] * 100
    
    # ---------------------------------------------------------
    # A. Header Section
    # ---------------------------------------------------------
    st.title(valid_ticker)
    
    st.divider()
    
    # ---------------------------------------------------------
    # B. Tabbed Layout
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🕯️ Technical Analysis", "🔮 AI Forecast"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.subheader("Performance and Key Indicators")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy Return", f"{ret:.2f}%", delta_color="normal")
        m2.metric("Buy & Hold Return", f"{buy_hold_roi:.2f}%", delta_color="normal")
        m3.metric("Final Equity", f"${equity:.0f}", delta_color="normal")
        m4.metric("MA Trend", "Bullish" if curr['SMA_Short'] > curr['SMA_Long'] else "Bearish")
        
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", f"{curr['RSI']:.1f}")
        c2.metric("MACD Hist", f"{curr['MACD_Hist']:.2f}")
        c3.metric("BB Width", f"{curr['BB_Width']:.2f}")
        c4.metric("KDJ K", f"{curr['K']:.1f}")
        
        st.subheader("Price Trend (Last 180 Days)")
        st.line_chart(df_raw[['Close', 'SMA_Short', 'SMA_Long']].tail(180), color=["#FFFFFF", "#FFA500", "#00BFFF"])

    # --- TAB 2: TECHNICALS ---
    with tab2:
        st.caption(f"Interactive Candlestick Chart for {valid_ticker} (Last 120 Days)")
        
        mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True, facecolor='#0E1117', figcolor='#0E1117', gridcolor='#30363D')
        
        plots = [
            mpf.make_addplot(df['SMA_Short'], color='#FFA726', width=1.5),
            mpf.make_addplot(df['SMA_Long'], color='#29B6F6', width=1.5),
        ]
        
        if show_bollinger:
            plots.append(mpf.make_addplot(df['BB_Upper'], color='#80DEEA', alpha=0.3, width=0.8))
            plots.append(mpf.make_addplot(df['BB_Lower'], color='#80DEEA', alpha=0.3, width=0.8))

        panel_data = []
        if add_indicator == "MACD":
            panel_data.extend([
                mpf.make_addplot(df['MACD'], panel=1, color='#AB47BC', ylabel='MACD'),
                mpf.make_addplot(df['Signal_Line'], panel=1, color='#29B6F6'),
                mpf.make_addplot(df['MACD_Hist'], type='bar', panel=1, color='dimgray', alpha=0.5)
            ])
        elif add_indicator == "RSI":
            panel_data.extend([
                mpf.make_addplot(df['RSI'], panel=1, color='#AB47BC', ylabel='RSI', ylim=(0, 100)),
                mpf.make_addplot([70]*len(df), panel=1, color='#EF5350', linestyle='--', width=0.8),
                mpf.make_addplot([30]*len(df), panel=1, color='#66BB6A', linestyle='--', width=0.8),
                mpf.make_addplot([50]*len(df), panel=1, color='gray', linestyle=':', width=0.5) # RSI 50 Midline
            ])
        elif add_indicator == "KDJ":
            panel_data.extend([
                mpf.make_addplot(df['K'], panel=1, color='orange', ylabel='KDJ'),
                mpf.make_addplot(df['D'], panel=1, color='blue'),
                mpf.make_addplot(df['J'], panel=1, color='purple')
            ])
        elif add_indicator == "OBV":
            panel_data.append(mpf.make_addplot(df['OBV'], panel=1, color='teal', ylabel='OBV'))

        plots.extend(panel_data)

        # Signals Visualization based on FILTERED signal
        # We look for where Signal changes (Pos != 0)
        # Note: 'Pos' is diff of Signal. 
        # If Signal goes 0 -> 1 (Buy), Pos=1. 
        # If Signal goes 1 -> 0 (Hold/Neutral), Pos=-1 (but we don't sell here in this simple logic, we just stop buying)
        # To simplify visualization: Mark BUY when Signal is 1 and we weren't before. Mark SELL when Signal is -1.
        
        # Let's refine the plot markers to match the loop logic
        buy_indices = []
        sell_indices = []
        temp_pos = 0
        for i, row in df.iterrows():
            if row['Signal'] == 1 and temp_pos == 0:
                buy_indices.append(i)
                temp_pos = 1
            elif row['Signal'] == -1 and temp_pos == 1:
                sell_indices.append(i)
                temp_pos = 0
        
        # Create NaN arrays for plotting
        buy_plot = pd.Series(np.nan, index=df.index)
        sell_plot = pd.Series(np.nan, index=df.index)
        buy_plot.loc[buy_indices] = df.loc[buy_indices]['Low'] * 0.98
        sell_plot.loc[sell_indices] = df.loc[sell_indices]['High'] * 1.02
        
        plots.append(mpf.make_addplot(buy_plot, type='scatter', markersize=80, marker='^', color='#00E676'))
        plots.append(mpf.make_addplot(sell_plot, type='scatter', markersize=80, marker='v', color='#FF1744'))
        
        fig, axlist = mpf.plot(df, type='candle', style=s, addplot=plots, returnfig=True, figsize=(12, 6), 
                               volume=(add_indicator == "None"),
                               panel_ratios=(2, 1) if add_indicator != "None" else (1,))
        
        # Annotations
        ax = axlist[0]
        for date in buy_indices:
            idx_loc = df.index.get_loc(date)
            price = df.loc[date]['Close']
            ax.annotate(f"{price:.0f}", xy=(idx_loc, df.loc[date]['Low']*0.98), xytext=(0, -15), textcoords='offset points', ha='center', color='#00E676', fontsize=8)
            
        for date in sell_indices:
            idx_loc = df.index.get_loc(date)
            price = df.loc[date]['Close']
            ax.annotate(f"{price:.0f}", xy=(idx_loc, df.loc[date]['High']*1.02), xytext=(0, 15), textcoords='offset points', ha='center', color='#FF1744', fontsize=8)
            
        st.pyplot(fig)

    # --- TAB 3: FORECAST ---
    with tab3:
        st.subheader("3-Day Linear Regression Forecast")
        df_predict = predict_future_ma(df_raw, short_ma, long_ma, days_to_predict=3)
        c1, c2, c3 = st.columns(3)
        for i, row in df_predict.iterrows():
            with [c1, c2, c3][i]:
                signal_icon = "🟢" if row['SMA_Short'] > row['SMA_Long'] else "🔴"
                st.markdown(f"""
                <div style="background-color: #161B22; padding: 20px; border-radius: 8px; border: 1px solid #30363D; text-align: center;">
                    <div style="color: #8B949E; font-size: 14px;">{row.name.strftime('%b %d')}</div>
                    <div style="color: #FAFAFA; font-size: 24px; font-weight: bold;">${row['Close']:.2f}</div>
                    <div style="color: #E6EDF3; font-size: 14px; margin-top: 5px;">Trend: {signal_icon}</div>
                </div>
                """, unsafe_allow_html=True)
