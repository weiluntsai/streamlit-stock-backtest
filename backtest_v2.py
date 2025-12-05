import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# =========================================================
# UI Configuration (Must be the first Streamlit command)
# =========================================================
st.set_page_config(
    layout="wide", 
    page_title="Quant Dashboard Pro",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = 20
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = 30

# =========================================================
# Custom CSS (Dribbble Style with Fixes)
# =========================================================
st.markdown("""
    <style>
    /* 1. Global Background and Font */
    .stApp {
        background-color: #0E1117; 
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* 2. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* 3. Metric Card Design (Border Radius Reduced) */
    div[data-testid="stMetric"] {
        background-color: #1F242D; 
        padding: 15px;
        border-radius: 8px; /* Reduced from 12px */
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #58A6FF;
    }
    
    /* 4. Button Styling */
    /* Primary Button (Run Full Analysis) - Green */
    .stButton > button.primary {
        background: linear-gradient(45deg, #238636, #2EA043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button.primary:hover {
        box-shadow: 0 0 10px #2EA043;
    }

    /* Optimization Button (Non-Primary) - Navy Blue and White Text */
    .stButton > button:not(.primary) {
        background-color: #1A73E8; /* Navy Blue */
        color: white; /* White Text */
        border: 1px solid #1A73E8;
        border-radius: 6px; 
    }
    .stButton > button:not(.primary):hover {
        background-color: #3B82F6;
        border-color: #3B82F6;
    }

    
    /* 5. DataFrame Styling (Border Radius Reduced) */
    [data-testid="stDataFrame"] {
        background-color: #161B22;
        border-radius: 8px; /* Reduced from 10px */
        padding: 10px;
    }
    
    /* 6. Titles and Text Colors */
    h1, h2, h3 {
        color: #E6EDF3 !important;
        font-weight: 600;
    }
    p, label {
        color: #8B949E !important;
    }
    
    /* 7. Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 8. Expander Style (Border Radius Reduced) */
    .streamlit-expanderHeader {
        background-color: #1F242D;
        border-radius: 8px; /* Reduced from 8px, ensuring consistency */
        color: #E6EDF3;
    }
    
    /* === Input Field Styling for Dark UI === */
    
    div[data-testid="stTextInput"] input {
        background-color: #161B22 !important;
        color: #FAFAFA !important; 
        border: 1px solid #30363D !important;
    }
    
    div[data-testid="stNumberInput"] > div {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px;
    }
    
    div[data-testid="stNumberInput"] input {
        background-color: #161B22 !important;
        color: #FAFAFA !important;
    }
    
    div[data-testid="stSlider"] label {
        color: #FAFAFA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# Helper Functions (Calculation Logic)
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / rolling_mean
    return df

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

def run_optimization(stock_symbol):
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50, 60]
    results = []

    try:
        df_raw = yf.download(stock_symbol, period="6mo", interval="1d", progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        
        if df_raw.empty:
            return None, "No data found."

        for short_w in short_windows:
            for long_w in long_windows:
                if short_w >= long_w:
                    continue
                
                df = df_raw.copy()
                df['Short'] = df['Close'].rolling(window=short_w).mean()
                df['Long'] = df['Close'].rolling(window=long_w).mean()
                df['Signal'] = np.where(df['Short'] > df['Long'], 1, 0)
                
                df['Daily_Return'] = df['Close'].pct_change()
                df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
                
                total_return = (df['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
                total_return_pct = total_return * 100
                
                results.append({
                    'Short': short_w,
                    'Long': long_w,
                    'Return': total_return_pct
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Return', ascending=False)
        return results_df, None
    except Exception as e:
        return None, str(e)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## ⚙️ Configuration")
sidebar_stock = st.sidebar.text_input("Ticker", value="TSLA")
days_to_test = st.sidebar.slider("Lookback Days", 30, 365, 90)
initial_capital = st.sidebar.number_input("Capital ($)", value=10000.0)

st.sidebar.markdown("### Strategy (MA)")
# Connect number inputs to session state
short_window = st.sidebar.number_input("Short Term", key='short_window', min_value=1)
long_window = st.sidebar.number_input("Long Term", key='long_window', min_value=2)

st.sidebar.markdown("### Indicators")
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
add_indicator = st.sidebar.selectbox("Sub-Chart", ["None", "MACD", "RSI"], index=1)

# =========================================================
# Main Layout
# =========================================================

# 1. Header & Optimization (Top Section)
st.title(f"📊 {sidebar_stock} Analytics")

with st.expander("🚀 Strategy Optimizer", expanded=False):
    st.markdown("Run a 6-month historical simulation to find the best parameters.")
    
    # Check if the optimization button is pressed
    if st.button("Auto-Optimize Parameters"):
        with st.spinner("Simulating strategies..."):
            results_df, error = run_optimization(sidebar_stock)
        
        if error:
            st.error(error)
        elif results_df is not None:
            best_short = results_df.iloc[0]['Short']
            best_long = results_df.iloc[0]['Long']
            best_ret = results_df.iloc[0]['Return']
            
            # Update State - FIX: Convert float to int before updating session state
            st.session_state['short_window'] = int(best_short)
            st.session_state['long_window'] = int(best_long)
            
            st.success(f"Best Found: {best_short}/{best_long} (Return: {best_ret:.2f}%)")
            st.dataframe(results_df.head(5).style.format("{:.2f}"))

# 2. Main Analysis Trigger
if st.button("Run Full Analysis", type="primary"):
    
    # Data Processing
    try:
        with st.spinner("Fetching market data..."):
            df_raw = yf.download(sidebar_stock, period="2y", interval="1d", progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            
            # Indicators
            df_raw['SMA_Short'] = df_raw['Close'].rolling(window=short_window).mean()
            df_raw['SMA_Long'] = df_raw['Close'].rolling(window=long_window).mean()
            df_raw['RSI'] = calculate_rsi(df_raw['Close'])
            df_raw = calculate_macd(df_raw)
            df_raw = calculate_bollinger_bands(df_raw)
            
            # Slice
            df = df_raw.tail(days_to_test).copy()
            
            # Backtest
            df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
            df['Position_Change'] = df['Signal'].diff()
            
            pos = 0; cash = initial_capital
            for _, row in df.iterrows():
                if row['Position_Change'] == 1 and pos == 0:
                    pos = cash / row['Close']; cash = 0
                elif row['Position_Change'] == -1 and pos > 0:
                    cash = pos * row['Close']; pos = 0
            final_val = cash if pos == 0 else pos * df.iloc[-1]['Close']
            roi = (final_val - initial_capital) / initial_capital * 100
            
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

    # 3. Dashboard Grid (Cards)
    st.markdown("### 📈 Market Status")
    
    curr = df_raw.iloc[-1]
    prev = df_raw.iloc[-2]
    
    # Determine Status Colors
    trend = "Bullish" if curr['SMA_Short'] > curr['SMA_Long'] else "Bearish"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${curr['Close']:.2f}", f"{curr['Close']-prev['Close']:.2f}")
    col2.metric("Trend", trend, f"{short_window}/{long_window} MA")
    col3.metric("RSI (14)", f"{curr['RSI']:.1f}", "Overbought" if curr['RSI']>70 else "Normal")
    col4.metric("Strategy ROI", f"{roi:.2f}%", f"Eq: ${final_val:.0f}")

    # 4. Professional Chart (Dark Mode)
    st.markdown("### 🕯️ Technical Chart")
    
    # Customizing mplfinance to match Dribbble dark theme
    market_colors = mpf.make_marketcolors(up='#26A69A', down='#EF5350', inherit=True)
    dribbble_style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle=':', y_on_right=True, facecolor='#0E1117', figcolor='#0E1117', gridcolor='#30363D')

    plots = []
    plots.append(mpf.make_addplot(df['SMA_Short'], color='#FFA726', width=1.5))
    plots.append(mpf.make_addplot(df['SMA_Long'], color='#29B6F6', width=1.5))
    
    if show_bollinger:
        plots.append(mpf.make_addplot(df['BB_Upper'], color='#80DEEA', alpha=0.3))
        plots.append(mpf.make_addplot(df['BB_Lower'], color='#80DEEA', alpha=0.3))
        
    if add_indicator == "MACD":
        plots.append(mpf.make_addplot(df['MACD_Hist'], type='bar', panel=1, color='dimgray', alpha=0.5, ylabel='MACD'))
        plots.append(mpf.make_addplot(df['MACD_Line'], panel=1, color='#AB47BC'))
        plots.append(mpf.make_addplot(df['MACD_Signal'], panel=1, color='#29B6F6'))
    elif add_indicator == "RSI":
        plots.append(mpf.make_addplot(df['RSI'], panel=1, color='#AB47BC', ylabel='RSI', ylim=(0,100)))
        plots.append(mpf.make_addplot([70]*len(df), panel=1, color='#EF5350', linestyle='--', width=0.8))
        plots.append(mpf.make_addplot([30]*len(df), panel=1, color='#66BB6A', linestyle='--', width=0.8))

    # Buy/Sell Arrows
    buy_sig = np.where(df['Position_Change'] == 1, df['Low']*0.98, np.nan)
    sell_sig = np.where(df['Position_Change'] == -1, df['High']*1.02, np.nan)
    if not np.all(np.isnan(buy_sig)):
        plots.append(mpf.make_addplot(buy_sig, type='scatter', markersize=80, marker='^', color='#00E676'))
    if not np.all(np.isnan(sell_sig)):
        plots.append(mpf.make_addplot(sell_sig, type='scatter', markersize=80, marker='v', color='#FF1744'))

    fig, axlist = mpf.plot(df, type='candle', style=dribbble_style, 
                       volume=(add_indicator=="None"), 
                       addplot=plots, returnfig=True, figsize=(12, 8),
                       panel_ratios=(2, 1) if add_indicator != "None" else (1,),
                       fontscale=0.8)
    
    # Text Annotations (White text for dark mode)
    ax_main = axlist[0]
    ax_main.yaxis.label.set_color('#8B949E')
    ax_main.xaxis.label.set_color('#8B949E')
    ax_main.tick_params(axis='x', colors='#8B949E')
    ax_main.tick_params(axis='y', colors='#8B949E')

    for i, (index, row) in enumerate(df.iterrows()):
        if row['Position_Change'] == 1:
            ax_main.annotate(f"{row['Close']:.0f}", xy=(i, row['Low']*0.98), 
                             xytext=(0, -15), textcoords='offset points', ha='center', color='#00E676', fontsize=8)
        elif row['Position_Change'] == -1:
            ax_main.annotate(f"{row['Close']:.0f}", xy=(i, row['High']*1.02), 
                             xytext=(0, 15), textcoords='offset points', ha='center', color='#FF1744', fontsize=8)

    st.pyplot(fig)

    # 5. Future Prediction (Footer Cards)
    st.markdown("### 🔮 AI Forecast (3-Day)")
    df_pred = predict_future_ma(df_raw, short_window, long_window)
    
    f_cols = st.columns(3)
    for i, (idx, row) in enumerate(df_pred.iterrows()):
        signal_icon = "🟢" if row['SMA_Short'] > row['SMA_Long'] else "🔴"
        with f_cols[i]:
            st.markdown(f"""
            <div style="background-color: #1F242D; padding: 10px; border-radius: 8px; border: 1px solid #30363D; text-align: center;">
                <div style="color: #8B949E; font-size: 12px;">{idx.strftime('%b %d')}</div>
                <div style="color: #FAFAFA; font-size: 18px; font-weight: bold;">${row['Close']:.2f}</div>
                <div style="color: #E6EDF3; font-size: 14px; margin-top: 5px;">{signal_icon}</div>
            </div>
            """, unsafe_allow_html=True)
