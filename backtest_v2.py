import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# Initialize session state for parameters if not present
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = 20
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = 30

# =========================================================
# Helper Function: Calculate RSI
# =========================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================================
# Helper Function: Future Prediction (Linear Regression)
# =========================================================
def predict_future_ma(df_historical, short_window, long_window, days_to_predict=3):
    # Use last 15 days for trend analysis
    recent_data = df_historical['Close'].tail(15)
    
    x = np.arange(len(recent_data))
    y = recent_data.values
    
    # Linear Regression
    z = np.polyfit(x, y, 1) 
    p = np.poly1d(z)
    
    # Predict future prices
    future_x = np.arange(len(recent_data), len(recent_data) + days_to_predict)
    future_prices = p(future_x)
    
    # Generate future dates (skip weekends)
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
# Optimization Logic
# =========================================================
def run_optimization(stock_symbol):
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50, 60]
    results = []

    try:
        df_raw = yf.download(stock_symbol, period="6mo", interval="1d", progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        
        if df_raw.empty:
            return None, "No data found for symbol."

        for short_w in short_windows:
            for long_w in long_windows:
                if short_w >= long_w:
                    continue
                
                df = df_raw.copy()
                df['Short'] = df['Close'].rolling(window=short_w).mean()
                df['Long'] = df['Close'].rolling(window=long_w).mean()
                
                df['Signal'] = 0
                df.loc[df['Short'] > df['Long'], 'Signal'] = 1
                
                df['Daily_Return'] = df['Close'].pct_change()
                df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
                
                total_return = (df['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
                total_return_pct = total_return * 100
                trades = df['Signal'].diff().abs().sum() / 2
                
                results.append({
                    'Short MA': short_w,
                    'Long MA': long_w,
                    'Return (%)': total_return_pct,
                    'Trades': trades
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Return (%)', ascending=False)
        return results_df, None
    except Exception as e:
        return None, str(e)

# =========================================================
# UI Configuration
# =========================================================
st.set_page_config(layout="wide", page_title="Quantitative Strategy Backtester")
st.title("Quantitative Strategy Backtester")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Settings")
sidebar_stock = st.sidebar.text_input("Ticker Symbol", value="TSLA")
days_to_test = st.sidebar.slider("Backtest Period (Days)", 30, 365, 60)
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters (MA)")

# =========================================================
# Section 1: Parameter Optimization (Moved to Top)
# =========================================================
with st.expander("Parameter Optimization", expanded=True):
    st.markdown("Test multiple Moving Average combinations based on 6-month historical data to find the optimal strategy.")
    
    col_opt1, col_opt2 = st.columns([1, 4])
    with col_opt1:
        if st.button("Run Optimization"):
            with st.spinner("Optimizing..."):
                results_df, error = run_optimization(sidebar_stock)
            
            if error:
                st.error(f"Error: {error}")
            elif results_df is not None:
                best_short = int(results_df.iloc[0]['Short MA'])
                best_long = int(results_df.iloc[0]['Long MA'])
                best_return = results_df.iloc[0]['Return (%)']
                
                # Auto-update session state
                st.session_state['short_window'] = best_short
                st.session_state['long_window'] = best_long
                
                st.success(f"Optimal Parameters Found: {best_short} / {best_long} (Return: {best_return:.2f}%)")
                st.info("Parameters have been automatically updated in the sidebar.")
                st.dataframe(results_df.head(5).style.format({'Return (%)': '{:.2f}%'}))

# Input widgets linked to session state
short_window = st.sidebar.number_input("Short MA", key='short_window', min_value=1)
long_window = st.sidebar.number_input("Long MA", key='long_window', min_value=2)

# =========================================================
# Section 2: Backtest & Analysis
# =========================================================
if st.button("Run Analysis (Backtest + Forecast)"):
    if short_window >= long_window:
        st.error("Invalid Parameters: Short MA must be less than Long MA.")
        st.stop()
        
    st.info(f"Fetching data for {sidebar_stock}...")
    
    try:
        # Fetch Data
        df_raw = yf.download(sidebar_stock, period="2y", interval="1d", progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        if df_raw.empty:
            st.error(f"Error: No data found for {sidebar_stock}.")
            st.stop()
        
        # Calculate Indicators
        df_raw['SMA_Short'] = df_raw['Close'].rolling(window=short_window).mean()
        df_raw['SMA_Long'] = df_raw['Close'].rolling(window=long_window).mean()
        df_raw['RSI'] = calculate_rsi(df_raw['Close'])
        
        df = df_raw.tail(days_to_test).copy()

        # Generate Signals
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = 0
        df['Position_Change'] = df['Signal'].diff()

        # Backtest Logic
        position = 0      
        cash = initial_capital
        trade_log = []    

        for date, row in df.iterrows():
            price = row['Close']
            change = row['Position_Change']
            date_str = date.strftime('%Y-%m-%d')
            
            if change == 1 and position == 0:
                position = cash / price
                cash = 0
                trade_log.append(f"[{date_str}] BUY  @ ${price:.2f}")
            elif change == -1 and position > 0:
                cash = position * price
                position = 0
                trade_log.append(f"[{date_str}] SELL @ ${price:.2f} (Cash: ${cash:.2f})")

        final_value = cash
        if position > 0:
            final_value = position * df.iloc[-1]['Close']

        roi = ((final_value - initial_capital) / initial_capital) * 100
        buy_hold_roi = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

        # ---------------------------------------------------------
        # Dashboard: Market Status
        # ---------------------------------------------------------
        st.subheader("Market Status Dashboard")
        last_row = df_raw.iloc[-1]
        prev_row = df_raw.iloc[-2]
        
        trend_status = "Bullish" if last_row['SMA_Short'] > last_row['SMA_Long'] else "Bearish"
        rsi_val = last_row['RSI']
        
        if rsi_val > 70: rsi_status = "Overbought"
        elif rsi_val < 30: rsi_status = "Oversold"
        else: rsi_status = "Neutral"

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Close Price", f"${last_row['Close']:.2f}", f"{last_row['Close'] - prev_row['Close']:.2f}")
        col_m2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
        col_m3.metric(f"Short MA ({short_window})", f"${last_row['SMA_Short']:.2f}")
        col_m4.metric(f"Long MA ({long_window})", f"${last_row['SMA_Long']:.2f}")
        
        st.write(f"**Trend Condition:** {trend_status}")

        # ---------------------------------------------------------
        # Backtest Performance
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Backtest Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Capital", f"${initial_capital:,.2f}")
        col2.metric("Final Equity", f"${final_value:,.2f}", delta=f"{roi:.2f}%")
        col3.metric("Buy & Hold Return", f"{buy_hold_roi:.2f}%")

        # ---------------------------------------------------------
        # Charting
        # ---------------------------------------------------------
        st.subheader("Price Chart & Signals")
        plots = []
        plots.append(mpf.make_addplot(df['SMA_Short'], color='orange', width=1.5, label=f'SMA {short_window}'))
        plots.append(mpf.make_addplot(df['SMA_Long'], color='blue', width=1.5, label=f'SMA {long_window}'))

        buy_signals = np.where(df['Position_Change'] == 1, df['Low']*0.98, np.nan)
        sell_signals = np.where(df['Position_Change'] == -1, df['High']*1.02, np.nan)

        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red', label='Buy'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green', label='Sell'))

        fig, axlist = mpf.plot(df, type='candle', style='yahoo', 
                           title=f'{sidebar_stock} Backtest Analysis',
                           volume=True, addplot=plots, returnfig=True, figsize=(12, 6))
        
        # Annotate prices on chart
        ax_main = axlist[0]
        for i, (index, row) in enumerate(df.iterrows()):
            if row['Position_Change'] == 1: # Buy
                ax_main.annotate(f"{row['Close']:.0f}", 
                                   xy=(i, row['Low']*0.98), 
                                   xytext=(0, -20), 
                                   textcoords='offset points', 
                                   ha='center', va='top', color='red', fontsize=8,
                                   arrowprops=dict(arrowstyle='-', color='red', alpha=0.3))
            elif row['Position_Change'] == -1: # Sell
                ax_main.annotate(f"{row['Close']:.0f}", 
                                   xy=(i, row['High']*1.02), 
                                   xytext=(0, 20), 
                                   textcoords='offset points', 
                                   ha='center', va='bottom', color='green', fontsize=8,
                                   arrowprops=dict(arrowstyle='-', color='green', alpha=0.3))

        st.pyplot(fig)

        # ---------------------------------------------------------
        # Future Prediction
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("3-Day Forecast (Beta)")
        st.markdown("Linear regression projection based on 15-day price momentum.")

        df_predict = predict_future_ma(df_raw, short_window, long_window, days_to_predict=3)
        pred_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_predict.iterrows()):
            date_label = idx.strftime('%m/%d (%a)')
            with pred_cols[i]:
                st.write(f"**{date_label}**")
                st.metric("Proj. Close", f"${row['Close']:.2f}")
                st.caption(f"SMA {short_window}: ${row['SMA_Short']:.2f}")
                st.caption(f"SMA {long_window}: ${row['SMA_Long']:.2f}")
                
                if row['SMA_Short'] > row['SMA_Long']:
                    st.success("Bullish")
                else:
                    st.error("Bearish")

    except Exception as e:
        st.error(f"An error occurred: {e}")
