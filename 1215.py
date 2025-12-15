import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# Initialize session state for parameters
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = 20
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = 60

# =========================================================
# Technical Indicator Helper Functions
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
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
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

def calculate_adx(df, period=14):
    """Average Directional Index (ADX)"""
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    # Smoothing using Wilder's method (alpha=1/period is close approximation)
    df['TR14'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=1/period, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=1/period, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
    df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    
    return df['ADX']

# =========================================================
# Helper Function: Future Prediction (Linear Regression)
# =========================================================
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
            return None, f"No data found for symbol: {stock_symbol}"

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
                trades = df['Signal'].diff().abs().sum() / 2
                
                results.append({
                    'Short MA': short_w,
                    'Long MA': long_w,
                    'Return (%)': total_return_pct,
                    'Trades': trades
                })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(by='Return (%)', ascending=False)
        else:
            return None, "Not enough data to optimize."
            
        return results_df, None
    except Exception as e:
        return None, str(e)

# =========================================================
# UI Configuration
# =========================================================
st.set_page_config(layout="wide", page_title="Global Quant Backtester Pro")
st.title("Global Quant Backtester Pro (With Advanced Filters)")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Settings")

# --- Market Selection Logic ---
market_type = st.sidebar.selectbox("Market / Exchange", ["US Market", "Taiwan (TWSE - .TW)", "Taiwan (TPEx - .TWO)"])
ticker_input = st.sidebar.text_input("Ticker Symbol (e.g. TSLA or 2330)", value="2330")

# Auto-append suffix based on selection
final_ticker = ticker_input.strip().upper()

if market_type == "Taiwan (TWSE - .TW)":
    if not final_ticker.endswith(".TW"):
        final_ticker += ".TW"
elif market_type == "Taiwan (TPEx - .TWO)":
    if not final_ticker.endswith(".TWO"):
        final_ticker += ".TWO"

st.sidebar.caption(f"Processing as: **{final_ticker}**")

days_to_test = st.sidebar.slider("Backtest Period (Days)", 30, 365, 120)
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters (MA)")

# Optimization Section
with st.expander("Parameter Optimization", expanded=False):
    st.markdown("Find optimal MA combinations based on 6-month history.")
    if st.button("Run Optimizer"):
        with st.spinner(f"Optimizing for {final_ticker}..."):
            results_df, error = run_optimization(final_ticker)
        
        if error:
            st.error(f"Error: {error}")
        elif results_df is not None:
            best_short = int(results_df.iloc[0]['Short MA'])
            best_long = int(results_df.iloc[0]['Long MA'])
            best_return = results_df.iloc[0]['Return (%)']
            st.session_state['short_window'] = best_short
            st.session_state['long_window'] = best_long
            st.success(f"Optimal: {best_short}/{best_long} (Ret: {best_return:.2f}%)")

short_window = st.sidebar.number_input("Short MA", key='short_window', min_value=1)
long_window = st.sidebar.number_input("Long MA", key='long_window', min_value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Filters (Strategy Refinement)")
st.sidebar.caption("條件更嚴格，交易次數會減少，但準確度提升。")

use_rsi_filter = st.sidebar.checkbox("1. RSI Trend Filter", value=True, help="Buy if RSI > 50 (Strong), Sell if RSI < 50 (Weak).")
use_adx_filter = st.sidebar.checkbox("2. ADX Trend Strength (> 25)", value=False, help="Only trade when ADX > 25. Avoids sideways markets.")
use_price_filter = st.sidebar.checkbox("3. Price Breakout Filter", value=False, help="Buy only if Price > MA Cross Point + X%.")
price_filter_pct = 0.0
if use_price_filter:
    price_filter_pct = st.sidebar.slider("Breakout Threshold (%)", 0.5, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Settings")
show_bollinger = st.sidebar.checkbox("Bollinger Bands (Overlay)", value=True)
add_indicator = st.sidebar.selectbox("Additional Panel Indicator", ["None", "MACD", "KDJ", "OBV", "RSI", "ADX"], index=1)

# =========================================================
# Analysis Section
# =========================================================
if st.button("Run Analysis (Backtest + Forecast)"):
    if short_window >= long_window:
        st.error("Short MA must be less than Long MA.")
        st.stop()
        
    st.info(f"Analyzing {final_ticker} with filters...")
    
    try:
        # Fetch Data (Fetch extra data for calculating indicators properly)
        fetch_days = days_to_test + 100 
        df_raw = yf.download(final_ticker, period=f"{fetch_days}d", interval="1d", progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        if df_raw.empty:
            st.error(f"No data found for {final_ticker}.")
            st.stop()
        
        # --- Calculate Indicators ---
        # 1. Moving Averages
        df_raw['SMA_Short'] = df_raw['Close'].rolling(window=short_window).mean()
        df_raw['SMA_Long'] = df_raw['Close'].rolling(window=long_window).mean()
        
        # 2. RSI
        df_raw['RSI'] = calculate_rsi(df_raw['Close'])
        
        # 3. Bollinger Bands
        df_raw = calculate_bollinger_bands(df_raw)
        
        # 4. MACD
        df_raw = calculate_macd(df_raw)
        
        # 5. KDJ
        df_raw = calculate_kdj(df_raw)
        
        # 6. OBV
        df_raw = calculate_obv(df_raw)
        
        # 7. ADX (New)
        df_raw['ADX'] = calculate_adx(df_raw)
        
        # Slice for Backtest (Ensure we have data after indicator calc)
        df = df_raw.tail(days_to_test).copy()

        # =================================================
        # Advanced Signal Logic (State Machine)
        # =================================================
        
        # Raw MA Signal (Vectorized) - Just for reference or base
        df['Raw_MA_Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
        df['Raw_Cross'] = df['Raw_MA_Signal'].diff() # 1=Gold, -1=Death
        
        final_signals = []
        current_state = 0 # 0=Cash, 1=Holding
        
        for index, row in df.iterrows():
            signal_out = current_state
            
            # --- Check Buy Conditions ---
            if current_state == 0:
                # Basic condition: Golden Cross
                # Note: We use Raw_Cross == 1 which means Short crossed ABOVE Long today
                if row['Raw_Cross'] == 1:
                    can_buy = True
                    reasons = []
                    
                    # Filter 1: RSI
                    if use_rsi_filter:
                        if row['RSI'] <= 50:
                            can_buy = False
                            reasons.append("RSI<=50")
                    
                    # Filter 2: ADX
                    if use_adx_filter:
                        if row['ADX'] <= 25:
                            can_buy = False
                            reasons.append("ADX<=25")
                            
                    # Filter 3: Price Breakout
                    # "Price must be > SMA_Long * (1 + x%)"
                    if use_price_filter:
                        threshold_price = row['SMA_Long'] * (1 + price_filter_pct/100)
                        if row['Close'] <= threshold_price:
                            can_buy = False
                            reasons.append(f"Price<{price_filter_pct}%Breakout")
                    
                    if can_buy:
                        signal_out = 1 # BUY
            
            # --- Check Sell Conditions ---
            elif current_state == 1:
                # Basic condition: Death Cross
                if row['Raw_Cross'] == -1:
                    can_sell = True
                    
                    # Filter 1: RSI (Sell if Weak)
                    # User: "Death Cross AND RSI < 50"
                    if use_rsi_filter:
                        if row['RSI'] >= 50: # If RSI is still strong, maybe hold?
                            can_sell = False 
                            # Note: This means we ignore the Death Cross and keep holding
                            # until a Death Cross happens AND RSI is weak.
                            # But Raw_Cross only happens ONCE. 
                            # So if we miss it, we might hold forever until next cross?
                            # Improved Logic: If Short < Long (Bearish Zone) AND RSI < 50 -> Sell
                            pass

                    # Refined Sell Logic for State Machine:
                    # If we are holding, and MA is bearish (Short < Long), 
                    # we check if we should exit based on filters.
                    # If filters say "Don't Sell", we keep holding (Signal=1) even if MA crossed.
                    
                    if use_rsi_filter and row['RSI'] >= 50:
                        can_sell = False
                    
                    if use_adx_filter and row['ADX'] <= 25:
                        # User: "If ADX < 25... ignore MA signals"
                        # Means don't sell in chop? Or force sell? 
                        # Usually "Trend Following" means don't trade in chop. 
                        # If we are already IN, and market goes chop, do we exit?
                        # Usually we exit to protect capital. 
                        # But strictly following "Ignore Signal":
                        can_sell = False 

                    if can_sell:
                        signal_out = 0 # SELL
                
                # Special Check: If we missed the cross day because of filters, 
                # but now conditions are met while still in Bearish MA zone?
                # Simple version: Only act on Cross Day to strictly follow "Signal".
                # If we filter out the Death Cross, we hold until next Golden -> Death cycle?
                # Or we check every day "Is Short < Long AND RSI < 50"? -> This is better.
                
                if current_state == 1 and row['SMA_Short'] < row['SMA_Long']:
                    # We are in Bearish Zone, but maybe still holding due to filters
                    should_liquidate = True
                    if use_rsi_filter and row['RSI'] >= 50: should_liquidate = False
                    if use_adx_filter and row['ADX'] <= 25: should_liquidate = False
                    
                    if should_liquidate:
                        signal_out = 0

            current_state = signal_out
            final_signals.append(signal_out)
            
        df['Signal'] = final_signals
        df['Position_Change'] = df['Signal'].diff()

        # Backtest Engine
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
                trade_log.append(f"[{date_str}] BUY  @ {price:.2f}")
            elif change == -1 and position > 0:
                cash = position * price
                position = 0
                trade_log.append(f"[{date_str}] SELL @ {price:.2f} (Cash: {cash:.2f})")

        final_value = cash
        if position > 0:
            final_value = position * df.iloc[-1]['Close']

        roi = ((final_value - initial_capital) / initial_capital) * 100
        buy_hold_roi = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

        # ---------------------------------------------------------
        # Dashboard
        # ---------------------------------------------------------
        st.subheader(f"Market Status Dashboard ({final_ticker})")
        curr = df_raw.iloc[-1]
        
        # Logic for status
        trend_status = "Bullish" if curr['SMA_Short'] > curr['SMA_Long'] else "Bearish"
        adx_status = f"Strong ({curr['ADX']:.1f})" if curr['ADX'] > 25 else f"Weak/Chop ({curr['ADX']:.1f})"
        
        # RSI Logic
        if curr['RSI'] > 70: rsi_status = "Overbought"
        elif curr['RSI'] < 30: rsi_status = "Oversold"
        else: rsi_status = "Neutral"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trend (MA)", trend_status, f"{curr['Close']:.2f}")
        c2.metric("Trend Strength (ADX)", adx_status)
        c3.metric("RSI (14)", f"{curr['RSI']:.1f}", rsi_status)
        c4.metric("Volatility (BB)", f"{curr['BB_Width']:.2f}")

        # ---------------------------------------------------------
        # Performance & Chart
        # ---------------------------------------------------------
        st.markdown("---")
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Strategy Return", f"{roi:.2f}%")
        c_p2.metric("Buy & Hold Return", f"{buy_hold_roi:.2f}%")
        c_p3.metric("Final Equity", f"${final_value:,.2f}")

        st.subheader("Technical Analysis Chart")
        
        # Prepare plots
        plots = []
        # MAs
        plots.append(mpf.make_addplot(df['SMA_Short'], color='orange', width=1.5, label=f'SMA {short_window}'))
        plots.append(mpf.make_addplot(df['SMA_Long'], color='blue', width=1.5, label=f'SMA {long_window}'))
        
        if show_bollinger:
            plots.append(mpf.make_addplot(df['BB_Upper'], color='gray', alpha=0.3, width=0.8))
            plots.append(mpf.make_addplot(df['BB_Lower'], color='gray', alpha=0.3, width=0.8))

        # Additional Indicator Panel
        if add_indicator == "MACD":
            plots.append(mpf.make_addplot(df['MACD_Line'], panel=1, color='fuchsia', ylabel='MACD'))
            plots.append(mpf.make_addplot(df['MACD_Signal'], panel=1, color='b'))
            plots.append(mpf.make_addplot(df['MACD_Hist'], type='bar', panel=1, color='dimgray', alpha=0.5))
        elif add_indicator == "RSI":
            plots.append(mpf.make_addplot(df['RSI'], panel=1, color='purple', ylabel='RSI', ylim=(0, 100)))
            plots.append(mpf.make_addplot([70]*len(df), panel=1, color='red', linestyle='--', width=0.8))
            plots.append(mpf.make_addplot([30]*len(df), panel=1, color='green', linestyle='--', width=0.8))
            plots.append(mpf.make_addplot([50]*len(df), panel=1, color='black', linestyle=':', width=0.5))
        elif add_indicator == "KDJ":
            plots.append(mpf.make_addplot(df['K'], panel=1, color='orange', ylabel='KDJ'))
            plots.append(mpf.make_addplot(df['D'], panel=1, color='blue'))
            plots.append(mpf.make_addplot(df['J'], panel=1, color='purple'))
        elif add_indicator == "OBV":
            plots.append(mpf.make_addplot(df['OBV'], panel=1, color='teal', ylabel='OBV'))
        elif add_indicator == "ADX":
            plots.append(mpf.make_addplot(df['ADX'], panel=1, color='black', ylabel='ADX', ylim=(0, 100)))
            plots.append(mpf.make_addplot([25]*len(df), panel=1, color='red', linestyle='--', width=0.8))

        # Buy/Sell Markers (Based on Filtered Signals)
        buy_signals = np.where(df['Position_Change'] == 1, df['Low']*0.98, np.nan)
        sell_signals = np.where(df['Position_Change'] == -1, df['High']*1.02, np.nan)
        
        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green'))

        has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
        show_vol_panel = (add_indicator == "None") and has_volume

        fig, axlist = mpf.plot(df, type='candle', style='yahoo', 
                           title=f'{final_ticker} Technical Analysis (Filtered)',
                           volume=show_vol_panel, 
                           addplot=plots, returnfig=True, figsize=(12, 8),
                           panel_ratios=(2, 1) if add_indicator != "None" else (1,))
        
        st.pyplot(fig)

        # ---------------------------------------------------------
        # Future Forecast
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("3-Day Price Forecast")
        
        df_predict = predict_future_ma(df_raw, short_window, long_window, days_to_predict=3)
        cols = st.columns(3)
        for i, (idx, row) in enumerate(df_predict.iterrows()):
            with cols[i]:
                st.write(f"**{idx.strftime('%m/%d')}**")
                st.metric("Proj. Price", f"{row['Close']:.2f}")
                signal = "Bullish" if row['SMA_Short'] > row['SMA_Long'] else "Bearish"
                color = "green" if signal == "Bullish" else "red"
                st.markdown(f"Signal: :{color}[{signal}]")

    except Exception as e:
        st.error(f"An error occurred: {e}")
