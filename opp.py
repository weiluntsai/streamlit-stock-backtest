import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import timedelta

# 設定初始參數
DEFAULT_SHORT_MA = 20
DEFAULT_LONG_MA = 30
DEFAULT_CAPITAL = 10000.0

# =========================================================
# 輔助函式：計算 RSI
# =========================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================================
# 輔助函式：預測未來股價 (線性回歸)
# =========================================================
def predict_future_ma(df_historical, short_window, long_window, days_to_predict=3):
    # 1. 準備數據：取最近 15 天來抓趨勢
    recent_data = df_historical['Close'].tail(15)
    
    # 建立 X (時間序) 和 Y (價格)
    x = np.arange(len(recent_data))
    y = recent_data.values
    
    # 2. 線性回歸 (擬合一條直線 y = mx + b)
    z = np.polyfit(x, y, 1) 
    p = np.poly1d(z)
    
    # 3. 預測未來 N 天的價格
    future_x = np.arange(len(recent_data), len(recent_data) + days_to_predict)
    future_prices = p(future_x)
    
    # 4. 產生未來的日期 (跳過週末)
    last_date = df_historical.index[-1]
    future_dates = []
    current_date = last_date
    while len(future_dates) < days_to_predict:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5: # 0-4 是週一到週五
            future_dates.append(current_date)
            
    # 5. 建立未來數據的 DataFrame
    df_future = pd.DataFrame(index=future_dates)
    df_future['Close'] = future_prices
    
    # 合併歷史與未來數據以計算均線
    # (我們只需要 Close 欄位來算均線)
    df_combined = pd.concat([df_historical[['Close']], df_future[['Close']]])
    
    # 計算均線
    df_combined['SMA_Short'] = df_combined['Close'].rolling(window=short_window).mean()
    df_combined['SMA_Long'] = df_combined['Close'].rolling(window=long_window).mean()
    
    # 只回傳未來預測的部分
    return df_combined.tail(days_to_predict)

# =========================================================
# 1. 網頁介面配置
# =========================================================
st.set_page_config(layout="wide", page_title="美股自動回測與預測系統")
st.title("📈 美股自動回測與預測系統")
st.markdown("---")

st.sidebar.header("🎯 參數設定")
sidebar_stock = st.sidebar.text_input("輸入股票代碼 (例如 TSLA, AMD, NVDA)", value="TSLA")
days_to_test = st.sidebar.slider("回測天數 (抓取最近N天的數據)", 30, 365, 60)
short_window = st.sidebar.number_input("短期均線 (MA)", value=DEFAULT_SHORT_MA, min_value=1)
long_window = st.sidebar.number_input("長期均線 (MA)", value=DEFAULT_LONG_MA, min_value=2)
initial_capital = st.sidebar.number_input("初始資金 ($)", value=DEFAULT_CAPITAL)


# =========================================================
# 執行參數優化函式
# =========================================================
def run_optimization(stock_symbol):
    """執行參數最佳化"""
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50, 60]
    results = []

    df_raw = yf.download(stock_symbol, period="6mo", interval="1d")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    if df_raw.empty:
        return None, "無法抓取優化所需的數據。"

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
                '短均線': short_w,
                '長均線': long_w,
                '報酬率(%)': total_return_pct,
                '交易次數': trades
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='報酬率(%)', ascending=False)
    return results_df, None


# =========================================================
# 2. 執行回測、分析與預測
# =========================================================
if st.button("開始分析 (回測 + 預測)"):
    if short_window >= long_window:
        st.error("❌ 錯誤：短期均線天數必須小於長期均線天數！")
        st.stop()
        
    st.info(f"正在抓取 {sidebar_stock} 數據並進行 AI 運算...")
    
    try:
        # 抓取資料
        df_raw = yf.download(sidebar_stock, period="2y", interval="1d") # 抓久一點以確保 RSI 和長天期均線準確
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        if df_raw.empty:
            st.error(f"⚠️ 錯誤：無法抓取股票代碼 {sidebar_stock} 的數據。")
            st.stop()
        
        # --- A. 技術指標計算 ---
        df_raw['SMA_Short'] = df_raw['Close'].rolling(window=short_window).mean()
        df_raw['SMA_Long'] = df_raw['Close'].rolling(window=long_window).mean()
        df_raw['RSI'] = calculate_rsi(df_raw['Close']) # 新增 RSI
        
        # 準備回測用的數據 (只取最後 N 天)
        df = df_raw.tail(days_to_test).copy()

        # --- B. 產生買賣訊號 ---
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = 0
        df['Position_Change'] = df['Signal'].diff()

        # --- C. 回測運算 ---
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
                trade_log.append(f"[{date_str}] 黃金交叉買進 📈 @ ${price:.2f}")
            elif change == -1 and position > 0:
                cash = position * price
                position = 0
                trade_log.append(f"[{date_str}] 死亡交叉賣出 📉 @ ${price:.2f} (資產: ${cash:.2f})")

        final_value = cash
        if position > 0:
            final_value = position * df.iloc[-1]['Close']

        roi = ((final_value - initial_capital) / initial_capital) * 100
        buy_hold_roi = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

        # ==========================================
        # 顯示區塊 1: 技術分析儀表板 (New!)
        # ==========================================
        st.subheader("🔍 當前技術分析儀表板")
        
        last_row = df_raw.iloc[-1]
        prev_row = df_raw.iloc[-2]
        
        # 判斷多空狀態
        trend_status = "🟢 多頭排列 (強勢)" if last_row['SMA_Short'] > last_row['SMA_Long'] else "🔴 空頭排列 (弱勢)"
        rsi_val = last_row['RSI']
        
        # RSI 狀態判讀
        if rsi_val > 70: rsi_status = "🔥 超買區 (小心回檔)"
        elif rsi_val < 30: rsi_status = "❄️ 超賣區 (醞釀反彈)"
        else: rsi_status = "⚖️ 中性區間"

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("最新收盤價", f"${last_row['Close']:.2f}", f"{last_row['Close'] - prev_row['Close']:.2f}")
        col_m2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
        col_m3.metric(f"短期均線 ({short_window}MA)", f"${last_row['SMA_Short']:.2f}")
        col_m4.metric(f"長期均線 ({long_window}MA)", f"${last_row['SMA_Long']:.2f}")
        
        st.info(f"📊 目前市場狀態：**{trend_status}**")

        # ==========================================
        # 顯示區塊 2: 回測結果
        # ==========================================
        st.subheader("🔙 歷史回測結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("初始資金", f"${initial_capital:,.2f}")
        col2.metric("最終資產", f"${final_value:,.2f}", delta=f"{roi:.2f}%")
        col3.metric("持有基準", f"{buy_hold_roi:.2f}%")

        # K線圖
        st.subheader("📉 K 線圖與交易訊號")
        plots = []
        plots.append(mpf.make_addplot(df['SMA_Short'], color='orange', width=1.5, label=f'SMA {short_window}'))
        plots.append(mpf.make_addplot(df['SMA_Long'], color='blue', width=1.5, label=f'SMA {long_window}'))

        buy_signals = np.where(df['Position_Change'] == 1, df['Low']*0.98, np.nan)
        sell_signals = np.where(df['Position_Change'] == -1, df['High']*1.02, np.nan)

        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red', label='Buy'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green', label='Sell'))

        fig, ax = mpf.plot(df, type='candle', style='yahoo', 
                           title=f'{sidebar_stock} Strategy Backtest',
                           volume=True, addplot=plots, returnfig=True, figsize=(12, 6))
        st.pyplot(fig)

        # ==========================================
        # 顯示區塊 3: 未來預測 (New!)
        # ==========================================
        st.markdown("---")
        st.subheader("🔮 未來 3 日趨勢預測 (Beta)")
        st.markdown(f"此模組使用**線性回歸**演算法，根據過去 15 天的價格慣性，推估如果趨勢不變，未來 3 天的均線走向。")

        # 執行預測
        df_predict = predict_future_ma(df_raw, short_window, long_window, days_to_predict=3)
        
        # 顯示預測表格
        pred_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_predict.iterrows()):
            date_label = idx.strftime('%m/%d (%a)')
            with pred_cols[i]:
                st.markdown(f"##### 📅 {date_label}")
                st.metric("預測收盤", f"${row['Close']:.2f}")
                st.write(f"短均線: **${row['SMA_Short']:.2f}**")
                st.write(f"長均線: **${row['SMA_Long']:.2f}**")
                
                # 簡單的預測解讀
                if row['SMA_Short'] > row['SMA_Long']:
                    st.success("預測: 維持多頭")
                else:
                    st.error("預測: 維持空頭")

    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")
        st.info("請檢查股票代碼或數據源連線。")


# =========================================================
# 7. 參數優化器區塊
# =========================================================
st.markdown("---")
with st.expander("🛠️ 參數優化器 (找出最佳均線組合)", expanded=False):
    st.markdown("此功能將測試多組短期/長期均線組合，並依據過去 6 個月的歷史報酬率進行排名。")
    if st.button(f"開始優化 {sidebar_stock} 參數"):
        with st.spinner("🚀 正在運行回測模擬，請稍候..."):
            results_df, error = run_optimization(sidebar_stock)
        
        if error:
            st.error(error)
        elif results_df is not None:
            best_short = results_df.iloc[0]['短均線']
            best_long = results_df.iloc[0]['長均線']
            best_return = results_df.iloc[0]['報酬率(%)']
            
            st.success(f"最佳組合: **{best_short}日 / {best_long}日** (報酬率: {best_return:.2f}%)")
            st.dataframe(results_df.head(5).style.format({'報酬率(%)': '{:.2f}%'}))
