import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np

# 設定初始參數，你可以隨時在側邊欄修改
DEFAULT_SHORT_MA = 20
DEFAULT_LONG_MA = 30
DEFAULT_CAPITAL = 10000.0

# =========================================================
# 1. 網頁介面配置
# =========================================================
st.set_page_config(layout="wide")
st.title("📈 美股自動回測系統")
st.markdown("---")

st.sidebar.header("🎯 參數設定")
sidebar_stock = st.sidebar.text_input("輸入股票代碼 (例如 TSLA, AMD, NVDA)", value="TSLA")
days_to_test = st.sidebar.slider("回測天數 (抓取最近N天的數據)", 30, 365, 60)
short_window = st.sidebar.number_input("短期均線 (MA)", value=DEFAULT_SHORT_MA, min_value=1)
long_window = st.sidebar.number_input("長期均線 (MA)", value=DEFAULT_LONG_MA, min_value=2)
initial_capital = st.sidebar.number_input("初始資金 ($)", value=DEFAULT_CAPITAL)


# =========================================================
# 2. 執行回測與繪圖
# =========================================================
if st.button("開始回測"):
    if short_window >= long_window:
        st.error("❌ 錯誤：短期均線天數必須小於長期均線天數！")
        st.stop()
        
    st.info(f"正在抓取 {sidebar_stock} 的資料 (將分析過去 {days_to_test} 個交易日的 {short_window}/{long_window} MA 策略)...")
    
    try:
        # 抓取資料 (抓 1 年，確保均線計算的數據是充足的)
        df_raw = yf.download(sidebar_stock, period="1y", interval="1d")
        
        # 處理資料標題
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        # === 錯誤檢查：如果抓不到資料，立即停止 ===
        if df_raw.empty:
            st.error(f"⚠️ 錯誤：無法抓取股票代碼 {sidebar_stock} 的數據。請檢查代碼或稍候再試。")
            st.stop()
        
        # 計算均線
        df_raw['SMA_Short'] = df_raw['Close'].rolling(window=short_window).mean()
        df_raw['SMA_Long'] = df_raw['Close'].rolling(window=long_window).mean()
        
        # 只取回測期間
        df = df_raw.tail(days_to_test).copy()

        # ==========================================
        # 3. 定義訊號
        # ==========================================
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1  # 買進/持有
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = 0  # 賣出/空手
        df['Position_Change'] = df['Signal'].diff()

        # ==========================================
        # 4. 回測運算 (計算損益)
        # ==========================================
        position = 0      
        cash = initial_capital
        trade_log = []    

        for date, row in df.iterrows():
            price = row['Close']
            change = row['Position_Change']
            date_str = date.strftime('%Y-%m-%d')
            
            # 買進訊號 (黃金交叉)
            if change == 1 and position == 0:
                position = cash / price
                cash = 0
                trade_log.append(f"[{date_str}] 黃金交叉買進 📈 @ ${price:.2f}")
                
            # 賣出訊號 (死亡交叉)
            elif change == -1 and position > 0:
                cash = position * price
                position = 0
                trade_log.append(f"[{date_str}] 死亡交叉賣出 📉 @ ${price:.2f} (資產: ${cash:.2f})")

        # 最終結算
        final_value = cash
        if position > 0:
            final_value = position * df.iloc[-1]['Close']

        roi = ((final_value - initial_capital) / initial_capital) * 100
        buy_hold_roi = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

        # ==========================================
        # 5. 結果呈現 (文字)
        # ==========================================
        st.subheader("📊 回測結果總結")
        col1, col2, col3 = st.columns(3)
        col1.metric("初始資金", f"${initial_capital:,.2f}")
        col2.metric("最終資產", f"${final_value:,.2f}", delta=f"{roi:.2f}%")
        col3.metric("持有基準 (Buy & Hold)", f"{buy_hold_roi:.2f}%")

        if roi > buy_hold_roi:
            st.success("✅ 策略表現優於單純持有！(Alpha)")
        elif roi > 0:
            st.warning("⚠️ 策略獲利，但報酬率輸給單純持有。")
        else:
             st.error("❌ 策略虧損。")

        # 交易紀錄
        st.subheader("📒 交易紀錄")
        if trade_log:
            for log in trade_log:
                st.code(log)
        else:
            st.info("該期間內沒有觸發任何交易訊號。")
            
        st.subheader("⬇️ 最近 5 天數據表")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_Short', 'SMA_Long', 'Signal']].tail())


        # ==========================================
        # 6. 繪圖 (K線圖與訊號)
        # ==========================================
        st.subheader("📉 K 線圖與交易訊號")
        plots = []

        # 加入均線
        plots.append(mpf.make_addplot(df['SMA_Short'], color='orange', width=1.5, label=f'SMA {short_window}'))
        plots.append(mpf.make_addplot(df['SMA_Long'], color='blue', width=1.5, label=f'SMA {long_window}'))

        # 標記買賣點
        buy_signals = np.where(df['Position_Change'] == 1, df['Low']*0.95, np.nan)
        sell_signals = np.where(df['Position_Change'] == -1, df['High']*1.05, np.nan)

        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red', label='Buy'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green', label='Sell'))

        # 繪圖
        fig, ax = mpf.plot(df, type='candle', style='yahoo', 
                           title=f'{sidebar_stock} {short_window}/{long_window} MA Cross Strategy',
                           volume=True, addplot=plots, returnfig=True, figsize=(12, 6))
        
        st.pyplot(fig)
        st.success("回測與繪圖完成！")

    except Exception as e:
        # 捕捉所有運行時的錯誤，並顯示在網頁上
        st.error(f"❌ 發生錯誤 (可能是數據格式問題)：{e}")
        st.info("請嘗試使用其他股票代碼或檢查均線參數設定。")
