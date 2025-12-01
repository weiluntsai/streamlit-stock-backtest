import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np

# 1. 網頁標題
st.title("📈 美股自動回測系統")
st.write("輸入股票代碼，自動尋找最佳均線策略！")

# 2. 側邊欄輸入參數
sidebar_stock = st.sidebar.text_input("輸入股票代碼 (例如 TSLA, AMD, NVDA)", value="TSLA")
days_to_test = st.sidebar.slider("回測天數", 30, 365, 60)

# 當使用者按下按鈕才開始跑
if st.button("開始回測"):
    st.write(f"正在抓取 {sidebar_stock} 的資料...")
    
    # --- 以下是你原本的邏輯，稍微改寫一點點 ---
    try:
        df = yf.download(sidebar_stock, period="1y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 簡單示範：直接用參數優化後的結果 (假設 AMD 最佳是 5/20)
        # 你也可以把原本的雙重迴圈加進來，讓網頁當場幫你算最佳參數
        short_window = 20
        long_window = 30
        
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # 只取回測期間
        data_slice = df.tail(days_to_test).copy()
        
        # 產生訊號
        data_slice['Signal'] = 0
        data_slice.loc[data_slice['SMA_Short'] > data_slice['SMA_Long'], 'Signal'] = 1
        data_slice.loc[data_slice['SMA_Short'] < data_slice['SMA_Long'], 'Signal'] = 0
        data_slice['Position_Change'] = data_slice['Signal'].diff()

        # 3. 顯示數據表格
        st.subheader(f"{sidebar_stock} 最近 5 天數據")
        st.dataframe(data_slice.tail())

        # 4. 畫圖 (這是最重要的一步)
        st.subheader("K 線圖與交易訊號")
        
        # 設定買賣點
        plots = []
        plots.append(mpf.make_addplot(data_slice['SMA_Short'], color='orange'))
        plots.append(mpf.make_addplot(data_slice['SMA_Long'], color='blue'))
        
        buy_signals = np.where(data_slice['Position_Change'] == 1, data_slice['Low']*0.95, np.nan)
        sell_signals = np.where(data_slice['Position_Change'] == -1, data_slice['High']*1.05, np.nan)
        
        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green'))

        # 關鍵：在 Streamlit 畫圖要用 fig, ax
        fig, ax = mpf.plot(data_slice, type='candle', style='yahoo', 
                           volume=True, addplot=plots, returnfig=True)
        
        st.pyplot(fig)
        st.success("回測完成！")

    except Exception as e:
        st.error(f"發生錯誤：{e}")
