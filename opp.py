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
# 1. 網頁介面配置 - 採用深色風格
# =========================================================
st.set_page_config(layout="wide")
st.title("📈 美股自動回測系統")
st.markdown("---")

# 透過 CSS 讓 Streamlit 介面更接近暗色風格 (依賴 Streamlit 運行環境的支援)
# 註：若要在 Streamlit Cloud 強制暗色，需要在 .streamlit/config.toml 中設定，這裡提供軟性調整
st.markdown("""
    <style>
    /* 讓 Streamlit 的主要內容區域使用深色背景，以配合圖表 */
    .stApp {
        background-color: #121417; 
        color: #ddd;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

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
    """執行參數最佳化，找出歷史上報酬率最高的均線組合"""
    
    # 我們要測試的均線組合
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50, 60]
    results = []

    # 抓取資料 (確保 6 個月資料用於優化)
    df_raw = yf.download(stock_symbol, period="6mo", interval="1d")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    # 檢查數據
    if df_raw.empty:
        return None, "無法抓取優化所需的數據。"

    # 雙重迴圈：測試每一種組合
    for short_w in short_windows:
        for long_w in long_windows:
            if short_w >= long_w:
                continue
            
            df = df_raw.copy()
            df['Short'] = df['Close'].rolling(window=short_w).mean()
            df['Long'] = df['Close'].rolling(window=long_w).mean()
            
            # 產生訊號
            df['Signal'] = 0
            df.loc[df['Short'] > df['Long'], 'Signal'] = 1
            
            # 計算策略報酬
            df['Daily_Return'] = df['Close'].pct_change()
            df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
            
            total_return = (df['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
            total_return_pct = total_return * 100
            
            # 紀錄交易次數
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
        # 6. 繪圖 (K線圖與訊號) - TradingView 風格
        # ==========================================
        st.subheader("📉 K 線圖與交易訊號")
        plots = []

        # 加入均線
        # 調整均線顏色使其在暗色背景下更顯眼
        plots.append(mpf.make_addplot(df['SMA_Short'], color='#FF9900', width=1.5, label=f'SMA {short_window}')) # 亮橘色
        plots.append(mpf.make_addplot(df['SMA_Long'], color='#00BCD4', width=1.5, label=f'SMA {long_window}')) # 淺藍色

        # 標記買賣點
        # 買入 (紅) 和 賣出 (綠) 保持高對比
        buy_signals = np.where(df['Position_Change'] == 1, df['Low']*0.95, np.nan)
        sell_signals = np.where(df['Position_Change'] == -1, df['High']*1.05, np.nan)

        if not np.all(np.isnan(buy_signals)):
            plots.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='red', label='Buy'))
        if not np.all(np.isnan(sell_signals)):
            plots.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='green', label='Sell'))

        # 繪圖 - 使用 'binance' style，這是常見的暗色高對比風格
        fig, ax = mpf.plot(df, type='candle', style='binance', 
                           title=f'{sidebar_stock} {short_window}/{long_window} MA Cross Strategy',
                           volume=True, addplot=plots, returnfig=True, figsize=(12, 6))
        
        st.pyplot(fig)
        st.success("回測與繪圖完成！")

    except Exception as e:
        # 捕捉所有運行時的錯誤，並顯示在網頁上
        st.error(f"❌ 發生錯誤 (可能是數據格式問題)：{e}")
        st.info("請嘗試使用其他股票代碼或檢查均線參數設定。")


# =========================================================
# 7. 參數優化器區塊
# =========================================================
st.markdown("---")
with st.expander("🛠️ 參數優化器 (找出最佳均線組合)", expanded=False):
    st.markdown("此功能將測試多組短期/長期均線組合 (例如 5/20, 10/30...)，並依據過去 6 個月的歷史報酬率進行排名。")
    
    if st.button(f"開始優化 {sidebar_stock} 參數 (約 5-10 秒)"):
        with st.spinner("🚀 正在運行回測模擬，請稍候..."):
            results_df, error = run_optimization(sidebar_stock)
        
        if error:
            st.error(error)
        elif results_df is not None:
            # 取得第一名的參數
            best_short = results_df.iloc[0]['短均線']
            best_long = results_df.iloc[0]['長均線']
            best_return = results_df.iloc[0]['報酬率(%)']
            best_trades = results_df.iloc[0]['交易次數']
            
            st.subheader(f"🥇 {sidebar_stock} 最佳策略參數")
            st.success(f"最佳組合: **{best_short}日 / {best_long}日**")
            st.info(f"歷史報酬率: **{best_return:.2f}%** (交易次數: {best_trades:.1f})")
            st.markdown(f"您可以將側邊欄的 MA 參數改為 **{best_short} / {best_long}** 進行精確回測。")
            
            st.subheader("完整參數排行榜 (Top 10)")
            st.dataframe(results_df.head(10).style.format({'報酬率(%)': '{:.2f}%'}))
