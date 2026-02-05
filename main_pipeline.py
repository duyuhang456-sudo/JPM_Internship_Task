import pandas as pd
import numpy as np
import yfinance as yf
import os
from textblob import TextBlob
from scipy.optimize import minimize
from datetime import datetime, timedelta


# ==========================================
# JPM Quantitative Data Pipeline
# Author: Intern @ Huatai
# Description: Automated data collection, cleaning, and feature engineering.
# ==========================================

class JPMDataPipeline:
    def __init__(self, start_date="2018-01-01", end_date="2025-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        # 自动生成文件名
        self.output_raw = "JPM_Raw_Merged.csv"
        self.output_cleaned = "JPM_Cleaned.csv"
        self.output_final = "JPM_Final_Features.csv"

        # 确保时区处理一致
        print(f"🚀 初始化流水线: {start_date} 至 {end_date}")

    # -------------------------------------------------------
    # 模块 1: 数据收集 (Data Collection)
    # -------------------------------------------------------
    def step_1_fetch_market_data(self):
        """收集核心金融数据 (Yahoo Finance)"""
        print("\n[Step 1] Fetching Market Data (JPM Stock, Div, VIX)...")

        # 1. JPM Stock Data
        jpm = yf.Ticker("JPM")
        df_price = jpm.history(start=self.start_date, end=self.end_date)
        df_price.index = df_price.index.tz_localize(None)  # 去除时区
        df_price = df_price[['Close', 'Volume', 'Dividends']]
        df_price.rename(columns={
            'Close': 'JPMorgan_Stock_Price',
            'Volume': 'JPMorgan_Stock_Volume',
            'Dividends': 'JPMorgan_Dividends'
        }, inplace=True)

        # Calculate Dividend Yield
        df_price['Dividend_Yield'] = df_price['JPMorgan_Dividends'].rolling(252).sum() / df_price[
            'JPMorgan_Stock_Price']

        # 2. VIX Data
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(start=self.start_date, end=self.end_date)
        df_vix.index = df_vix.index.tz_localize(None)
        df_price['VIX_Volatility_Index'] = df_vix['Close']

        return df_price

    def step_2_fetch_macro_data(self):
        """收集宏观经济数据 (FRED / Treasury)"""
        print("[Step 2] Fetching Macro Data (Rates, Fed)...")
        # 注意：在真实流水线中，应使用 pandas_datareader 或 FRED API
        # 这里为了演示，我们尝试读取本地 CSV，如果不存在则模拟下载或报错

        macro_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))

        try:
            # 尝试读取之前下载的 CSV
            if os.path.exists("DFF.csv"):
                dff = pd.read_csv("DFF.csv", index_col=0, parse_dates=True)
                macro_df['Fed_Interest_Rate'] = dff['DFF']
            else:
                # [备用逻辑] 如果没有文件，使用常量填充或报错 (此处用前值填充模拟)
                print("   ⚠️ 未找到 DFF.csv，使用前值填充策略...")
                macro_df['Fed_Interest_Rate'] = 4.33

            if os.path.exists("DGS1.csv"):
                dgs1 = pd.read_csv("DGS1.csv", index_col=0, parse_dates=True)
                macro_df['US_Treasury_1Y_Rate'] = dgs1['DGS1']
            else:
                print("   ⚠️ 未找到 DGS1.csv，使用前值填充策略...")
                macro_df['US_Treasury_1Y_Rate'] = 4.20

        except Exception as e:
            print(f"   宏观数据加载错误: {e}")

        return macro_df

    def step_3_process_sentiment_data(self):
        """处理情绪数据 (News & Social)"""
        print("[Step 3] Processing Sentiment Data...")

        sentiment_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))

        # A. News Sentiment (GNews/TextBlob)
        # 如果有 GNews API 结果文件则读取，否则生成模拟数据
        if os.path.exists("JPM_GNews_Sentiment_Daily_2018_2025.csv"):
            gnews = pd.read_csv("JPM_GNews_Sentiment_Daily_2018_2025.csv", index_col=0, parse_dates=True)
            sentiment_df = sentiment_df.join(gnews['GNews_Sentiment_Score'], how='left')
            sentiment_df.rename(columns={'GNews_Sentiment_Score': 'Financial_News_Sentiment'}, inplace=True)
        else:
            print("   ⚠️ 未找到新闻情绪文件，需运行 Financial news sentiment.py 获取。")
            sentiment_df['Financial_News_Sentiment'] = 0.5  # 默认中性

        # B. Social Media Trend (Google Trends)
        # =======================================================
        # [MANUAL STEP REQUIRED / 手动步骤]
        # 由于 Google Trends 反爬严格，此数据需人工下载 CSV (multiTimeline.csv)
        # 并重命名为 JPM_Social_Trend_Intensity_2018_2025.csv
        # =======================================================
        if os.path.exists("JPM_Social_Trend_Intensity_2018_2025.csv"):
            social = pd.read_csv("JPM_Social_Trend_Intensity_2018_2025.csv", index_col=0, parse_dates=True)
            # 确保列名匹配
            col_name = 'Social_Trend_Score' if 'Social_Trend_Score' in social.columns else social.columns[0]
            sentiment_df = sentiment_df.join(social[col_name], how='left')
            sentiment_df.rename(columns={col_name: 'Social_Media_Market_Discussion_Score'}, inplace=True)
        else:
            print("   ⚠️ [MANUAL] 未找到 Google Trends 手动下载文件。")
            sentiment_df['Social_Media_Market_Discussion_Score'] = 0.0

        return sentiment_df

    def step_4_fetch_options_data(self):
        """收集/生成期权数据"""
        print("[Step 4] Fetching Options Data...")
        # 期权数据通常通过 'Synthetic Strike' 生成 (Strike = Round(Close, 5))
        options_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))
        return options_df  # 在 Feature Engineering 中会基于股价生成

    # -------------------------------------------------------
    # 模块 2: 数据合并与清洗 (Merge & Clean)
    # -------------------------------------------------------
    def step_5_merge_and_clean(self, df_market, df_macro, df_sentiment):
        print("\n[Step 5] Merging and Cleaning Data...")

        # 1. Left Join on Market Data (只保留交易日)
        df = df_market.join(df_macro, how='left')
        df = df.join(df_sentiment, how='left')

        # 2. 生成 Synthetic Option Strike (基于清洗前的股价)
        # 逻辑：寻找最近的 5 的倍数作为 ATM Strike
        df['CME_Option_Strike_Price'] = (df['JPMorgan_Stock_Price'] / 5).round() * 5

        # 3. Standardize Time & Outliers (IQR)
        # 按照 Clean data.py 的逻辑
        df = df.resample('D').asfreq()  # 扩展为日历日

        # IQR Outlier Removal (仅对非价格类指标)
        cols_to_check = ['JPMorgan_Stock_Volume', 'VIX_Volatility_Index']
        for col in cols_to_check:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                df.loc[mask, col] = np.nan  # 设为 NaN 等待插值

        # 4. Interpolation
        df = df.interpolate(method='time')
        df = df.ffill().bfill()

        # 只保留原始交易日 (可选，或者保留日历日)
        # 这里我们保留日历日以便计算 lag 特征

        df.to_csv(self.output_cleaned)
        print(f"   ✅ Cleaned data saved to {self.output_cleaned}")
        return df

    # -------------------------------------------------------
    # 模块 3: 特征工程 (Feature Engineering)
    # -------------------------------------------------------
    def step_6_feature_engineering(self, df):
        print("\n[Step 6] Feature Engineering (GARCH, Rolling)...")

        # 1. Returns
        df['Daily_Return'] = df['JPMorgan_Stock_Price'].pct_change()
        df['Log_Return'] = np.log(df['JPMorgan_Stock_Price'] / df['JPMorgan_Stock_Price'].shift(1))

        # 2. Rolling Volatility
        for w in [30, 60, 90]:
            df[f'Vol_Rolling_{w}D'] = df['Log_Return'].rolling(w).std() * np.sqrt(252)

        # 3. GARCH(1,1) Volatility (Manual Implementation)
        returns_clean = df['Log_Return'].dropna().values * 100

        def garch_likelihood(params, returns):
            omega, alpha, beta = params
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)
            if alpha + beta >= 1 or omega <= 0 or alpha < 0 or beta < 0: return 1e10
            likelihood = 0
            for t in range(1, n):
                sigma2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sigma2[t - 1]
                likelihood += 0.5 * (np.log(sigma2[t]) + (returns[t] ** 2) / sigma2[t])
            return likelihood

        res = minimize(garch_likelihood, [0.01, 0.1, 0.8], args=(returns_clean,),
                       method='L-BFGS-B', bounds=((1e-6, None), (0, 1), (0, 1)))

        # Reconstruct GARCH series
        omega, alpha, beta = res.x
        sigma2 = np.zeros(len(returns_clean))
        sigma2[0] = np.var(returns_clean)
        for t in range(1, len(returns_clean)):
            sigma2[t] = omega + alpha * (returns_clean[t - 1] ** 2) + beta * sigma2[t - 1]

        garch_vol = np.sqrt(sigma2) / 100 * np.sqrt(252)
        # Align
        df.loc[df.index[1:], 'GARCH_Volatility'] = garch_vol

        # 4. Momentum & Correlations
        df['VIX_Change'] = df['VIX_Volatility_Index'].diff()
        df['VIX_JPM_Corr_60D'] = df['Log_Return'].rolling(60).corr(df['VIX_Change'])

        df['Rate_Momentum_1Y_20D'] = df['US_Treasury_1Y_Rate'].diff(20)
        df['Fed_Rate_Momentum_20D'] = df['Fed_Interest_Rate'].diff(20)

        # 5. Dividend Growth (YoY)
        df['Implied_Div_Sum'] = (df['Dividend_Yield'] * df['JPMorgan_Stock_Price']).rolling(365).sum()
        df['Dividend_Growth_Rate_YoY'] = df['Implied_Div_Sum'].pct_change(365)

        # 6. Sentiment Moving Averages
        df['News_Sentiment_MA7'] = df['Financial_News_Sentiment'].rolling(7).mean()
        df['Social_Sentiment_MA7'] = df['Social_Media_Market_Discussion_Score'].rolling(7).mean()

        # Final Clean (Backfill NaN from rolling windows)
        df = df.drop(columns=['VIX_Change', 'Log_Return', 'Implied_Div_Sum'])
        df = df.bfill().ffill()

        df.to_csv(self.output_final)
        print(f"   ✅ Final feature set saved to {self.output_final}")
        print(f"   Columns: {len(df.columns)}")
        return df

    # -------------------------------------------------------
    # 主运行函数
    # -------------------------------------------------------
    def run(self):
        print(">>> Starting JPM Pipeline...")

        # Step 1-4: Collection
        df_market = self.step_1_fetch_market_data()
        df_macro = self.step_2_fetch_macro_data()
        df_sentiment = self.step_3_process_sentiment_data()

        # Step 5: Merge & Clean
        df_clean = self.step_5_merge_and_clean(df_market, df_macro, df_sentiment)

        # Step 6: Feature Engineering
        df_final = self.step_6_feature_engineering(df_clean)

        print("\n>>> Pipeline Completed Successfully!")
        print(df_final.tail())


if __name__ == "__main__":
    pipeline = JPMDataPipeline()
    pipeline.run()