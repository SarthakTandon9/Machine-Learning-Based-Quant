
import yfinance as yf
import ta
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from backtesting import Strategy, Backtest

"""# Gradient Boosting Model Training"""

START_DATE = '2010-01-01'
END_DATE = '2023-01-01'
ema_window = 20
rsi_window = 14
bb_window = 20
bb_dev = 1.5
atr_window = 14
mean_reversion_threshold = 0.02  # 2% threshold for mean reversion
zscore_window = 20
adx_window = 14
roc_window = 12
vwap_window = 14
cci_window = 20
wr_window = 14
keltner_window = 20
vma_window = 20
roc2_window = 6
macd_fast = 12
macd_slow = 26
macd_signal = 9
stoch_window = 14
stoch_smooth_window = 3
sar_acceleration = 0.02
sar_maximum = 0.2

def calculate_zscore(series, window):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    zscore = (series - mean) / std
    return zscore
def identify_hammer_candlestick(df):
    body = abs(df['Close'] - df['Open'])
    range_ = df['High'] - df['Low']
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    hammer = (
        (lower_shadow > 2 * body) &
        (upper_shadow < body * 0.3) &
        (body < 0.3 * range_)
    )
    return hammer.astype(int)

def identify_inverted_hammer_candlestick(df):
    body = abs(df['Close'] - df['Open'])
    range_ = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    inverted_hammer = (
        (upper_shadow > 2 * body) &
        (lower_shadow < body * 0.3) &
        (body < 0.3 * range_)
    )
    return inverted_hammer.astype(int)

def identify_doji_candlestick(df):
    body = abs(df['Close'] - df['Open'])
    range_ = df['High'] - df['Low']
    doji = body < (0.1 * range_)
    return doji.astype(int)

def identify_spinning_top_candlestick(df):
    body = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    range_ = df['High'] - df['Low']
    spinning_top = (
        (body < (0.3 * range_)) &
        (upper_shadow > (0.1 * range_)) &
        (lower_shadow > (0.1 * range_))
    )
    return spinning_top.astype(int)

def identify_bullish_engulfing(df):
    bullish = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    )
    return bullish.astype(int)

def identify_bearish_engulfing(df):
    bearish = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )
    return bearish.astype(int)

def identify_morning_star(df):
    first_candle = df['Close'].shift(2) < df['Open'].shift(2)
    second_candle_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
    second_candle_range = df['High'].shift(1) - df['Low'].shift(1)
    second_candle = second_candle_body <= 0.3 * second_candle_range
    third_candle = df['Close'] > df['Open']
    morning_star = first_candle & second_candle & third_candle
    return morning_star.astype(int)

def identify_evening_star(df):
    first_candle = df['Close'].shift(2) > df['Open'].shift(2)
    second_candle_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
    second_candle_range = df['High'].shift(1) - df['Low'].shift(1)
    second_candle = second_candle_body <= 0.3 * second_candle_range
    third_candle = df['Close'] < df['Open']
    evening_star = first_candle & second_candle & third_candle
    return evening_star.astype(int)

def identify_three_white_soldiers(df):
    soldiers = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Open'] > df['Open'].shift(1)) &
        (df['Open'].shift(1) > df['Open'].shift(2))
    )
    return soldiers.astype(int)

def identify_three_black_crows(df):
    crows = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Open'] < df['Open'].shift(1)) &
        (df['Open'].shift(1) < df['Open'].shift(2))
    )
    return crows.astype(int)

def prepare_data(tickers, start_date, end_date):
    buy_data_ls = []
    sell_data_ls = []

    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

        if data.empty:
            print(f"Warning: No data found for {ticker}. Skipping.")
            continue

        data = data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()

        data['EMA'] = ta.trend.EMAIndicator(data['Close'], window=ema_window).ema_indicator()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=cci_window).cci()
        data['Williams_%R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=wr_window).williams_r()
        bollinger = ta.volatility.BollingerBands(data['Close'], window=bb_window, window_dev=bb_dev)
        data['BB_High'] = bollinger.bollinger_hband()
        data['BB_Low'] = bollinger.bollinger_lband()

        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=keltner_window, window_atr=atr_window, fillna=True)
        data['KC_High'] = keltner.keltner_channel_hband()
        data['KC_Low'] = keltner.keltner_channel_lband()

        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=atr_window).average_true_range()
        data['Mean_Reversion'] = (data['Close'] - data['EMA']) / data['EMA']
        data['Z_Score'] = calculate_zscore(data['Close'], zscore_window)
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=adx_window).adx()
        data['ROC'] = ta.momentum.ROCIndicator(data['Close'], window=roc_window).roc()
        data['ROC2'] = ta.momentum.ROCIndicator(data['Close'], window=roc2_window).roc()
        data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume'], window=vwap_window).volume_weighted_average_price()
        data['VMA'] = ta.trend.SMAIndicator(data['Volume'], window=vma_window).sma_indicator()

        macd = ta.trend.MACD(data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()

        stoch = ta.momentum.StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=stoch_window,
            smooth_window=stoch_smooth_window
        )
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()

        data['SAR'] = ta.trend.PSARIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            step=sar_acceleration,
            max_step=sar_maximum
        ).psar()

        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

        data['Hammer'] = identify_hammer_candlestick(data)
        data['Inverted_Hammer'] = identify_inverted_hammer_candlestick(data)
        data['Doji'] = identify_doji_candlestick(data)
        data['Spinning_Top'] = identify_spinning_top_candlestick(data)
        data['Bullish_Engulfing'] = identify_bullish_engulfing(data)
        data['Bearish_Engulfing'] = identify_bearish_engulfing(data)
        data['Morning_Star'] = identify_morning_star(data)
        data['Evening_Star'] = identify_evening_star(data)
        data['Three_White_Soldiers'] = identify_three_white_soldiers(data)
        data['Three_Black_Crows'] = identify_three_black_crows(data)

        data['Prev_High'] = data['High'].shift(1)
        data['Prev_Low'] = data['Low'].shift(1)
        data['Prev_Close'] = data['Close'].shift(1)
        data['Prev_Volume'] = data['Volume'].shift(1)

        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['TR'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
        data['ATR_14'] = data['TR'].rolling(window=14).mean()
        data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
        data['Volatility_14'] = data['Close'].rolling(window=14).std()
        data['Daily_Return'] = data['Close'].pct_change()

        data['Bollinger_Success'] = np.where(
            (data['Close'] < data['BB_Low']) & (data['Close'].shift(-1) > data['BB_Low']),
            1,
            0
        )
        data['RSI_Success'] = np.where(
            ((data['RSI'] < 30) & (data['RSI'].shift(-1) > 30)) |
            ((data['RSI'] > 70) & (data['RSI'].shift(-1) < 70)),
            1,
            0
        )
        data['EMA_Success'] = np.where(
            ((data['Close'] > data['EMA']) & (data['Close'].shift(-1) > data['EMA'])) |
            ((data['Close'] < data['EMA']) & (data['Close'].shift(-1) < data['EMA'])),
            1,
            0
        )
        data['Mean_Reversion_Success'] = np.where(
            ((data['Mean_Reversion'] < -mean_reversion_threshold) & (data['Mean_Reversion'].shift(-1) > -mean_reversion_threshold)) |
            ((data['Mean_Reversion'] > mean_reversion_threshold) & (data['Mean_Reversion'].shift(-1) < mean_reversion_threshold)),
            1,
            0
        )
        data['Z_Score_Success'] = np.where(
            ((data['Z_Score'] < -2) & (data['Z_Score'].shift(-1) > -2)) |
            ((data['Z_Score'] > 2) & (data['Z_Score'].shift(-1) < 2)),
            1,
            0
        )
        data['ADX_Success'] = np.where(
            data['ADX'] < 25,
            1,
            0
        )
        data['ROC_Success'] = np.where(
            data['ROC'] > 0,
            1,
            0
        )
        data['VWAP_Success'] = np.where(
            data['Close'] > data['VWAP'],
            1,
            0
        )
        data['MACD_Success'] = np.where(
            (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)),
            1,
            0
        )
        data['Stoch_Success'] = np.where(
            (data['Stoch_K'] > data['Stoch_D']) & (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)),
            1,
            0
        )
        data['SAR_Success'] = np.where(
            (data['Close'] > data['SAR']) & (data['Close'].shift(1) <= data['SAR'].shift(1)),
            1,
            0
        )
        data['OBV_Success'] = np.where(
            data['OBV'] > data['OBV'].shift(1),
            1,
            0
        )
        data['Hammer_Success'] = data['Hammer']
        data['Inverted_Hammer_Success'] = data['Inverted_Hammer']
        data['Doji_Success'] = data['Doji']
        data['Spinning_Top_Success'] = data['Spinning_Top']
        data['Bullish_Engulfing_Success'] = data['Bullish_Engulfing']
        data['Bearish_Engulfing_Success'] = data['Bearish_Engulfing']
        data['Morning_Star_Success'] = data['Morning_Star']
        data['Evening_Star_Success'] = data['Evening_Star']
        data['Three_White_Soldiers_Success'] = data['Three_White_Soldiers']
        data['Three_Black_Crows_Success'] = data['Three_Black_Crows']

        bullish_indicators = [
            'Bollinger_Success',
            'RSI_Success',
            'EMA_Success',
            'Mean_Reversion_Success', #31.10
            'Z_Score_Success',
            'MACD_Success',
            'Stoch_Success',
            'SAR_Success',
            'OBV_Success',
            'Hammer_Success',
            'Bullish_Engulfing_Success',
            'Morning_Star_Success',
            'Three_White_Soldiers_Success'
        ]

        bearish_indicators = [
            'ADX_Success',
            'Bearish_Engulfing_Success',
            'Evening_Star_Success',
            'Three_Black_Crows_Success'
        ]

        data['Bullish_Signal'] = data[bullish_indicators].any(axis=1)
        data['Bearish_Signal'] = data[bearish_indicators].any(axis=1)

        data = data[~(data['Bullish_Signal'] & data['Bearish_Signal'])]

        buy_data = data[data['Bullish_Signal']].copy()
        sell_data = data[data['Bearish_Signal']].copy()

        buy_data['Target'] = (buy_data['Close'].shift(-1) > buy_data['Close']).astype(int)
        sell_data['Target'] = (sell_data['Close'].shift(-1) < sell_data['Close']).astype(int)

        buy_data.dropna(subset=['Target'], inplace=True)
        sell_data.dropna(subset=['Target'], inplace=True)
        buy_data.dropna(inplace=True)
        sell_data.dropna(inplace=True)
        buy_data['Ticker'] = ticker
        sell_data['Ticker'] = ticker

        buy_data_ls.append(buy_data)
        sell_data_ls.append(sell_data)

    combined_buy_df = pd.concat(buy_data_ls).reset_index()
    combined_sell_df = pd.concat(sell_data_ls).reset_index()
    return combined_buy_df, combined_sell_df

def train_and_save_model_buy(data, model_path='buy_model.joblib', scaler_path='buy_scaler.joblib', pca_path='buy_pca.joblib'):
    features = [
        'Bollinger_Success',
        'RSI_Success',
        'EMA_Success',
        'Mean_Reversion_Success',
        'Z_Score_Success',
        'MACD_Success',
        'Stoch_Success',
        'SAR_Success',
        'OBV_Success',
        'Hammer_Success',
        'Bullish_Engulfing_Success',
        'Morning_Star_Success',
        'Three_White_Soldiers_Success',
        'CCI',
        'Williams_%R',
        'VMA',
        'ROC2',
        'Prev_High',
        'Prev_Low',
        'Prev_Close',
        'Prev_Volume',
        'MA_5',
        'MA_10',
        'TR',
        'ATR_14',
        'Momentum_5',
        'Volatility_14',
        'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    scaler = StandardScaler()
    pca = PCA(n_components=15)
    X_scaled = scaler.fit_transform(X)

    X_pca = pca.fit_transform(X_scaled)

    gbc = GradientBoostingClassifier(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [5, 7, 9],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_pca, y)
    best_gbc = grid_search.best_estimator_

    joblib.dump(best_gbc, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    print(f"Best Model Parameters for Buy Model: {grid_search.best_params_}")
    print(f"Buy Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"PCA saved to {pca_path}")

    return best_gbc, scaler, pca

def train_and_save_model_sell(data, model_path='sell_model.joblib', scaler_path='sell_scaler.joblib', pca_path='sell_pca.joblib'):
    features = [
        'ADX_Success',
        'Bearish_Engulfing_Success',
        'Evening_Star_Success',
        'Three_Black_Crows_Success',
        'CCI',
        'Williams_%R',
        'VMA',
        'ROC2',
        'Prev_High',
        'Prev_Low',
        'Prev_Close',
        'Prev_Volume',
        'MA_5',
        'MA_10',
        'TR',
        'ATR_14',
        'Momentum_5',
        'Volatility_14',
        'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    scaler = StandardScaler()
    pca = PCA(n_components=10)

    X_scaled = scaler.fit_transform(X)

    X_pca = pca.fit_transform(X_scaled)

    gbc = GradientBoostingClassifier(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_pca, y)
    best_gbc = grid_search.best_estimator_

    joblib.dump(best_gbc, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    print(f"Best Model Parameters for Sell Model: {grid_search.best_params_}")
    print(f"Sell Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"PCA saved to {pca_path}")

    return best_gbc, scaler, pca

def evaluate_model_buy(model, scaler, pca, data):
    features = [
        'Bollinger_Success',
        'RSI_Success',
        'EMA_Success',
        'Mean_Reversion_Success',
        'Z_Score_Success',
        'MACD_Success',
        'Stoch_Success',
        'SAR_Success',
        'OBV_Success',
        'Hammer_Success',
        'Bullish_Engulfing_Success',
        'Morning_Star_Success',
        'Three_White_Soldiers_Success',
        'CCI',
        'Williams_%R',
        'VMA',
        'ROC2',
        'Prev_High',
        'Prev_Low',
        'Prev_Close',
        'Prev_Volume',
        'MA_5',
        'MA_10',
        'TR',
        'ATR_14',
        'Momentum_5',
        'Volatility_14',
        'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    X_train_scaled = scaler.transform(X_train)
    X_train_pca = pca.transform(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Buy Model Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Buy Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def evaluate_model_sell(model, scaler, pca, data):
    """Evaluate the Sell model."""
    features = [
        'ADX_Success',
        'Bearish_Engulfing_Success',
        'Evening_Star_Success',
        'Three_Black_Crows_Success',
        'CCI',
        'Williams_%R',
        'VMA',
        'ROC2',
        'Prev_High',
        'Prev_Low',
        'Prev_Close',
        'Prev_Volume',
        'MA_5',
        'MA_10',
        'TR',
        'ATR_14',
        'Momentum_5',
        'Volatility_14',
        'Daily_Return'
    ]

    X = data[features]
    y = data['Target']

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    X_train_scaled = scaler.transform(X_train)
    X_train_pca = pca.transform(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Sell Model Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Sell Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

TICKER = ['AAPL', 'MSFT', 'NFLX', 'QQQ', 'SPY']
START_DATE = '2010-01-01'
END_DATE = '2019-12-31'

buy_data, sell_data = prepare_data(TICKER, START_DATE, END_DATE)

if len(buy_data) > 100:
    buy_model, buy_scaler, buy_pca = train_and_save_model_buy(buy_data)
    evaluate_model_buy(buy_model, buy_scaler, buy_pca, buy_data)
else:
    print("Not enough Buy signals to train the Buy model.")

if len(sell_data) > 100:
    sell_model, sell_scaler, sell_pca = train_and_save_model_sell(sell_data)
    evaluate_model_sell(sell_model, sell_scaler, sell_pca, sell_data)
else:
    print("Not enough Sell signals to train the Sell model.")

""" Developing and Backtesting the Strategy"""

# Data preparation function
def prepare_data(data):
    df = data.copy()
    ema_window = 20
    rsi_window = 14
    cci_window = 20
    wr_window = 14
    bb_window = 20
    bb_dev = 2
    atr_window = 14
    zscore_window = 20
    adx_window = 14
    roc_window = 12
    roc2_window = 6
    vwap_window = 14
    vma_window = 20
    macd_slow = 26
    macd_fast = 12
    macd_signal = 9
    stoch_window = 14
    stoch_smooth_window = 3
    sar_acceleration = 0.02
    sar_maximum = 0.2
    mean_reversion_threshold = 0.05

    df['EMA'] = ta.trend.EMAIndicator(df['Close'], window=ema_window).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=cci_window).cci()
    df['Williams_%R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=wr_window).williams_r()
    bollinger = ta.volatility.BollingerBands(df['Close'], window=bb_window, window_dev=bb_dev)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()

    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_window).average_true_range()
    df['ATR_14'] = df['ATR'] 
    df['Mean_Reversion'] = (df['Close'] - df['EMA']) / df['EMA']
    df['Z_Score'] = calculate_zscore(df['Close'], zscore_window)
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=adx_window).adx()
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=roc_window).roc()
    df['ROC2'] = ta.momentum.ROCIndicator(df['Close'], window=roc2_window).roc()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=vwap_window).volume_weighted_average_price()
    df['VMA'] = ta.trend.SMAIndicator(df['Volume'], window=vma_window).sma_indicator()
    macd = ta.trend.MACD(df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=stoch_window,
        smooth_window=stoch_smooth_window
    )
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df['SAR'] = ta.trend.PSARIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        step=sar_acceleration,
        max_step=sar_maximum
    ).psar()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['Hammer'] = identify_hammer_candlestick(df)
    df['Inverted_Hammer'] = identify_inverted_hammer_candlestick(df)
    df['Doji'] = identify_doji_candlestick(df)
    df['Spinning_Top'] = identify_spinning_top_candlestick(df)
    df['Bullish_Engulfing'] = identify_bullish_engulfing(df)
    df['Bearish_Engulfing'] = identify_bearish_engulfing(df)
    df['Morning_Star'] = identify_morning_star(df)
    df['Evening_Star'] = identify_evening_star(df)
    df['Three_White_Soldiers'] = identify_three_white_soldiers(df)
    df['Three_Black_Crows'] = identify_three_black_crows(df)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['TR'] = df[['High', 'Low', 'Close']].apply(
        lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Volatility_14'] = df['Close'].rolling(window=14).std()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Bollinger_Success'] = np.where(
        (df['Close'] < df['BB_Low']) & (df['Close'].shift(-1) > df['BB_Low']), 1, 0)
    df['RSI_Success'] = np.where(
        ((df['RSI'] < 30) & (df['RSI'].shift(-1) > 30)) |
        ((df['RSI'] > 70) & (df['RSI'].shift(-1) < 70)), 1, 0)
    df['EMA_Success'] = np.where(
        ((df['Close'] > df['EMA']) & (df['Close'].shift(-1) > df['EMA'])) |
        ((df['Close'] < df['EMA']) & (df['Close'].shift(-1) < df['EMA'])), 1, 0)
    df['Mean_Reversion_Success'] = np.where(
        ((df['Mean_Reversion'] < -mean_reversion_threshold) & (df['Mean_Reversion'].shift(-1) > -mean_reversion_threshold)) |
        ((df['Mean_Reversion'] > mean_reversion_threshold) & (df['Mean_Reversion'].shift(-1) < mean_reversion_threshold)), 1, 0)
    df['Z_Score_Success'] = np.where(
        ((df['Z_Score'] < -2) & (df['Z_Score'].shift(-1) > -2)) |
        ((df['Z_Score'] > 2) & (df['Z_Score'].shift(-1) < 2)), 1, 0)
    df['ADX_Success'] = np.where(df['ADX'] < 25, 1, 0)
    df['ROC_Success'] = np.where(df['ROC'] > 0, 1, 0)
    df['VWAP_Success'] = np.where(df['Close'] > df['VWAP'], 1, 0)
    df['MACD_Success'] = np.where(
        (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 1, 0)
    df['Stoch_Success'] = np.where(
        (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)), 1, 0)
    df['SAR_Success'] = np.where(
        (df['Close'] > df['SAR']) & (df['Close'].shift(1) <= df['SAR'].shift(1)), 1, 0)
    df['OBV_Success'] = np.where(df['OBV'] > df['OBV'].shift(1), 1, 0)
    df[['Hammer_Success', 'Bullish_Engulfing_Success', 'Morning_Star_Success', 'Three_White_Soldiers_Success']]  = \
                              df[['Hammer', 'Bullish_Engulfing', 'Morning_Star', 'Three_White_Soldiers']]
    df[['Bearish_Engulfing_Success', 'Evening_Star_Success', 'Three_Black_Crows_Success']] =\
                              df[['Bearish_Engulfing', 'Evening_Star', 'Three_Black_Crows']]

    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    return df

class MLStrategy(Strategy):
    def init(self):
        self.buy_model = joblib.load('buy_model.joblib')
        self.buy_scaler = joblib.load('buy_scaler.joblib')
        self.buy_pca = joblib.load('buy_pca.joblib')

        self.sell_model = joblib.load('sell_model.joblib')
        self.sell_scaler = joblib.load('sell_scaler.joblib')
        self.sell_pca = joblib.load('sell_pca.joblib')
        df = self.data.df
        self.df = prepare_data(df)
        self.df['Mean'] = self.df['Close'].rolling(window=10).mean()
        self.df['Mean_Deviation'] = (self.df['Close'] - self.df['Mean']) / self.df['Mean']

        self.open_trades = []
        self.my_closed_trades = []

        self.initial_equity = self.equity
        self.max_drawdown = 0
        self.drawdown_flag = False
        self.drawdown_start_bar = None
        self.current_drawdown_level = None

        self.drawdown_levels = {
            0.03: 'resize',    # 3% drawdown
            0.05: 'resize',    # 5% drawdown
            0.07: 'exit_halt'  # 7% drawdown
        }

        self.max_risk_per_trade = 0.01  # 1% of current equity
        self.max_cumulative_risk = 0.5  

        self.base_holding_period = 1  # Base holding period
        self.trade_history = []
        self.win_rate = 0.5
        self.payout_ratio = 2
        self.trailing_stop_factor = 0.5  # Multiplier for ATR to set trailing stops

    def on_trade_open(self, trade):
        entry_bar = len(self.data) - 1
        self.open_trades.append({
            'index': entry_bar,
            'type': 'long' if trade.is_long else 'short',
            'trade': trade,
            'min_holding': self.calculate_holding_period(entry_bar),
            'initial_stop': trade.sl
        })

    def on_trade_close(self, trade):
        self.my_closed_trades.append(trade)
        profit = trade.profit
        self.trade_history.append(profit)
        self.update_kelly_parameters()

    def update_kelly_parameters(self):
        if len(self.trade_history) < 30:
            return
        wins = [p for p in self.trade_history if p > 0]
        losses = [p for p in self.trade_history if p <= 0]
        if len(wins) == 0 or len(losses) == 0:
            return
        self.win_rate = len(wins) / len(self.trade_history)
        average_win = np.mean(wins)
        average_loss = np.mean([abs(p) for p in losses])
        if average_loss == 0:
            self.payout_ratio = 2
        else:
            self.payout_ratio = average_win / average_loss

    def calculate_kelly_fraction(self):
        """
        Calculate the Kelly fraction based on current win rate and payout ratio.
        """
        if self.payout_ratio == 0:
            return 0
        kelly = self.win_rate - (1 - self.win_rate) / self.payout_ratio
        return max(kelly, 0)  # Ensure non-negative

    def calculate_holding_period(self, index):
        """
        Calculate the holding period based on ADX and ATR indicators.
        """
        df = self.df
        adx = df.iloc[index]['ADX']
        atr = df.iloc[index]['ATR_14']
        atr_ma = df['ATR_14'].rolling(window=14).mean().iloc[index]
        atr_normalized = atr / atr_ma if atr_ma != 0 else 1
        volatility_indicator = atr_normalized if atr_normalized != 0 else 1
        trend_strength_indicator = adx if adx != 0 else 1
        holding_period = self.base_holding_period * (1 + trend_strength_indicator / volatility_indicator)

        holding_period = max(self.base_holding_period, int(holding_period))

        return holding_period

    def handle_drawdown(self, severity):
        if severity == 'resize':
            self.max_risk_per_trade *= 0.5 
        elif severity == 'exit_halt':
            self.close_all_positions()
            self.drawdown_flag = True
            self.drawdown_start_bar = len(self.data) - 1

    def close_all_positions(self):
        for trade_info in self.open_trades:
            trade = trade_info['trade']
            if not trade.is_closed:
                trade.close()
        self.open_trades = []

    def next(self):
        i = len(self.data) - 1
        equity = self.equity
        current_drawdown = (self.initial_equity - equity) / self.initial_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        for level, action in sorted(self.drawdown_levels.items()):
            if current_drawdown >= level and (self.current_drawdown_level is None or level > self.current_drawdown_level):
                self.current_drawdown_level = level
                self.handle_drawdown(action)
                break

        if self.drawdown_flag:
            bars_since_drawdown = i - self.drawdown_start_bar
            if bars_since_drawdown >= 2:
                self.drawdown_flag = False
                self.current_drawdown_level = None
                self.max_drawdown = 0
                self.max_risk_per_trade = 0.01
            else:
                return

        total_risk = 0
        for t in self.open_trades:
            trade = t['trade']
            if trade.is_long:
                potential_loss = t['trade'].entry_price - t['trade'].sl
            else: #9.07.05
                potential_loss = t['trade'].sl - t['trade'].entry_price
            total_risk += (trade.size * potential_loss) / self.equity

        if total_risk >= self.max_cumulative_risk:
            return

        df = self.df
        buy_features = [
            'Bollinger_Success', 'RSI_Success', 'EMA_Success', 'Mean_Reversion_Success',
            'Z_Score_Success', 'MACD_Success', 'Stoch_Success', 'SAR_Success',
            'OBV_Success', 'Hammer_Success', 'Bullish_Engulfing_Success',
            'Morning_Star_Success', 'Three_White_Soldiers_Success', 'CCI', 'Williams_%R',
            'VMA', 'ROC2', 'Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume',
            'MA_5', 'MA_10', 'TR', 'ATR_14', 'Momentum_5', 'Volatility_14', 'Daily_Return'
        ]

        sell_features = [
            'ADX_Success', 'Bearish_Engulfing_Success', 'Evening_Star_Success',
            'Three_Black_Crows_Success', 'CCI', 'Williams_%R', 'VMA', 'ROC2',
            'Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume', 'MA_5', 'MA_10',
            'TR', 'ATR_14', 'Momentum_5', 'Volatility_14', 'Daily_Return'
        ]

        if i < max(self.buy_pca.n_components_, self.sell_pca.n_components_):
            return

        X_buy = df.iloc[[i]][buy_features]
        X_sell = df.iloc[[i]][sell_features]

        if X_buy.isnull().values.any() or X_sell.isnull().values.any():
            return

        X_buy_scaled = self.buy_scaler.transform(X_buy)
        X_buy_pca = self.buy_pca.transform(X_buy_scaled)
        X_sell_scaled = self.sell_scaler.transform(X_sell)
        X_sell_pca = self.sell_pca.transform(X_sell_scaled)

        buy_proba = self.buy_model.predict_proba(X_buy_pca)[0][1]
        sell_proba = self.sell_model.predict_proba(X_sell_pca)[0][1]

        buy_threshold = 0.65
        sell_threshold = 0.65

        kelly_fraction = self.calculate_kelly_fraction()

        cash = self.equity
        risk_per_trade = self.max_risk_per_trade * cash * kelly_fraction

        atr = df.iloc[i]['ATR_14']

        take_profit_multiplier = 2

        if buy_proba > buy_threshold:
            entry_price = self.data.Close[-1]
            stop_loss_price = entry_price - self.trailing_stop_factor * atr
            take_profit_price = entry_price + take_profit_multiplier * atr
            stop_loss_distance = entry_price - stop_loss_price
            if stop_loss_distance == 0:
                position_size = 0
            else:
                position_size = (risk_per_trade / stop_loss_distance) * 15

            max_position_size = cash / entry_price
            position_size = min(position_size, max_position_size)
            if position_size > 0:
                self.buy(size=int(position_size), sl=stop_loss_price, tp=take_profit_price)
        mean_deviation = df.iloc[i]['Mean_Deviation']
        mean_reversion_threshold = -0.03

        if sell_proba > sell_threshold:
            entry_price = self.data.Close[-1]
            stop_loss_price = entry_price + self.trailing_stop_factor * atr
            take_profit_price = entry_price - take_profit_multiplier * atr
            stop_loss_distance = stop_loss_price - entry_price
            if stop_loss_distance == 0:
                position_size = 0
            else:
                position_size = risk_per_trade / stop_loss_distance
            max_position_size = cash / entry_price
            position_size = min(position_size, max_position_size)

            if int(position_size) > 0:
                self.sell(size=int(position_size), sl=stop_loss_price, tp=take_profit_price)
        self.open_trades = [t for t in self.open_trades if not t['trade'].is_closed]

        for trade_info in self.open_trades:
            trade = trade_info['trade']
            entry_index = trade_info['index']
            holding_period = i - entry_index

            if holding_period >= trade_info['min_holding']:
                is_profitable = trade.profit > 0
                if is_profitable:
                    new_holding_period = self.calculate_holding_period(i)
                    trade_info['min_holding'] = new_holding_period
                if trade.is_long:
                    X_buy_current = df.iloc[[i]][buy_features]
                    if X_buy_current.isnull().values.any():
                        continue
                    X_buy_scaled_current = self.buy_scaler.transform(X_buy_current)
                    X_buy_pca_current = self.buy_pca.transform(X_buy_scaled_current)
                    buy_proba_current = self.buy_model.predict_proba(X_buy_pca_current)[0][1]

                    if buy_proba_current < buy_threshold:
                        trade.close()
                    else:
                        new_stop = max(trade.sl, self.data.Close[-1] - self.trailing_stop_factor * atr)
                        if new_stop > trade.sl:
                            trade.sl = new_stop
                        new_tp = self.data.Close[-1] + take_profit_multiplier * atr
                        if new_tp > trade.tp:
                            trade.tp = new_tp
                elif trade.is_short:
                    X_sell_current = df.iloc[[i]][sell_features]
                    if X_sell_current.isnull().values.any():
                        continue
                    X_sell_scaled_current = self.sell_scaler.transform(X_sell_current)
                    X_sell_pca_current = self.sell_pca.transform(X_sell_scaled_current)
                    sell_proba_current = self.sell_model.predict_proba(X_sell_pca_current)[0][1]

                    if sell_proba_current < sell_threshold:
                        trade.close()
                    else:
                        new_stop = min(trade.sl, self.data.Close[-1] + self.trailing_stop_factor * atr)
                        if new_stop < trade.sl:
                            trade.sl = new_stop
                        new_tp = self.data.Close[-1] - take_profit_multiplier * atr
                        if new_tp < trade.tp:
                            trade.tp = new_tp

for ticker in ['AAPL', 'MSFT', 'NFLX', 'QQQ', 'SPY']:
    start_date = '2023-09-01'
    end_date = '2024-09-30'

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    bt = Backtest(data, MLStrategy, cash=10000, commission=0.001, exclusive_orders=False)
    stats = bt.run()
    bt.plot()

    print(stats)

