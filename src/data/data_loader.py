import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataLoader:
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                if not data.empty:
                    self.data[symbol] = data
                    print(f"Successfully fetched data for {symbol}")
                else:
                    print(f"No data found for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return self.data
    
    def calculate_technical_indicators(self, symbol: str, window_short: int = 12, window_long: int = 26) -> pd.DataFrame:
        if symbol not in self.data:
            raise ValueError(f"No data available for {symbol}")
        
        df = self.data[symbol].copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_Short'] = df['Close'].rolling(window=window_short).mean()
        df['SMA_Long'] = df['Close'].rolling(window=window_long).mean()
        df['EMA_Short'] = df['Close'].ewm(span=window_short).mean()
        df['EMA_Long'] = df['Close'].ewm(span=window_long).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_Short'] - df['EMA_Long']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def get_feature_matrix(self, lookback_window: int = 30) -> Tuple[np.ndarray, List[str]]:
        feature_columns = [
            'Returns', 'Log_Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Signal',
            'Volume_Ratio', 'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long',
            'BB_Upper', 'BB_Lower'
        ]
        
        all_features = []
        valid_symbols = []
        
        for symbol in self.symbols:
            if symbol in self.data:
                df = self.calculate_technical_indicators(symbol)
                
                # Normalize features
                feature_df = df[feature_columns].dropna()
                if len(feature_df) >= lookback_window:
                    # Z-score normalization
                    normalized_features = (feature_df - feature_df.mean()) / feature_df.std()
                    all_features.append(normalized_features.values)
                    valid_symbols.append(symbol)
        
        return np.array(all_features), valid_symbols
    
    def create_sequences(self, features: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        for i in range(len(features)):
            stock_features = features[i]
            for j in range(sequence_length, len(stock_features)):
                X.append(stock_features[j-sequence_length:j])
                # Target: next day's return
                if j < len(stock_features):
                    y.append(stock_features[j, 0])  # Returns column
        
        return np.array(X), np.array(y)

class MarketDataPreprocessor:
    def __init__(self):
        self.scaler = None
        
    def preprocess_data(self, data_loader: StockDataLoader, sequence_length: int = 30) -> Dict:
        features, symbols = data_loader.get_feature_matrix()
        
        if len(features) == 0:
            raise ValueError("No valid features extracted from data")
        
        # Create sequences for time series prediction
        all_X, all_y = [], []
        
        for i, symbol in enumerate(symbols):
            X_seq, y_seq = data_loader.create_sequences(features[i:i+1], sequence_length)
            if len(X_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)
        
        if not all_X:
            raise ValueError("No valid sequences created")
        
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        
        return {
            'X_train': X[:split_idx],
            'X_test': X[split_idx:],
            'y_train': y[:split_idx],
            'y_test': y[split_idx:],
            'symbols': symbols,
            'features': features
        }