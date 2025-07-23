import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class Trade:
    symbol: str
    action: Action
    quantity: float
    price: float
    timestamp: pd.Timestamp
    confidence: float = 0.0
    uncertainty: float = 0.0

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    total_value: float = 0.0
    
    def update_value(self, prices: Dict[str, float]):
        position_value = sum(qty * prices.get(symbol, 0) for symbol, qty in self.positions.items())
        self.total_value = self.cash + position_value

class TradingStrategy(ABC):
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio = Portfolio(cash=initial_capital, positions={})
        
    @abstractmethod
    def generate_signals(self, predictions: Dict, uncertainty: Dict, 
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        pass

class BayesianMomentumStrategy(TradingStrategy):
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 confidence_threshold: float = 0.6, uncertainty_penalty: float = 2.0,
                 max_position_size: float = 0.1):
        super().__init__(initial_capital, transaction_cost)
        self.confidence_threshold = confidence_threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.max_position_size = max_position_size
        
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        for symbol in predictions:
            if symbol not in current_prices:
                continue
                
            pred_return = predictions[symbol]
            pred_uncertainty = uncertainty.get(symbol, 1.0)
            current_price = current_prices[symbol]
            
            # Calculate confidence-adjusted signal
            confidence = self._calculate_confidence(pred_return, pred_uncertainty)
            
            if confidence > self.confidence_threshold:
                # Determine action based on predicted return
                if pred_return > 0.02:  # Buy threshold: 2% expected return
                    action = Action.BUY
                    signal_strength = min(pred_return * confidence, 1.0)
                elif pred_return < -0.02:  # Sell threshold: -2% expected return  
                    action = Action.SELL
                    signal_strength = min(abs(pred_return) * confidence, 1.0)
                else:
                    action = Action.HOLD
                    signal_strength = 0.0
                
                if action != Action.HOLD:
                    quantity = self.calculate_position_size(
                        signal_strength, pred_uncertainty, 
                        portfolio.cash, current_price
                    )
                    
                    if quantity > 0:
                        trade = Trade(
                            symbol=symbol,
                            action=action,
                            quantity=quantity,
                            price=current_price,
                            timestamp=pd.Timestamp.now(),
                            confidence=confidence,
                            uncertainty=pred_uncertainty
                        )
                        trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        # Kelly Criterion with uncertainty adjustment
        win_prob = 0.5 + signal_strength * 0.3  # Adjust based on signal strength
        avg_win = signal_strength * 0.1  # Expected return
        avg_loss = 0.05  # Expected loss
        
        # Kelly fraction
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Adjust for uncertainty (reduce position size with higher uncertainty)
        uncertainty_adjustment = 1.0 / (1.0 + uncertainty * self.uncertainty_penalty)
        adjusted_fraction = kelly_fraction * uncertainty_adjustment
        
        # Cap at maximum position size
        position_fraction = min(adjusted_fraction, self.max_position_size)
        position_fraction = max(position_fraction, 0.01)  # Minimum position size
        
        # Calculate quantity
        position_value = available_capital * position_fraction
        quantity = position_value / current_price
        
        return quantity
    
    def _calculate_confidence(self, prediction: float, uncertainty: float) -> float:
        # Confidence inversely related to uncertainty
        base_confidence = abs(prediction) / (abs(prediction) + 0.01)  # Normalize prediction
        uncertainty_discount = 1.0 / (1.0 + uncertainty)
        return base_confidence * uncertainty_discount

class BayesianMeanReversionStrategy(TradingStrategy):
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 lookback_window: int = 20, deviation_threshold: float = 2.0):
        super().__init__(initial_capital, transaction_cost)
        self.lookback_window = lookback_window
        self.deviation_threshold = deviation_threshold
        self.price_history = {}
        
    def update_price_history(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback_window:
            self.price_history[symbol].pop(0)
    
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        for symbol in predictions:
            if symbol not in current_prices or symbol not in self.price_history:
                continue
            
            if len(self.price_history[symbol]) < self.lookback_window:
                continue
                
            current_price = current_prices[symbol]
            prices = np.array(self.price_history[symbol])
            
            # Calculate mean and standard deviation
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price > 0:
                # Z-score
                z_score = (current_price - mean_price) / std_price
                pred_uncertainty = uncertainty.get(symbol, 1.0)
                
                # Generate mean reversion signal
                if z_score > self.deviation_threshold:
                    # Price is too high, expect reversion (sell)
                    action = Action.SELL
                    signal_strength = min(abs(z_score) / self.deviation_threshold, 1.0)
                elif z_score < -self.deviation_threshold:
                    # Price is too low, expect reversion (buy)
                    action = Action.BUY
                    signal_strength = min(abs(z_score) / self.deviation_threshold, 1.0)
                else:
                    action = Action.HOLD
                    signal_strength = 0.0
                
                if action != Action.HOLD:
                    quantity = self.calculate_position_size(
                        signal_strength, pred_uncertainty,
                        portfolio.cash, current_price
                    )
                    
                    if quantity > 0:
                        trade = Trade(
                            symbol=symbol,
                            action=action,
                            quantity=quantity,
                            price=current_price,
                            timestamp=pd.Timestamp.now(),
                            confidence=signal_strength,
                            uncertainty=pred_uncertainty
                        )
                        trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        # Simple position sizing based on signal strength and uncertainty
        base_fraction = 0.05  # 5% base position
        signal_multiplier = signal_strength * 2.0
        uncertainty_discount = 1.0 / (1.0 + uncertainty)
        
        position_fraction = base_fraction * signal_multiplier * uncertainty_discount
        position_fraction = min(position_fraction, 0.15)  # Max 15% position
        
        position_value = available_capital * position_fraction
        quantity = position_value / current_price
        
        return quantity

class RiskParityStrategy(TradingStrategy):
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 target_volatility: float = 0.15, rebalance_threshold: float = 0.05):
        super().__init__(initial_capital, transaction_cost)
        self.target_volatility = target_volatility
        self.rebalance_threshold = rebalance_threshold
        
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        # Calculate risk contributions and target weights
        symbols = list(predictions.keys())
        volatilities = np.array([uncertainty.get(symbol, 0.1) for symbol in symbols])
        
        # Inverse volatility weighting
        inv_vol = 1.0 / volatilities
        target_weights = inv_vol / np.sum(inv_vol)
        
        # Current weights
        current_weights = self._calculate_current_weights(symbols, current_prices, portfolio)
        
        # Generate rebalancing trades
        for i, symbol in enumerate(symbols):
            weight_diff = target_weights[i] - current_weights[i]
            
            if abs(weight_diff) > self.rebalance_threshold:
                current_price = current_prices[symbol]
                target_value = portfolio.total_value * target_weights[i]
                current_value = portfolio.positions.get(symbol, 0) * current_price
                
                trade_value = target_value - current_value
                quantity = abs(trade_value) / current_price
                
                if trade_value > 0:
                    action = Action.BUY
                else:
                    action = Action.SELL
                
                if quantity > 0:
                    trade = Trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=current_price,
                        timestamp=pd.Timestamp.now(),
                        confidence=0.8,  # High confidence for risk parity
                        uncertainty=uncertainty.get(symbol, 0.1)
                    )
                    trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        # Position size based on inverse volatility
        volatility_adjustment = self.target_volatility / max(uncertainty, 0.01)
        position_fraction = min(volatility_adjustment * 0.1, 0.2)
        
        position_value = available_capital * position_fraction
        quantity = position_value / current_price
        
        return quantity
    
    def _calculate_current_weights(self, symbols: List[str], current_prices: Dict[str, float],
                                 portfolio: Portfolio) -> np.ndarray:
        weights = []
        total_value = portfolio.total_value
        
        for symbol in symbols:
            position_value = portfolio.positions.get(symbol, 0) * current_prices.get(symbol, 0)
            weight = position_value / total_value if total_value > 0 else 0
            weights.append(weight)
        
        return np.array(weights)

class StrategyOptimizer:
    def __init__(self, strategies: List[TradingStrategy]):
        self.strategies = strategies
        self.performance_history = {i: [] for i in range(len(strategies))}
        
    def optimize_allocation(self, lookback_period: int = 252) -> np.ndarray:
        # Calculate Sharpe ratios for each strategy
        sharpe_ratios = []
        
        for i, strategy in enumerate(self.strategies):
            if len(self.performance_history[i]) >= lookback_period:
                returns = np.array(self.performance_history[i][-lookback_period:])
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
            sharpe_ratios.append(max(sharpe, 0))
        
        # Normalize to get allocation weights
        total_sharpe = sum(sharpe_ratios)
        if total_sharpe > 0:
            weights = np.array(sharpe_ratios) / total_sharpe
        else:
            weights = np.ones(len(self.strategies)) / len(self.strategies)
        
        return weights
    
    def update_performance(self, strategy_index: int, return_value: float):
        self.performance_history[strategy_index].append(return_value)

# Simple baseline strategies for comparison
class BuyAndHoldStrategy(TradingStrategy):
    """Simple buy-and-hold strategy"""
    
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        super().__init__(initial_capital, transaction_cost)
        self.initial_positions_set = False
        
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        # Only buy at the beginning, then hold
        if not self.initial_positions_set:
            n_symbols = len(current_prices)
            if n_symbols > 0:
                position_size = portfolio.cash / n_symbols
                
                for symbol, price in current_prices.items():
                    if position_size > price:
                        quantity = position_size / price
                        trade = Trade(
                            symbol=symbol,
                            action=Action.BUY,
                            quantity=quantity,
                            price=price,
                            timestamp=pd.Timestamp.now(),
                            confidence=1.0,
                            uncertainty=0.0
                        )
                        trades.append(trade)
                
                self.initial_positions_set = True
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        return available_capital / current_price

class SimpleMovingAverageStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 short_window: int = 20, long_window: int = 50):
        super().__init__(initial_capital, transaction_cost)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {}
        
    def update_price_history(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.long_window:
            self.price_history[symbol].pop(0)
    
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        for symbol in current_prices:
            current_price = current_prices[symbol]
            
            # Update price history
            self.update_price_history(symbol, current_price)
            
            if len(self.price_history[symbol]) >= self.long_window:
                prices = np.array(self.price_history[symbol])
                
                short_ma = np.mean(prices[-self.short_window:])
                long_ma = np.mean(prices[-self.long_window:])
                prev_short_ma = np.mean(prices[-self.short_window-1:-1])
                prev_long_ma = np.mean(prices[-self.long_window-1:-1])
                
                # Check for crossover
                current_position = portfolio.positions.get(symbol, 0)
                
                # Golden cross: short MA crosses above long MA (buy signal)
                if prev_short_ma <= prev_long_ma and short_ma > long_ma and current_position == 0:
                    quantity = self.calculate_position_size(1.0, 0.1, portfolio.cash, current_price)
                    if quantity > 0:
                        trade = Trade(
                            symbol=symbol,
                            action=Action.BUY,
                            quantity=quantity,
                            price=current_price,
                            timestamp=pd.Timestamp.now(),
                            confidence=0.7,
                            uncertainty=0.1
                        )
                        trades.append(trade)
                
                # Death cross: short MA crosses below long MA (sell signal)
                elif prev_short_ma >= prev_long_ma and short_ma < long_ma and current_position > 0:
                    trade = Trade(
                        symbol=symbol,
                        action=Action.SELL,
                        quantity=current_position,
                        price=current_price,
                        timestamp=pd.Timestamp.now(),
                        confidence=0.7,
                        uncertainty=0.1
                    )
                    trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        position_fraction = 0.1  # 10% of available capital per position
        position_value = available_capital * position_fraction
        return position_value / current_price

class RSIStrategy(TradingStrategy):
    """Simple RSI-based strategy"""
    
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        super().__init__(initial_capital, transaction_cost)
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.price_history = {}
        
    def update_price_history(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.rsi_period + 1:
            self.price_history[symbol].pop(0)
    
    def calculate_rsi(self, prices: List[float]) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        for symbol in current_prices:
            current_price = current_prices[symbol]
            
            # Update price history
            self.update_price_history(symbol, current_price)
            
            if len(self.price_history[symbol]) >= self.rsi_period + 1:
                rsi = self.calculate_rsi(self.price_history[symbol])
                current_position = portfolio.positions.get(symbol, 0)
                
                # Buy when oversold
                if rsi < self.oversold_threshold and current_position == 0:
                    quantity = self.calculate_position_size(1.0, 0.2, portfolio.cash, current_price)
                    if quantity > 0:
                        trade = Trade(
                            symbol=symbol,
                            action=Action.BUY,
                            quantity=quantity,
                            price=current_price,
                            timestamp=pd.Timestamp.now(),
                            confidence=0.6,
                            uncertainty=0.2
                        )
                        trades.append(trade)
                
                # Sell when overbought
                elif rsi > self.overbought_threshold and current_position > 0:
                    trade = Trade(
                        symbol=symbol,
                        action=Action.SELL,
                        quantity=current_position,
                        price=current_price,
                        timestamp=pd.Timestamp.now(),
                        confidence=0.6,
                        uncertainty=0.2
                    )
                    trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        position_fraction = 0.15  # 15% of available capital per position
        position_value = available_capital * position_fraction
        return position_value / current_price

class RandomStrategy(TradingStrategy):
    """Random trading strategy for comparison baseline"""
    
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001,
                 trade_probability: float = 0.05):
        super().__init__(initial_capital, transaction_cost)
        self.trade_probability = trade_probability
        np.random.seed(42)  # For reproducible results
        
    def generate_signals(self, predictions: Dict, uncertainty: Dict,
                        current_prices: Dict, portfolio: Portfolio) -> List[Trade]:
        trades = []
        
        for symbol in current_prices:
            if np.random.random() < self.trade_probability:
                current_price = current_prices[symbol]
                current_position = portfolio.positions.get(symbol, 0)
                
                # Random action
                action_choice = np.random.choice(['buy', 'sell', 'hold'], p=[0.4, 0.4, 0.2])
                
                if action_choice == 'buy' and portfolio.cash > current_price * 10:
                    quantity = self.calculate_position_size(0.5, 0.5, portfolio.cash, current_price)
                    if quantity > 0:
                        trade = Trade(
                            symbol=symbol,
                            action=Action.BUY,
                            quantity=quantity,
                            price=current_price,
                            timestamp=pd.Timestamp.now(),
                            confidence=0.5,
                            uncertainty=0.5
                        )
                        trades.append(trade)
                
                elif action_choice == 'sell' and current_position > 0:
                    quantity = min(current_position, current_position * np.random.uniform(0.1, 1.0))
                    trade = Trade(
                        symbol=symbol,
                        action=Action.SELL,
                        quantity=quantity,
                        price=current_price,
                        timestamp=pd.Timestamp.now(),
                        confidence=0.5,
                        uncertainty=0.5
                    )
                    trades.append(trade)
        
        return trades
    
    def calculate_position_size(self, signal_strength: float, uncertainty: float,
                              available_capital: float, current_price: float) -> float:
        position_fraction = np.random.uniform(0.05, 0.15)  # Random position size
        position_value = available_capital * position_fraction
        return position_value / current_price