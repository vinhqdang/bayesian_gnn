import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

from .strategy import Trade, Action, Portfolio

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional Value at Risk (95%)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation: float

class RiskManager:
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02,
                 max_correlation: float = 0.7,
                 stop_loss_threshold: float = -0.05,
                 var_confidence: float = 0.95):
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.stop_loss_threshold = stop_loss_threshold
        self.var_confidence = var_confidence
        
        # Track portfolio history for risk calculations
        self.portfolio_history = []
        self.return_history = []
        
    def assess_trade_risk(self, trade: Trade, portfolio: Portfolio, 
                         price_data: Dict[str, pd.DataFrame],
                         correlation_matrix: Optional[np.ndarray] = None) -> bool:
        """Assess if a trade meets risk management criteria"""
        
        # Position size check
        if not self._check_position_size(trade, portfolio):
            return False
            
        # Portfolio concentration check
        if not self._check_concentration(trade, portfolio):
            return False
            
        # Correlation check
        if correlation_matrix is not None and not self._check_correlation(trade, portfolio, correlation_matrix):
            return False
            
        # Volatility check
        if not self._check_volatility(trade, price_data):
            return False
            
        return True
    
    def _check_position_size(self, trade: Trade, portfolio: Portfolio) -> bool:
        """Check if position size is within limits"""
        position_value = trade.quantity * trade.price
        max_allowed = portfolio.total_value * self.max_position_size
        return position_value <= max_allowed
    
    def _check_concentration(self, trade: Trade, portfolio: Portfolio) -> bool:
        """Check portfolio concentration limits"""
        current_position = portfolio.positions.get(trade.symbol, 0)
        
        if trade.action == Action.BUY:
            new_position_value = (current_position + trade.quantity) * trade.price
        else:
            new_position_value = max(0, (current_position - trade.quantity) * trade.price)
            
        concentration = new_position_value / portfolio.total_value
        return concentration <= self.max_position_size
    
    def _check_correlation(self, trade: Trade, portfolio: Portfolio, 
                          correlation_matrix: np.ndarray) -> bool:
        """Check correlation with existing positions"""
        # This would require symbol-to-index mapping
        # Simplified implementation
        return True
    
    def _check_volatility(self, trade: Trade, price_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if stock volatility is within acceptable limits"""
        if trade.symbol not in price_data:
            return False
            
        df = price_data[trade.symbol]
        if len(df) < 30:  # Need minimum data
            return False
            
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Reject extremely volatile stocks (>50% annual volatility)
        return volatility < 0.5
    
    def calculate_portfolio_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 30:
            return 0.0
            
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_portfolio_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_portfolio_var(returns, confidence)
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
            
        return -np.mean(tail_returns)
    
    def calculate_risk_metrics(self, portfolio_values: List[float], 
                             benchmark_values: Optional[List[float]] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        if len(portfolio_values) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Value at Risk
        var_95 = self.calculate_portfolio_var(returns, 0.95)
        var_99 = self.calculate_portfolio_var(returns, 0.99)
        
        # Conditional Value at Risk
        cvar_95 = self.calculate_portfolio_cvar(returns, 0.95)
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Beta and Correlation (if benchmark provided)
        beta, correlation = 0.0, 0.0
        if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
            benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
            if len(benchmark_returns) == len(returns):
                correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            correlation=correlation
        )
    
    def generate_stop_loss_orders(self, portfolio: Portfolio, 
                                 current_prices: Dict[str, float],
                                 entry_prices: Dict[str, float]) -> List[Trade]:
        """Generate stop-loss orders for current positions"""
        
        stop_orders = []
        
        for symbol, quantity in portfolio.positions.items():
            if symbol in current_prices and symbol in entry_prices:
                current_price = current_prices[symbol]
                entry_price = entry_prices[symbol]
                
                # Calculate return since entry
                return_pct = (current_price - entry_price) / entry_price
                
                # Generate stop-loss if threshold breached
                if return_pct <= self.stop_loss_threshold:
                    stop_order = Trade(
                        symbol=symbol,
                        action=Action.SELL,
                        quantity=quantity,
                        price=current_price,
                        timestamp=pd.Timestamp.now(),
                        confidence=1.0,  # High confidence for risk management
                        uncertainty=0.0
                    )
                    stop_orders.append(stop_order)
        
        return stop_orders
    
    def update_portfolio_history(self, portfolio_value: float, portfolio_return: float):
        """Update portfolio history for risk tracking"""
        self.portfolio_history.append(portfolio_value)
        self.return_history.append(portfolio_return)
        
        # Keep only last 252 days (1 year)
        if len(self.portfolio_history) > 252:
            self.portfolio_history.pop(0)
            self.return_history.pop(0)
    
    def check_portfolio_risk_limits(self) -> bool:
        """Check if current portfolio risk is within limits"""
        if len(self.return_history) < 30:
            return True
            
        current_var = self.calculate_portfolio_var(np.array(self.return_history))
        return current_var <= self.max_portfolio_risk

class PositionSizer:
    def __init__(self, method: str = 'kelly', risk_per_trade: float = 0.02):
        self.method = method
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(self, 
                              signal_strength: float,
                              prediction_confidence: float,
                              uncertainty: float,
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float] = None) -> float:
        """Calculate optimal position size based on selected method"""
        
        if self.method == 'kelly':
            return self._kelly_criterion(signal_strength, prediction_confidence, 
                                       portfolio_value, entry_price)
        elif self.method == 'fixed_risk':
            return self._fixed_risk_sizing(portfolio_value, entry_price, 
                                         stop_loss_price, uncertainty)
        elif self.method == 'volatility':
            return self._volatility_sizing(portfolio_value, entry_price, uncertainty)
        else:
            return self._equal_weight_sizing(portfolio_value, entry_price)
    
    def _kelly_criterion(self, signal_strength: float, confidence: float,
                        portfolio_value: float, entry_price: float) -> float:
        """Kelly Criterion position sizing"""
        
        # Estimate win probability and win/loss ratio from signal strength and confidence
        win_prob = 0.5 + (signal_strength * confidence * 0.3)
        win_prob = np.clip(win_prob, 0.1, 0.9)
        
        avg_win = signal_strength * 0.1  # Expected win
        avg_loss = 0.05  # Expected loss
        
        # Kelly fraction
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = np.clip(kelly_fraction, 0, 0.2)  # Cap at 20%
        
        # Adjust for uncertainty
        uncertainty_adjustment = 1.0 / (1.0 + uncertainty)
        final_fraction = kelly_fraction * uncertainty_adjustment
        
        position_value = portfolio_value * final_fraction
        return position_value / entry_price
    
    def _fixed_risk_sizing(self, portfolio_value: float, entry_price: float,
                          stop_loss_price: Optional[float], uncertainty: float) -> float:
        """Fixed risk percentage sizing"""
        
        if stop_loss_price is None:
            stop_loss_price = entry_price * 0.95  # 5% stop loss
            
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Adjust for uncertainty
        uncertainty_adjustment = 1.0 / (1.0 + uncertainty)
        adjusted_risk = risk_amount * uncertainty_adjustment
        
        return adjusted_risk / risk_per_share if risk_per_share > 0 else 0
    
    def _volatility_sizing(self, portfolio_value: float, entry_price: float, 
                          uncertainty: float) -> float:
        """Inverse volatility sizing"""
        
        # Use uncertainty as proxy for volatility
        volatility = max(uncertainty, 0.01)
        target_volatility = 0.15  # 15% target portfolio volatility
        
        position_fraction = target_volatility / volatility * 0.1  # Scale down
        position_fraction = np.clip(position_fraction, 0.01, 0.2)
        
        position_value = portfolio_value * position_fraction
        return position_value / entry_price
    
    def _equal_weight_sizing(self, portfolio_value: float, entry_price: float) -> float:
        """Equal weight sizing (1/N)"""
        n_positions = 10  # Assume 10 position portfolio
        position_fraction = 1.0 / n_positions
        position_value = portfolio_value * position_fraction
        return position_value / entry_price