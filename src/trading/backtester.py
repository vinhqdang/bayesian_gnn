import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from .strategy import Trade, Action, Portfolio, TradingStrategy

@dataclass
class BacktestResults:
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    trade_count: int
    portfolio_values: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)

class Backtester:
    def __init__(self, initial_capital: float = 100000.0, 
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def run_backtest(self, strategy: TradingStrategy, 
                    predictions: Dict[str, List[float]], 
                    uncertainties: Dict[str, List[float]], 
                    price_data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str) -> BacktestResults:
        
        # Initialize portfolio
        portfolio = Portfolio(cash=self.initial_capital, positions={})
        
        # Get date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = [d for d in date_range if d.weekday() < 5]  # Trading days only
        
        # Track results
        portfolio_values = [self.initial_capital]
        returns = [0.0]
        all_trades = []
        timestamps = [pd.Timestamp(start_date)]
        
        for i, current_date in enumerate(date_range[1:], 1):
            # Get current prices
            current_prices = {}
            for symbol in price_data:
                try:
                    price_row = price_data[symbol][price_data[symbol].index.date == current_date.date()]
                    if not price_row.empty:
                        current_prices[symbol] = price_row['Close'].iloc[0]
                except:
                    continue
            
            if not current_prices:
                continue
                
            # Get predictions and uncertainties for current date
            current_predictions = {}
            current_uncertainties = {}
            
            for symbol in predictions:
                if i < len(predictions[symbol]):
                    current_predictions[symbol] = predictions[symbol][i]
                    current_uncertainties[symbol] = uncertainties[symbol][i] if symbol in uncertainties else 0.1
            
            # Generate trading signals
            signals = strategy.generate_signals(
                current_predictions, current_uncertainties, 
                current_prices, portfolio
            )
            
            # Execute trades
            for trade in signals:
                if self._can_execute_trade(trade, portfolio, current_prices):
                    self._execute_trade(trade, portfolio, current_prices)
                    all_trades.append(trade)
            
            # Update portfolio value
            portfolio.update_value(current_prices)
            portfolio_values.append(portfolio.total_value)
            
            # Calculate return
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                returns.append(daily_return)
            
            timestamps.append(current_date)
        
        # Calculate performance metrics
        results = self._calculate_metrics(portfolio_values, returns, all_trades, timestamps)
        return results
    
    def _can_execute_trade(self, trade: Trade, portfolio: Portfolio, 
                          current_prices: Dict[str, float]) -> bool:
        """Check if trade can be executed given current portfolio state"""
        
        if trade.action == Action.BUY:
            # Check if enough cash available
            required_cash = trade.quantity * trade.price * (1 + self.transaction_cost)
            return portfolio.cash >= required_cash
            
        elif trade.action == Action.SELL:
            # Check if enough shares available
            current_position = portfolio.positions.get(trade.symbol, 0)
            return current_position >= trade.quantity
            
        return False
    
    def _execute_trade(self, trade: Trade, portfolio: Portfolio, 
                      current_prices: Dict[str, float]):
        """Execute a trade and update portfolio"""
        
        # Apply slippage
        execution_price = trade.price
        if trade.action == Action.BUY:
            execution_price *= (1 + self.slippage)
        else:
            execution_price *= (1 - self.slippage)
        
        # Calculate transaction cost
        transaction_cost = trade.quantity * execution_price * self.transaction_cost
        
        if trade.action == Action.BUY:
            # Buy shares
            total_cost = trade.quantity * execution_price + transaction_cost
            portfolio.cash -= total_cost
            portfolio.positions[trade.symbol] = portfolio.positions.get(trade.symbol, 0) + trade.quantity
            
        elif trade.action == Action.SELL:
            # Sell shares
            total_proceeds = trade.quantity * execution_price - transaction_cost
            portfolio.cash += total_proceeds
            portfolio.positions[trade.symbol] = portfolio.positions.get(trade.symbol, 0) - trade.quantity
            
            # Remove position if fully sold
            if portfolio.positions[trade.symbol] <= 0:
                del portfolio.positions[trade.symbol]
    
    def _calculate_metrics(self, portfolio_values: List[float], 
                         returns: List[float], trades: List[Trade],
                         timestamps: List[pd.Timestamp]) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return
        days = (timestamps[-1] - timestamps[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility (annualized)
        returns_array = np.array(returns[1:])  # Skip first zero return
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in trades if self._is_winning_trade(t, portfolio_values)]
        losing_trades = [t for t in trades if not self._is_winning_trade(t, portfolio_values)]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        total_profit = sum(self._calculate_trade_profit(t) for t in winning_trades)
        total_loss = sum(abs(self._calculate_trade_profit(t)) for t in losing_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=len(trades),
            portfolio_values=portfolio_values,
            returns=returns,
            trades=trades,
            timestamps=timestamps
        )
    
    def _is_winning_trade(self, trade: Trade, portfolio_values: List[float]) -> bool:
        # Simplified winning trade detection
        # In practice, you'd track individual trade P&L
        return trade.action == Action.BUY  # Placeholder logic
    
    def _calculate_trade_profit(self, trade: Trade) -> float:
        # Simplified profit calculation
        # In practice, you'd track entry/exit prices
        return trade.quantity * trade.price * 0.02  # Placeholder: 2% profit per trade
    
    def plot_results(self, results: BacktestResults, benchmark_data: Optional[pd.Series] = None):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(results.timestamps, results.portfolio_values, label='Strategy', linewidth=2)
        if benchmark_data is not None:
            axes[0, 0].plot(benchmark_data.index, benchmark_data.values, label='Benchmark', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[0, 1].hist(results.returns[1:], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown
        peak = np.maximum.accumulate(results.portfolio_values)
        drawdown = (np.array(results.portfolio_values) - peak) / peak
        axes[1, 0].fill_between(results.timestamps, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        metrics_text = f"""
        Total Return: {results.total_return:.2%}
        Annual Return: {results.annual_return:.2%}
        Volatility: {results.volatility:.2%}
        Sharpe Ratio: {results.sharpe_ratio:.2f}
        Max Drawdown: {results.max_drawdown:.2%}
        Calmar Ratio: {results.calmar_ratio:.2f}
        Win Rate: {results.win_rate:.2%}
        Profit Factor: {results.profit_factor:.2f}
        Total Trades: {results.trade_count}
        """
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig

class WalkForwardAnalysis:
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        
    def run_analysis(self, strategy: TradingStrategy, 
                    predictions: Dict[str, List[float]], 
                    uncertainties: Dict[str, List[float]], 
                    price_data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str,
                    train_period: int = 252, test_period: int = 63) -> List[BacktestResults]:
        """Run walk-forward analysis"""
        
        results = []
        current_date = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        while current_date + timedelta(days=train_period + test_period) <= end_date_ts:
            # Define train and test periods
            train_start = current_date
            train_end = current_date + timedelta(days=train_period)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_period)
            
            # Run backtest on test period
            test_result = self.backtester.run_backtest(
                strategy, predictions, uncertainties, price_data,
                test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')
            )
            
            results.append(test_result)
            
            # Move to next period
            current_date = test_start
        
        return results
    
    def aggregate_results(self, results: List[BacktestResults]) -> BacktestResults:
        """Aggregate results from multiple periods"""
        
        # Combine all portfolio values and returns
        all_portfolio_values = []
        all_returns = []
        all_trades = []
        all_timestamps = []
        
        for result in results:
            all_portfolio_values.extend(result.portfolio_values)
            all_returns.extend(result.returns)
            all_trades.extend(result.trades)
            all_timestamps.extend(result.timestamps)
        
        # Calculate aggregated metrics
        return self.backtester._calculate_metrics(
            all_portfolio_values, all_returns, all_trades, all_timestamps
        )

class BenchmarkComparator:
    """Compare trading strategies against benchmarks like S&P 500"""
    
    def __init__(self):
        self.benchmark_data = {}
    
    def fetch_benchmark_data(self, symbol: str = '^GSPC', start_date: str = None, 
                           end_date: str = None) -> pd.Series:
        """Fetch benchmark data (default: S&P 500)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            benchmark_prices = data['Close']
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_prices.pct_change().fillna(0)
            
            # Store for later use
            self.benchmark_data[symbol] = {
                'prices': benchmark_prices,
                'returns': benchmark_returns
            }
            
            return benchmark_prices
            
        except Exception as e:
            print(f"Error fetching benchmark data: {e}")
            return pd.Series()
    
    def calculate_benchmark_performance(self, benchmark_prices: pd.Series, 
                                     initial_capital: float = 100000) -> Dict:
        """Calculate benchmark performance metrics"""
        if benchmark_prices.empty:
            return {}
        
        # Calculate portfolio value assuming buy-and-hold
        returns = benchmark_prices.pct_change().fillna(0)
        portfolio_values = [initial_capital]
        
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Calculate other metrics
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'returns': returns.tolist()
        }
    
    def compare_strategies(self, strategy_results: List[BacktestResults], 
                          benchmark_results: Dict, strategy_names: List[str] = None) -> pd.DataFrame:
        """Compare multiple strategies with benchmark"""
        
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(len(strategy_results))]
        
        comparison_data = []
        
        # Add benchmark
        if benchmark_results:
            comparison_data.append({
                'Strategy': 'S&P 500 (Benchmark)',
                'Total Return': benchmark_results['total_return'],
                'Annual Return': benchmark_results['annual_return'],
                'Volatility': benchmark_results['volatility'],
                'Sharpe Ratio': benchmark_results['sharpe_ratio'],
                'Max Drawdown': benchmark_results['max_drawdown'],
                'Win Rate': 'N/A',
                'Trade Count': 'N/A'
            })
        
        # Add strategies
        for i, result in enumerate(strategy_results):
            comparison_data.append({
                'Strategy': strategy_names[i],
                'Total Return': result.total_return,
                'Annual Return': result.annual_return,
                'Volatility': result.volatility,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Trade Count': result.trade_count
            })
        
        return pd.DataFrame(comparison_data)

class MultiPeriodBacktester:
    """Run backtests across multiple time periods"""
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.benchmark_comparator = BenchmarkComparator()
    
    def run_multi_period_backtest(self, strategy: TradingStrategy,
                                 predictions: Dict[str, List[float]], 
                                 uncertainties: Dict[str, List[float]], 
                                 price_data: Dict[str, pd.DataFrame],
                                 periods: List[Tuple[str, str]]) -> List[BacktestResults]:
        """Run backtests across multiple time periods"""
        
        results = []
        
        for start_date, end_date in periods:
            print(f"Running backtest for period: {start_date} to {end_date}")
            
            result = self.backtester.run_backtest(
                strategy, predictions, uncertainties, price_data, start_date, end_date
            )
            results.append(result)
        
        return results
    
    def analyze_period_performance(self, multi_period_results: List[BacktestResults],
                                 periods: List[Tuple[str, str]]) -> pd.DataFrame:
        """Analyze performance across different periods"""
        
        period_data = []
        
        for i, result in enumerate(multi_period_results):
            start_date, end_date = periods[i]
            period_data.append({
                'Period': f"{start_date} to {end_date}",
                'Total Return': result.total_return,
                'Annual Return': result.annual_return,
                'Volatility': result.volatility,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Trade Count': result.trade_count
            })
        
        return pd.DataFrame(period_data)
    
    def plot_multi_period_comparison(self, multi_period_results: List[BacktestResults],
                                   periods: List[Tuple[str, str]], 
                                   benchmark_data: Dict = None):
        """Plot comparison across multiple periods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Portfolio values over time for each period
        for i, result in enumerate(multi_period_results):
            start_date, end_date = periods[i]
            axes[0, 0].plot(result.timestamps, result.portfolio_values, 
                           label=f"{start_date} to {end_date}", alpha=0.7)
        
        axes[0, 0].set_title('Portfolio Value Across Different Periods')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Returns comparison
        metrics = ['Total Return', 'Annual Return', 'Sharpe Ratio']
        period_labels = [f"{start[:4]}-{end[:4]}" for start, end in periods]
        
        x = np.arange(len(period_labels))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = []
            for result in multi_period_results:
                if metric == 'Total Return':
                    values.append(result.total_return)
                elif metric == 'Annual Return':
                    values.append(result.annual_return)
                elif metric == 'Sharpe Ratio':
                    values.append(result.sharpe_ratio)
            
            axes[0, 1].bar(x + i * width, values, width, label=metric, alpha=0.7)
        
        axes[0, 1].set_title('Performance Metrics Across Periods')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(period_labels)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Risk metrics
        max_drawdowns = [result.max_drawdown for result in multi_period_results]
        volatilities = [result.volatility for result in multi_period_results]
        
        axes[1, 0].bar(period_labels, max_drawdowns, alpha=0.7, color='red', label='Max Drawdown')
        axes[1, 0].set_title('Max Drawdown Across Periods')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Trading activity
        trade_counts = [result.trade_count for result in multi_period_results]
        win_rates = [result.win_rate for result in multi_period_results]
        
        ax2 = axes[1, 1].twinx()
        axes[1, 1].bar(period_labels, trade_counts, alpha=0.7, color='blue', label='Trade Count')
        ax2.plot(period_labels, win_rates, color='green', marker='o', label='Win Rate')
        
        axes[1, 1].set_title('Trading Activity Across Periods')
        axes[1, 1].set_ylabel('Trade Count', color='blue')
        ax2.set_ylabel('Win Rate', color='green')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig