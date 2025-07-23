#!/usr/bin/env python3
"""
Comprehensive backtest comparison script
Compare Bayesian GNN strategies against S&P 500 and simple trading strategies
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import StockDataLoader
from trading.strategy import (BuyAndHoldStrategy, SimpleMovingAverageStrategy, 
                             RSIStrategy, RandomStrategy, BayesianMomentumStrategy)
from trading.backtester import Backtester, BenchmarkComparator, MultiPeriodBacktester
from utils.config import get_config

def run_comprehensive_backtest():
    """Run comprehensive backtest comparison"""
    
    print("=== Comprehensive Backtest Comparison ===")
    print("Timeframe: 2025-01-01 to 2025-06-30")
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2025-01-01'
    end_date = '2025-06-30'
    initial_capital = 100000.0
    
    # Load configuration
    try:
        config = get_config()
        trading_config = config.get_trading_config()
        initial_capital = trading_config.initial_capital
        transaction_cost = trading_config.transaction_cost
    except:
        print("Using default configuration")
        transaction_cost = 0.001
    
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Transaction Cost: {transaction_cost:.3f}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Load market data
    print("\nLoading market data...")
    try:
        data_loader = StockDataLoader(symbols, start_date, end_date)
        price_data = data_loader.fetch_data()
        print(f"Loaded data for {len(price_data)} symbols")
    except Exception as e:
        print(f"Error loading market data: {e}")
        # Create dummy data for demonstration
        print("Creating dummy data for demonstration...")
        price_data = create_dummy_data(symbols, start_date, end_date)
    
    # Create backtester and benchmark comparator
    backtester = Backtester(initial_capital=initial_capital, transaction_cost=transaction_cost)
    benchmark_comparator = BenchmarkComparator()
    
    # Fetch S&P 500 benchmark data
    print("\nFetching S&P 500 benchmark data...")
    try:
        sp500_prices = benchmark_comparator.fetch_benchmark_data('^GSPC', start_date, end_date)
        benchmark_performance = benchmark_comparator.calculate_benchmark_performance(
            sp500_prices, initial_capital
        )
        print("S&P 500 benchmark data loaded successfully")
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        benchmark_performance = None
    
    # Generate dummy predictions for strategies that need them
    predictions = generate_dummy_predictions(symbols, start_date, end_date)
    uncertainties = generate_dummy_uncertainties(symbols, start_date, end_date)
    
    # Define strategies to test
    strategies = [
        ('Buy & Hold', BuyAndHoldStrategy(initial_capital, transaction_cost)),
        ('Moving Average', SimpleMovingAverageStrategy(initial_capital, transaction_cost)),
        ('RSI Strategy', RSIStrategy(initial_capital, transaction_cost)),
        ('Random Strategy', RandomStrategy(initial_capital, transaction_cost)),
        ('Bayesian Momentum', BayesianMomentumStrategy(initial_capital, transaction_cost))
    ]
    
    # Run backtests
    print(f"\nRunning backtests for {len(strategies)} strategies...")
    strategy_results = []
    strategy_names = []
    
    for name, strategy in strategies:
        print(f"\nTesting {name} strategy...")
        try:
            result = backtester.run_backtest(
                strategy, predictions, uncertainties, price_data, start_date, end_date
            )
            strategy_results.append(result)
            strategy_names.append(name)
            
            print(f"✓ {name} completed")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    if benchmark_performance and strategy_results:
        comparison_df = benchmark_comparator.compare_strategies(
            strategy_results, benchmark_performance, strategy_names
        )
        
        # Format the DataFrame for better display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: f'{x:.3f}' if abs(x) < 1 else f'{x:.2f}')
        
        print("\nDetailed Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Summary insights
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        if len(strategy_results) > 0:
            best_return_idx = np.argmax([r.total_return for r in strategy_results])
            best_sharpe_idx = np.argmax([r.sharpe_ratio for r in strategy_results])
            
            print(f"🏆 Best Total Return: {strategy_names[best_return_idx]} ({strategy_results[best_return_idx].total_return:.2%})")
            print(f"📈 Best Sharpe Ratio: {strategy_names[best_sharpe_idx]} ({strategy_results[best_sharpe_idx].sharpe_ratio:.2f})")
            
            if benchmark_performance:
                print(f"📊 S&P 500 Return: {benchmark_performance['total_return']:.2%}")
                
                # Count strategies that beat the benchmark
                outperforming = sum(1 for r in strategy_results if r.total_return > benchmark_performance['total_return'])
                print(f"🎯 Strategies beating S&P 500: {outperforming}/{len(strategy_results)}")
    
    else:
        print("⚠️  Insufficient data for comprehensive comparison")
        if strategy_results:
            print("\nStrategy Results Summary:")
            for i, (name, result) in enumerate(zip(strategy_names, strategy_results)):
                print(f"{name}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETED")
    print("="*80)
    
    return strategy_results, benchmark_performance

def create_dummy_data(symbols, start_date, end_date):
    """Create dummy price data for demonstration"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # Trading days only
    
    dummy_data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        # Generate realistic price movement
        initial_price = np.random.uniform(100, 300)
        returns = np.random.normal(0.001, 0.02, len(dates))  # Small daily returns with volatility
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
            'Volume': [np.random.randint(1000000, 10000000) for _ in prices]
        }, index=dates)
        
        dummy_data[symbol] = df
    
    return dummy_data

def generate_dummy_predictions(symbols, start_date, end_date):
    """Generate dummy predictions for strategies"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]
    
    predictions = {}
    np.random.seed(123)
    
    for symbol in symbols:
        # Generate realistic prediction time series
        predictions[symbol] = np.random.normal(0.02, 0.05, len(dates)).tolist()
    
    return predictions

def generate_dummy_uncertainties(symbols, start_date, end_date):
    """Generate dummy uncertainties for strategies"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]
    
    uncertainties = {}
    np.random.seed(456)
    
    for symbol in symbols:
        # Generate realistic uncertainty time series
        uncertainties[symbol] = np.random.uniform(0.05, 0.15, len(dates)).tolist()
    
    return uncertainties

if __name__ == "__main__":
    try:
        results = run_comprehensive_backtest()
        print("\n✅ Backtest comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()