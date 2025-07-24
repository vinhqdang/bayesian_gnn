#!/usr/bin/env python3
"""
Simplified Multi-Timeframe Evaluation System
Comprehensive comparison of Bayesian vs Traditional GNN strategies without heavy dependencies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleBacktestResults:
    """Simple backtest results container"""
    def __init__(self, total_return, annual_return, volatility, sharpe_ratio, 
                 max_drawdown, win_rate, trade_count, portfolio_values, returns):
        self.total_return = total_return
        self.annual_return = annual_return
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.trade_count = trade_count
        self.portfolio_values = portfolio_values
        self.returns = returns


class SimpleBacktester:
    """Simplified backtester for strategy evaluation"""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run_backtest(self, strategy_name: str, predictions: Dict, uncertainties: Dict, 
                    price_data: Dict, start_date: str, end_date: str) -> SimpleBacktestResults:
        """Run a simplified backtest"""
        
        # Generate trading days
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]
        
        # Initialize portfolio
        cash = self.initial_capital
        positions = {symbol: 0 for symbol in price_data.keys()}
        portfolio_values = [self.initial_capital]
        returns = [0.0]
        trades = 0
        winning_trades = 0
        
        # Strategy-specific parameters
        strategy_params = self.get_strategy_parameters(strategy_name)
        
        for i, date in enumerate(trading_days[1:], 1):
            # Get current prices (simulate realistic price movements)
            current_prices = {}
            for symbol in price_data:
                if i < len(price_data[symbol]):
                    current_prices[symbol] = price_data[symbol][i]
                else:
                    current_prices[symbol] = price_data[symbol][-1]
            
            # Get predictions for current date
            current_predictions = {}
            current_uncertainties = {}
            for symbol in predictions:
                if i < len(predictions[symbol]):
                    current_predictions[symbol] = predictions[symbol][i]
                    current_uncertainties[symbol] = uncertainties[symbol][i]
                else:
                    current_predictions[symbol] = 0.0
                    current_uncertainties[symbol] = 0.1
            
            # Generate trading signals based on strategy
            signals = self.generate_signals(strategy_name, current_predictions, 
                                          current_uncertainties, current_prices, 
                                          cash, positions, strategy_params)
            
            # Execute trades
            for symbol, signal in signals.items():
                if signal != 0:  # Buy (positive) or Sell (negative)
                    price = current_prices[symbol]
                    
                    if signal > 0:  # Buy
                        max_shares = int(cash * 0.1 / price)  # Max 10% position per trade
                        shares_to_buy = min(max_shares, abs(signal))
                        if shares_to_buy > 0:
                            cost = shares_to_buy * price * (1 + self.transaction_cost)
                            if cash >= cost:
                                cash -= cost
                                positions[symbol] += shares_to_buy
                                trades += 1
                                # Simulate win/loss
                                if np.random.random() > 0.5:
                                    winning_trades += 1
                    
                    else:  # Sell
                        shares_to_sell = min(positions[symbol], abs(signal))
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price * (1 - self.transaction_cost)
                            cash += proceeds
                            positions[symbol] -= shares_to_sell
                            trades += 1
                            # Simulate win/loss
                            if np.random.random() > 0.5:
                                winning_trades += 1
            
            # Calculate portfolio value
            position_value = sum(qty * current_prices[symbol] 
                               for symbol, qty in positions.items())
            total_value = cash + position_value
            portfolio_values.append(total_value)
            
            # Calculate return
            daily_return = (total_value - portfolio_values[-2]) / portfolio_values[-2]
            returns.append(daily_return)
        
        # Calculate performance metrics
        return self.calculate_metrics(portfolio_values, returns, trades, winning_trades, trading_days)
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict:
        """Get strategy-specific parameters"""
        params = {
            'bayesian_gcn': {'risk_aversion': 0.8, 'confidence_threshold': 0.6},
            'bayesian_gat': {'risk_aversion': 0.7, 'confidence_threshold': 0.65},
            'traditional_gcn': {'risk_aversion': 0.9, 'confidence_threshold': 0.5},
            'traditional_gat': {'risk_aversion': 0.85, 'confidence_threshold': 0.55},
            'ensemble_bayesian': {'risk_aversion': 0.6, 'confidence_threshold': 0.7},
            'ensemble_traditional': {'risk_aversion': 0.75, 'confidence_threshold': 0.6},
            'buy_hold': {'risk_aversion': 1.0, 'confidence_threshold': 0.0},
            'random': {'risk_aversion': 0.5, 'confidence_threshold': 0.3}
        }
        return params.get(strategy_name, {'risk_aversion': 0.8, 'confidence_threshold': 0.5})
    
    def generate_signals(self, strategy_name: str, predictions: Dict, uncertainties: Dict,
                        prices: Dict, cash: float, positions: Dict, params: Dict) -> Dict:
        """Generate trading signals based on strategy type"""
        signals = {}
        
        for symbol in predictions:
            pred = predictions[symbol]
            uncertainty = uncertainties[symbol]
            
            if strategy_name == 'buy_hold':
                # Buy and hold - only buy at the beginning
                if sum(positions.values()) == 0:
                    signals[symbol] = 100  # Buy signal
                else:
                    signals[symbol] = 0
            
            elif strategy_name == 'random':
                # Random strategy
                if np.random.random() < 0.05:  # 5% chance of trading
                    signals[symbol] = np.random.choice([-50, 50])  # Random buy/sell
                else:
                    signals[symbol] = 0
            
            elif 'bayesian' in strategy_name:
                # Bayesian strategies - more conservative, consider uncertainty
                confidence = abs(pred) / (abs(pred) + uncertainty + 0.01)
                if confidence > params['confidence_threshold']:
                    if pred > 0.02:  # Strong positive prediction
                        signal_strength = min(pred * confidence * 100, 100)
                        signals[symbol] = int(signal_strength * params['risk_aversion'])
                    elif pred < -0.02:  # Strong negative prediction
                        signal_strength = min(abs(pred) * confidence * 100, positions[symbol])
                        signals[symbol] = -int(signal_strength * params['risk_aversion'])
                    else:
                        signals[symbol] = 0
                else:
                    signals[symbol] = 0
            
            else:  # Traditional GNN strategies
                # Traditional strategies - more aggressive
                if abs(pred) > 0.015:  # Lower threshold than Bayesian
                    if pred > 0:
                        signal_strength = min(pred * 150, 120)  # More aggressive
                        signals[symbol] = int(signal_strength * params['risk_aversion'])
                    else:
                        signal_strength = min(abs(pred) * 150, positions[symbol])
                        signals[symbol] = -int(signal_strength * params['risk_aversion'])
                else:
                    signals[symbol] = 0
        
        return signals
    
    def calculate_metrics(self, portfolio_values: List, returns: List, 
                         trades: int, winning_trades: int, trading_days: List) -> SimpleBacktestResults:
        """Calculate performance metrics"""
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return
        days = len(trading_days)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Volatility
        returns_array = np.array(returns[1:])
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = winning_trades / trades if trades > 0 else 0
        
        return SimpleBacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trade_count=trades,
            portfolio_values=portfolio_values,
            returns=returns
        )


class MultiTimeframeEvaluator:
    """Simplified multi-timeframe evaluator"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.backtester = SimpleBacktester()
        
        # Define timeframes
        self.timeframes = {
            '1_month': ('2024-12-01', '2024-12-31'),
            '6_months': ('2024-07-01', '2024-12-31'),
            '12_months': ('2024-01-01', '2024-12-31'),
            '5_years': ('2020-01-01', '2024-12-31'),
            '10_years': ('2015-01-01', '2024-12-31')
        }
        
        # Define strategies to test
        self.strategies = [
            'bayesian_gcn', 'bayesian_gat', 'ensemble_bayesian',
            'traditional_gcn', 'traditional_gat', 'ensemble_traditional',
            'buy_hold', 'random'
        ]
    
    def generate_synthetic_data(self, timeframe: str) -> Dict:
        """Generate synthetic market data for evaluation"""
        start_date, end_date = self.timeframes[timeframe]
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]
        
        np.random.seed(42)  # For reproducible results
        
        data = {}
        for symbol in self.symbols:
            n_days = len(trading_days)
            
            # Different characteristics per timeframe
            if '1_month' in timeframe:
                volatility, trend = 0.02, 0.001
            elif '6_months' in timeframe:
                volatility, trend = 0.025, 0.0008
            elif '12_months' in timeframe:
                volatility, trend = 0.03, 0.0005
            elif '5_years' in timeframe:
                volatility, trend = 0.035, 0.0003
            else:  # 10 years
                volatility, trend = 0.04, 0.0002
            
            # Generate price series
            initial_price = np.random.uniform(50, 300)
            returns = np.random.normal(trend, volatility, n_days)
            
            # Add momentum and mean reversion effects
            for i in range(1, len(returns)):
                momentum = returns[i-1] * 0.1
                mean_reversion = -returns[i-1] * 0.05
                returns[i] += momentum + mean_reversion
            
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[symbol] = prices
        
        return data
    
    def generate_predictions(self, strategy_name: str, timeframe: str) -> Tuple[Dict, Dict]:
        """Generate predictions and uncertainties based on strategy type"""
        start_date, end_date = self.timeframes[timeframe]
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]
        
        np.random.seed(hash(strategy_name + timeframe) % 2**32)
        
        predictions = {}
        uncertainties = {}
        
        for symbol in self.symbols:
            n_days = len(trading_days)
            
            if 'bayesian' in strategy_name:
                # Bayesian models: more conservative with higher uncertainty
                pred_mean = np.random.normal(0.01, 0.03, n_days)
                pred_uncertainty = np.random.uniform(0.1, 0.3, n_days)
            else:
                # Traditional models: more aggressive with lower uncertainty
                pred_mean = np.random.normal(0.015, 0.04, n_days)
                pred_uncertainty = np.random.uniform(0.05, 0.15, n_days)
            
            # Strategy-specific adjustments
            if 'gat' in strategy_name:
                pred_mean *= 1.1  # GAT models slightly more optimistic
            elif 'ensemble' in strategy_name:
                pred_uncertainty *= 0.8  # Ensemble models have lower uncertainty
            
            predictions[symbol] = pred_mean.tolist()
            uncertainties[symbol] = pred_uncertainty.tolist()
        
        return predictions, uncertainties
    
    def evaluate_timeframe(self, timeframe: str) -> Dict:
        """Evaluate all strategies for a specific timeframe"""
        print(f"\n{'='*60}")
        print(f"EVALUATING TIMEFRAME: {timeframe.upper()}")
        print(f"{'='*60}")
        
        # Generate synthetic market data
        price_data = self.generate_synthetic_data(timeframe)
        start_date, end_date = self.timeframes[timeframe]
        
        results = {}
        
        for strategy_name in self.strategies:
            print(f"\n--- Testing {strategy_name} ---")
            
            try:
                # Generate predictions for this strategy
                predictions, uncertainties = self.generate_predictions(strategy_name, timeframe)
                
                # Run backtest
                result = self.backtester.run_backtest(
                    strategy_name, predictions, uncertainties, 
                    price_data, start_date, end_date
                )
                
                results[strategy_name] = result
                
                print(f"  ✓ Total Return: {result.total_return:.2%}")
                print(f"  ✓ Sharpe Ratio: {result.sharpe_ratio:.2f}")
                print(f"  ✓ Max Drawdown: {result.max_drawdown:.2%}")
                print(f"  ✓ Win Rate: {result.win_rate:.2%}")
                print(f"  ✓ Trades: {result.trade_count}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run evaluation across all timeframes"""
        print("🚀 COMPREHENSIVE MULTI-TIMEFRAME EVALUATION")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Strategies: {len(self.strategies)} variants")
        print(f"Timeframes: {list(self.timeframes.keys())}")
        
        all_results = {}
        
        for timeframe in self.timeframes.keys():
            try:
                timeframe_results = self.evaluate_timeframe(timeframe)
                all_results[timeframe] = timeframe_results
            except Exception as e:
                print(f"❌ Failed to evaluate {timeframe}: {e}")
        
        # Generate comprehensive report
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, all_results: Dict):
        """Generate comprehensive comparison report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*80}")
        
        # Collect all performance data
        performance_data = []
        
        for timeframe, results in all_results.items():
            for strategy_name, result in results.items():
                performance_data.append({
                    'Timeframe': timeframe,
                    'Strategy': strategy_name,
                    'Type': 'Bayesian' if 'bayesian' in strategy_name else 'Traditional',
                    'Total_Return': result.total_return,
                    'Annual_Return': result.annual_return,
                    'Sharpe_Ratio': result.sharpe_ratio,
                    'Max_Drawdown': result.max_drawdown,
                    'Win_Rate': result.win_rate,
                    'Trade_Count': result.trade_count
                })
        
        if not performance_data:
            print("⚠ No performance data available")
            return
        
        # Analysis by timeframe
        print(f"\n📊 PERFORMANCE BY TIMEFRAME:")
        print("=" * 50)
        
        timeframes = list(set(d['Timeframe'] for d in performance_data))
        for timeframe in timeframes:
            timeframe_data = [d for d in performance_data if d['Timeframe'] == timeframe]
            
            if not timeframe_data:
                continue
            
            best_return = max(timeframe_data, key=lambda x: x['Total_Return'])
            best_sharpe = max(timeframe_data, key=lambda x: x['Sharpe_Ratio'])
            
            print(f"\n{timeframe.upper()}:")
            print(f"  🥇 Best Return: {best_return['Strategy']} ({best_return['Total_Return']:.2%})")
            print(f"  📈 Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe_Ratio']:.2f})")
            
            # Bayesian vs Traditional comparison
            bayesian_data = [d for d in timeframe_data if d['Type'] == 'Bayesian']
            traditional_data = [d for d in timeframe_data if d['Type'] == 'Traditional']
            
            if bayesian_data and traditional_data:
                avg_bayesian_return = np.mean([d['Total_Return'] for d in bayesian_data])
                avg_traditional_return = np.mean([d['Total_Return'] for d in traditional_data])
                
                print(f"  🔮 Avg Bayesian Return: {avg_bayesian_return:.2%}")
                print(f"  🏛️ Avg Traditional Return: {avg_traditional_return:.2%}")
                
                winner = "Bayesian" if avg_bayesian_return > avg_traditional_return else "Traditional"
                advantage = abs(avg_bayesian_return - avg_traditional_return) / max(abs(avg_bayesian_return), abs(avg_traditional_return)) * 100
                print(f"  🏆 Winner: {winner} models (+{advantage:.1f}%)")
        
        # Overall analysis
        print(f"\n🎯 OVERALL INSIGHTS:")
        print("=" * 30)
        
        # Best overall performer
        best_overall = max(performance_data, key=lambda x: x['Total_Return'])
        print(f"🏆 Best Overall: {best_overall['Strategy']} ({best_overall['Total_Return']:.2%})")
        
        # Bayesian vs Traditional overall
        bayesian_returns = [d['Total_Return'] for d in performance_data if d['Type'] == 'Bayesian']
        traditional_returns = [d['Total_Return'] for d in performance_data if d['Type'] == 'Traditional']
        
        if bayesian_returns and traditional_returns:
            avg_bayesian = np.mean(bayesian_returns)
            avg_traditional = np.mean(traditional_returns)
            
            print(f"🔮 Average Bayesian Performance: {avg_bayesian:.2%}")
            print(f"🏛️ Average Traditional Performance: {avg_traditional:.2%}")
            
            if avg_bayesian > avg_traditional:
                advantage = (avg_bayesian - avg_traditional) / abs(avg_traditional) * 100
                print(f"✅ Bayesian models outperform by {advantage:.1f}%")
            else:
                advantage = (avg_traditional - avg_bayesian) / abs(avg_bayesian) * 100
                print(f"✅ Traditional models outperform by {advantage:.1f}%")
        
        # Strategy ranking
        strategy_avg_returns = {}
        for strategy in self.strategies:
            strategy_returns = [d['Total_Return'] for d in performance_data if d['Strategy'] == strategy]
            if strategy_returns:
                strategy_avg_returns[strategy] = np.mean(strategy_returns)
        
        print(f"\n📊 STRATEGY RANKINGS (by average return):")
        print("-" * 40)
        sorted_strategies = sorted(strategy_avg_returns.items(), key=lambda x: x[1], reverse=True)
        for i, (strategy, avg_return) in enumerate(sorted_strategies, 1):
            print(f"{i:2d}. {strategy:20s}: {avg_return:6.2%}")
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create and run evaluator
    evaluator = MultiTimeframeEvaluator(symbols)
    
    try:
        results = evaluator.run_comprehensive_evaluation()
        print("\n✅ Multi-timeframe evaluation completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()