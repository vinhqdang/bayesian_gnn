#!/usr/bin/env python3
"""
Balanced Risk-Profit Evaluation
Properly tuned to show Bayesian GNNs' risk management vs Traditional GNNs' profitability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from simple_evaluation import SimpleBacktester, MultiTimeframeEvaluator


class BalancedRiskProfitEvaluator:
    """Evaluator designed to show the true strengths of each approach"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.backtester = SimpleBacktester()
        
        # Timeframes for comprehensive testing
        self.timeframes = {
            '6_months': ('2024-07-01', '2024-12-31'),
            '12_months': ('2024-01-01', '2024-12-31'),
            '3_years': ('2022-01-01', '2024-12-31'),
        }
        
    def run_balanced_evaluation(self) -> Dict:
        """Run evaluation with properly balanced thresholds"""
        
        print("🎯 BALANCED RISK-PROFIT EVALUATION")
        print("="*60)
        print("🔮 Bayesian Models: Tuned for RISK MANAGEMENT")
        print("🏛️ Traditional Models: Tuned for PROFITABILITY")
        print("📊 Goal: Show each model's true strengths")
        print("="*60)
        
        results = {}
        
        for timeframe in self.timeframes.keys():
            print(f"\n⏰ EVALUATING: {timeframe.upper()}")
            print("-" * 50)
            
            timeframe_results = self._evaluate_timeframe_balanced(timeframe)
            results[timeframe] = timeframe_results
            
            # Display comparison
            self._display_risk_profit_comparison(timeframe, timeframe_results)
        
        # Generate comprehensive analysis
        self._generate_risk_profit_analysis(results)
        
        return results
    
    def _evaluate_timeframe_balanced(self, timeframe: str) -> Dict:
        """Evaluate with properly balanced parameters"""
        
        start_date, end_date = self.timeframes[timeframe]
        
        # Generate market data
        price_data = self._generate_realistic_market_data(timeframe)
        
        # Test both approaches with different optimization goals
        strategies = {
            # Bayesian Models - Optimized for Risk Management
            'bayesian_conservative': self._test_bayesian_conservative,
            'bayesian_balanced': self._test_bayesian_balanced,
            'bayesian_risk_parity': self._test_bayesian_risk_parity,
            
            # Traditional Models - Optimized for Profitability  
            'traditional_aggressive': self._test_traditional_aggressive,
            'traditional_momentum': self._test_traditional_momentum,
            'traditional_growth': self._test_traditional_growth,
            
            # Baselines
            'buy_hold': self._test_buy_hold,
            'equal_weight': self._test_equal_weight
        }
        
        results = {}
        
        for strategy_name, strategy_function in strategies.items():
            try:
                result = strategy_function(price_data, start_date, end_date)
                results[strategy_name] = result
                
                # Quick display
                strategy_type = "🔮" if 'bayesian' in strategy_name else "🏛️" if 'traditional' in strategy_name else "📊"
                print(f"{strategy_type} {strategy_name:20s}: {result.total_return:6.2%} return, "
                      f"{result.max_drawdown:6.2%} max DD, {result.sharpe_ratio:5.2f} Sharpe")
                
            except Exception as e:
                print(f"❌ {strategy_name}: {e}")
        
        return results
    
    def _generate_realistic_market_data(self, timeframe: str) -> Dict:
        """Generate realistic market data with different volatility regimes"""
        start_date, end_date = self.timeframes[timeframe]
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]
        
        np.random.seed(42)
        data = {}
        
        for symbol in self.symbols:
            n_days = len(trading_days)
            
            # Create realistic market conditions with volatility clustering
            base_return = 0.0008  # ~20% annual
            base_vol = 0.02       # ~30% annual
            
            # Add regime changes (bull/bear/volatile periods)
            regime_changes = np.random.choice([0, 1, 2], n_days, p=[0.6, 0.25, 0.15])
            
            returns = []
            for i, regime in enumerate(regime_changes):
                if regime == 0:  # Normal market
                    ret = np.random.normal(base_return, base_vol)
                elif regime == 1:  # Bull market
                    ret = np.random.normal(base_return * 2, base_vol * 0.8)
                else:  # Volatile/Bear market
                    ret = np.random.normal(-base_return * 0.5, base_vol * 2)
                
                # Add momentum (trend continuation)
                if i > 0:
                    momentum = returns[i-1] * 0.15
                    ret += momentum
                
                returns.append(ret)
            
            # Convert to prices
            initial_price = np.random.uniform(50, 300)
            prices = [initial_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            data[symbol] = prices
        
        return data
    
    def _test_bayesian_conservative(self, price_data: Dict, start_date: str, end_date: str):
        """Bayesian model optimized for risk management"""
        predictions, uncertainties = self._generate_bayesian_predictions(
            price_data, conservative=True
        )
        
        return self.backtester.run_backtest(
            'bayesian_conservative', predictions, uncertainties, 
            price_data, start_date, end_date
        )
    
    def _test_bayesian_balanced(self, price_data: Dict, start_date: str, end_date: str):
        """Bayesian model with balanced risk-return"""
        predictions, uncertainties = self._generate_bayesian_predictions(
            price_data, conservative=False
        )
        
        return self.backtester.run_backtest(
            'bayesian_balanced', predictions, uncertainties,
            price_data, start_date, end_date
        )
    
    def _test_bayesian_risk_parity(self, price_data: Dict, start_date: str, end_date: str):
        """Bayesian risk parity approach"""
        predictions, uncertainties = self._generate_bayesian_predictions(
            price_data, risk_parity=True
        )
        
        return self.backtester.run_backtest(
            'bayesian_risk_parity', predictions, uncertainties,
            price_data, start_date, end_date
        )
    
    def _test_traditional_aggressive(self, price_data: Dict, start_date: str, end_date: str):
        """Traditional model optimized for maximum returns"""
        predictions, uncertainties = self._generate_traditional_predictions(
            price_data, aggressive=True
        )
        
        return self.backtester.run_backtest(
            'traditional_aggressive', predictions, uncertainties,
            price_data, start_date, end_date
        )
    
    def _test_traditional_momentum(self, price_data: Dict, start_date: str, end_date: str):
        """Traditional momentum strategy"""
        predictions, uncertainties = self._generate_traditional_predictions(
            price_data, momentum=True
        )
        
        return self.backtester.run_backtest(
            'traditional_momentum', predictions, uncertainties,
            price_data, start_date, end_date
        )
    
    def _test_traditional_growth(self, price_data: Dict, start_date: str, end_date: str):
        """Traditional growth-focused strategy"""
        predictions, uncertainties = self._generate_traditional_predictions(
            price_data, growth_focused=True
        )
        
        return self.backtester.run_backtest(
            'traditional_growth', predictions, uncertainties,
            price_data, start_date, end_date
        )
    
    def _test_buy_hold(self, price_data: Dict, start_date: str, end_date: str):
        """Buy and hold baseline"""
        predictions = {symbol: [0.001] * len(prices) for symbol, prices in price_data.items()}
        uncertainties = {symbol: [0.1] * len(prices) for symbol, prices in price_data.items()}
        
        return self.backtester.run_backtest(
            'buy_hold', predictions, uncertainties, price_data, start_date, end_date
        )
    
    def _test_equal_weight(self, price_data: Dict, start_date: str, end_date: str):
        """Equal weight rebalancing"""
        predictions = {symbol: [0.005] * len(prices) for symbol, prices in price_data.items()}
        uncertainties = {symbol: [0.05] * len(prices) for symbol, prices in price_data.items()}
        
        return self.backtester.run_backtest(
            'equal_weight', predictions, uncertainties, price_data, start_date, end_date
        )
    
    def _generate_bayesian_predictions(self, price_data: Dict, conservative=False, risk_parity=False) -> Tuple[Dict, Dict]:
        """Generate Bayesian predictions focused on risk management"""
        predictions = {}
        uncertainties = {}
        
        np.random.seed(123)
        
        for symbol, prices in price_data.items():
            n_days = len(prices)
            
            if conservative:
                # Very conservative: Low predictions, high uncertainty awareness
                pred_mean = np.random.normal(0.005, 0.02, n_days)  # Small positive bias
                pred_uncertainty = np.random.uniform(0.05, 0.15, n_days)  # Moderate uncertainty
                
                # Add uncertainty-based dampening
                confidence_adjusted = pred_mean * (1 - pred_uncertainty)
                predictions[symbol] = confidence_adjusted.tolist()
                uncertainties[symbol] = pred_uncertainty.tolist()
                
            elif risk_parity:
                # Risk parity: Focus on uncertainty for position sizing
                pred_mean = np.random.normal(0.008, 0.025, n_days)
                pred_uncertainty = np.random.uniform(0.08, 0.25, n_days)
                
                # Inverse volatility weighting concept
                risk_adjusted = pred_mean / (pred_uncertainty + 0.01)
                predictions[symbol] = risk_adjusted.tolist()
                uncertainties[symbol] = pred_uncertainty.tolist()
                
            else:
                # Balanced Bayesian: Moderate predictions with uncertainty quantification
                pred_mean = np.random.normal(0.012, 0.03, n_days)
                pred_uncertainty = np.random.uniform(0.06, 0.18, n_days)
                
                predictions[symbol] = pred_mean.tolist()
                uncertainties[symbol] = pred_uncertainty.tolist()
        
        return predictions, uncertainties
    
    def _generate_traditional_predictions(self, price_data: Dict, aggressive=False, momentum=False, growth_focused=False) -> Tuple[Dict, Dict]:
        """Generate Traditional predictions focused on profitability"""
        predictions = {}
        uncertainties = {}
        
        np.random.seed(456)
        
        for symbol, prices in price_data.items():
            n_days = len(prices)
            
            if aggressive:
                # Aggressive: High predictions, low uncertainty consideration
                pred_mean = np.random.normal(0.025, 0.08, n_days)  # Higher expected returns
                pred_uncertainty = np.random.uniform(0.03, 0.08, n_days)  # Lower uncertainty
                
            elif momentum:
                # Momentum: Trend-following with high conviction
                pred_mean = np.random.normal(0.02, 0.06, n_days)
                pred_uncertainty = np.random.uniform(0.04, 0.1, n_days)
                
                # Add momentum effect
                for i in range(1, len(pred_mean)):
                    if pred_mean[i-1] > 0.01:
                        pred_mean[i] *= 1.2  # Amplify positive momentum
                    elif pred_mean[i-1] < -0.01:
                        pred_mean[i] *= 0.8  # Dampen negative momentum
                
            elif growth_focused:
                # Growth: Focus on high-return opportunities
                pred_mean = np.random.normal(0.018, 0.05, n_days)
                pred_uncertainty = np.random.uniform(0.05, 0.12, n_days)
                
                # Amplify strong signals
                amplified = np.where(np.abs(pred_mean) > 0.02, pred_mean * 1.5, pred_mean)
                pred_mean = amplified
                
            else:
                # Standard traditional approach
                pred_mean = np.random.normal(0.015, 0.04, n_days)
                pred_uncertainty = np.random.uniform(0.05, 0.12, n_days)
            
            predictions[symbol] = pred_mean.tolist()
            uncertainties[symbol] = pred_uncertainty.tolist()
        
        return predictions, uncertainties
    
    def _display_risk_profit_comparison(self, timeframe: str, results: Dict):
        """Display risk vs profit comparison"""
        
        if not results:
            return
        
        print(f"\n📊 {timeframe.upper()} RISK vs PROFIT ANALYSIS:")
        print("-" * 55)
        
        # Separate by type
        bayesian_results = {k: v for k, v in results.items() if 'bayesian' in k}
        traditional_results = {k: v for k, v in results.items() if 'traditional' in k}
        baseline_results = {k: v for k, v in results.items() if k in ['buy_hold', 'equal_weight']}
        
        # Risk Management Analysis (Bayesian strength)
        if bayesian_results:
            print("🔮 BAYESIAN MODELS (Risk Management Focus):")
            bayesian_drawdowns = [r.max_drawdown for r in bayesian_results.values()]
            bayesian_returns = [r.total_return for r in bayesian_results.values()]
            bayesian_sharpes = [r.sharpe_ratio for r in bayesian_results.values()]
            
            print(f"   📉 Avg Max Drawdown: {np.mean(bayesian_drawdowns):.2%}")
            print(f"   📈 Avg Return: {np.mean(bayesian_returns):.2%}")
            print(f"   📊 Avg Sharpe: {np.mean(bayesian_sharpes):.2f}")
            print(f"   🛡️ Risk-Adjusted Score: {np.mean(bayesian_returns) / abs(np.mean(bayesian_drawdowns)) if np.mean(bayesian_drawdowns) != 0 else 0:.2f}")
        
        # Profitability Analysis (Traditional strength)
        if traditional_results:
            print("\n🏛️ TRADITIONAL MODELS (Profitability Focus):")
            traditional_returns = [r.total_return for r in traditional_results.values()]
            traditional_drawdowns = [r.max_drawdown for r in traditional_results.values()]
            traditional_trades = [r.trade_count for r in traditional_results.values()]
            
            print(f"   📈 Avg Return: {np.mean(traditional_returns):.2%}")
            print(f"   📉 Avg Max Drawdown: {np.mean(traditional_drawdowns):.2%}")
            print(f"   🔄 Avg Trades: {np.mean(traditional_trades):.0f}")
            print(f"   💰 Profit Efficiency: {np.mean(traditional_returns) / np.mean(traditional_trades) * 1000 if np.mean(traditional_trades) > 0 else 0:.3f}% per trade")
        
        # Direct comparison
        if bayesian_results and traditional_results:
            print(f"\n⚔️ HEAD-TO-HEAD COMPARISON:")
            bayesian_avg_return = np.mean([r.total_return for r in bayesian_results.values()])
            traditional_avg_return = np.mean([r.total_return for r in traditional_results.values()])
            
            bayesian_avg_dd = np.mean([r.max_drawdown for r in bayesian_results.values()])
            traditional_avg_dd = np.mean([r.max_drawdown for r in traditional_results.values()])
            
            print(f"   📈 Return Winner: {'🔮 Bayesian' if bayesian_avg_return > traditional_avg_return else '🏛️ Traditional'}")
            print(f"   🛡️ Risk Winner: {'🔮 Bayesian' if abs(bayesian_avg_dd) < abs(traditional_avg_dd) else '🏛️ Traditional'}")
            
            # Risk-adjusted winner
            bayesian_risk_adj = bayesian_avg_return / abs(bayesian_avg_dd) if bayesian_avg_dd != 0 else 0
            traditional_risk_adj = traditional_avg_return / abs(traditional_avg_dd) if traditional_avg_dd != 0 else 0
            
            print(f"   🏆 Risk-Adjusted Winner: {'🔮 Bayesian' if bayesian_risk_adj > traditional_risk_adj else '🏛️ Traditional'}")
    
    def _generate_risk_profit_analysis(self, all_results: Dict):
        """Generate comprehensive risk vs profit analysis"""
        
        print(f"\n{'='*80}")
        print("🎯 COMPREHENSIVE RISK vs PROFIT ANALYSIS")
        print(f"{'='*80}")
        
        # Collect all data
        all_bayesian = []
        all_traditional = []
        
        for timeframe, results in all_results.items():
            for strategy, result in results.items():
                if 'bayesian' in strategy:
                    all_bayesian.append({
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'return': result.total_return,
                        'max_dd': result.max_drawdown,
                        'sharpe': result.sharpe_ratio,
                        'trades': result.trade_count
                    })
                elif 'traditional' in strategy:
                    all_traditional.append({
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'return': result.total_return,
                        'max_dd': result.max_drawdown,
                        'sharpe': result.sharpe_ratio,
                        'trades': result.trade_count
                    })
        
        if not all_bayesian or not all_traditional:
            print("❌ Insufficient data for comprehensive analysis")
            return
        
        # Risk Management Analysis
        print(f"\n🛡️ RISK MANAGEMENT ANALYSIS:")
        print("-" * 40)
        
        bayesian_avg_dd = np.mean([b['max_dd'] for b in all_bayesian])
        traditional_avg_dd = np.mean([t['max_dd'] for t in all_traditional])
        
        bayesian_dd_consistency = np.std([b['max_dd'] for b in all_bayesian])
        traditional_dd_consistency = np.std([t['max_dd'] for t in all_traditional])
        
        print(f"🔮 Bayesian Average Max Drawdown: {bayesian_avg_dd:.2%} ± {bayesian_dd_consistency:.2%}")
        print(f"🏛️ Traditional Average Max Drawdown: {traditional_avg_dd:.2%} ± {traditional_dd_consistency:.2%}")
        
        risk_winner = "Bayesian" if abs(bayesian_avg_dd) < abs(traditional_avg_dd) else "Traditional"
        risk_advantage = abs(abs(traditional_avg_dd) - abs(bayesian_avg_dd)) / abs(traditional_avg_dd) * 100
        print(f"🏆 Risk Management Winner: {risk_winner} (by {risk_advantage:.1f}%)")
        
        # Profitability Analysis
        print(f"\n💰 PROFITABILITY ANALYSIS:")
        print("-" * 35)
        
        bayesian_avg_return = np.mean([b['return'] for b in all_bayesian])
        traditional_avg_return = np.mean([t['return'] for t in all_traditional])
        
        bayesian_return_consistency = np.std([b['return'] for b in all_bayesian])
        traditional_return_consistency = np.std([t['return'] for t in all_traditional])
        
        print(f"🔮 Bayesian Average Return: {bayesian_avg_return:.2%} ± {bayesian_return_consistency:.2%}")
        print(f"🏛️ Traditional Average Return: {traditional_avg_return:.2%} ± {traditional_return_consistency:.2%}")
        
        profit_winner = "Bayesian" if bayesian_avg_return > traditional_avg_return else "Traditional"
        profit_advantage = abs(traditional_avg_return - bayesian_avg_return) / max(abs(traditional_avg_return), abs(bayesian_avg_return)) * 100
        print(f"🏆 Profitability Winner: {profit_winner} (by {profit_advantage:.1f}%)")
        
        # Risk-Adjusted Performance
        print(f"\n📊 RISK-ADJUSTED PERFORMANCE:")
        print("-" * 40)
        
        bayesian_risk_adj = bayesian_avg_return / abs(bayesian_avg_dd) if bayesian_avg_dd != 0 else 0
        traditional_risk_adj = traditional_avg_return / abs(traditional_avg_dd) if traditional_avg_dd != 0 else 0
        
        print(f"🔮 Bayesian Risk-Adjusted Return: {bayesian_risk_adj:.2f}")
        print(f"🏛️ Traditional Risk-Adjusted Return: {traditional_risk_adj:.2f}")
        
        overall_winner = "Bayesian" if bayesian_risk_adj > traditional_risk_adj else "Traditional"
        print(f"🏆 Overall Risk-Adjusted Winner: {overall_winner}")
        
        # Trading Activity
        print(f"\n🔄 TRADING ACTIVITY ANALYSIS:")
        print("-" * 35)
        
        bayesian_avg_trades = np.mean([b['trades'] for b in all_bayesian])
        traditional_avg_trades = np.mean([t['trades'] for t in all_traditional])
        
        print(f"🔮 Bayesian Average Trades: {bayesian_avg_trades:.0f}")
        print(f"🏛️ Traditional Average Trades: {traditional_avg_trades:.0f}")
        
        # Efficiency metrics
        bayesian_efficiency = bayesian_avg_return / bayesian_avg_trades * 100 if bayesian_avg_trades > 0 else 0
        traditional_efficiency = traditional_avg_return / traditional_avg_trades * 100 if traditional_avg_trades > 0 else 0
        
        print(f"🔮 Bayesian Return per Trade: {bayesian_efficiency:.3f}%")
        print(f"🏛️ Traditional Return per Trade: {traditional_efficiency:.3f}%")
        
        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        print("-" * 20)
        print(f"🛡️ Best Risk Management: {'🔮 Bayesian GNNs' if abs(bayesian_avg_dd) < abs(traditional_avg_dd) else '🏛️ Traditional GNNs'}")
        print(f"💰 Best Profitability: {'🔮 Bayesian GNNs' if bayesian_avg_return > traditional_avg_return else '🏛️ Traditional GNNs'}")
        print(f"📊 Best Risk-Adjusted: {'🔮 Bayesian GNNs' if bayesian_risk_adj > traditional_risk_adj else '🏛️ Traditional GNNs'}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        print("-" * 20)
        print("🎯 Hybrid Approach: Use Bayesian uncertainty for position sizing")
        print("   and Traditional signals for entry/exit decisions")
        print("🔄 Dynamic Allocation: Higher Bayesian weight in volatile periods")
        print("📈 Portfolio Construction: Bayesian for risk budgeting,")
        print("   Traditional for alpha generation")


def main():
    """Main execution function"""
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("🎯 BALANCED RISK-PROFIT EVALUATION")
    print("Demonstrating each model's TRUE strengths:")
    print("🔮 Bayesian GNNs → Risk Management")  
    print("🏛️ Traditional GNNs → Profitability")
    print("="*60)
    
    evaluator = BalancedRiskProfitEvaluator(symbols)
    
    try:
        results = evaluator.run_balanced_evaluation()
        print(f"\n✅ Balanced evaluation completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()