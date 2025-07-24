#!/usr/bin/env python3
"""
Comprehensive Stock Universe Evaluation
Bayesian vs Traditional GNNs across diverse sectors, market caps, and volatility profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our existing evaluation framework
from enhanced_evaluation import RealDataBacktester, EnhancedBacktestResults


class ComprehensiveStockEvaluator:
    """Comprehensive evaluator across diverse stock universe"""
    
    def __init__(self):
        self.backtester = RealDataBacktester()
        
        # Comprehensive stock universe across sectors and market caps
        self.stock_universe = {
            # Large Cap Tech (Previous focus)
            'large_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            
            # Large Cap Non-Tech
            'large_cap_traditional': ['JPM', 'JNJ', 'PG', 'UNH', 'HD', 'V', 'WMT', 'DIS'],
            
            # Financial Sector
            'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB'],
            
            # Healthcare & Pharma
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY'],
            
            # Consumer Goods
            'consumer_goods': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX'],
            
            # Energy & Commodities
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO'],
            
            # Utilities (Low Volatility)
            'utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'PEG'],
            
            # Mid Cap Growth
            'mid_cap_growth': ['PLTR', 'SNOW', 'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'OKTA'],
            
            # Value Stocks
            'value_stocks': ['BRK-B', 'XOM', 'CVX', 'JNJ', 'PG', 'KO', 'VZ', 'T'],
            
            # High Volatility
            'high_volatility': ['GME', 'AMC', 'RIVN', 'LCID', 'COIN', 'ROKU', 'PTON', 'ZM'],
            
            # International/ADRs
            'international': ['TSM', 'ASML', 'NVO', 'TM', 'UL', 'SNY', 'NVS', 'RHHBY'],
            
            # REITs
            'reits': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'AVB', 'EQR'],
            
            # Small Cap
            'small_cap': ['SIRI', 'PLUG', 'TLRY', 'SNDL', 'WISH', 'CLOV', 'SOFI', 'OPEN']
        }
        
        # Evaluation timeframes
        self.timeframes = {
            '3_months': ('2024-10-01', '2024-12-31'),
            '6_months': ('2024-07-01', '2024-12-31'),
            '12_months': ('2024-01-01', '2024-12-31'),
            '3_years': ('2022-01-01', '2024-12-31'),
            '5_years': ('2020-01-01', '2024-12-31')
        }
        
        # GNN strategies
        self.strategies = [
            'bayesian_gcn', 'bayesian_gat', 'ensemble_bayesian',
            'traditional_gcn', 'traditional_gat', 'ensemble_traditional'
        ]
        
        # Results storage
        self.all_results = {}
        
    def run_comprehensive_evaluation(self) -> Dict:
        """Run evaluation across all stock categories and timeframes"""
        
        print("🌟 COMPREHENSIVE STOCK UNIVERSE EVALUATION")
        print("="*80)
        print(f"📊 Stock Categories: {len(self.stock_universe)}")
        print(f"🎯 Total Stocks: {sum(len(stocks) for stocks in self.stock_universe.values())}")
        print(f"⏰ Timeframes: {list(self.timeframes.keys())}")
        print(f"🤖 Strategies: {len(self.strategies)} GNN variants")
        print("="*80)
        
        category_results = {}
        
        for category_name, stock_list in self.stock_universe.items():
            print(f"\n🔍 EVALUATING CATEGORY: {category_name.upper()}")
            print(f"Stocks: {', '.join(stock_list)}")
            print("-" * 60)
            
            try:
                category_result = self._evaluate_stock_category(category_name, stock_list)
                category_results[category_name] = category_result
                
                # Display category summary
                self._display_category_summary(category_name, category_result)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {category_name}: {e}")
                continue
        
        # Generate comprehensive cross-category analysis
        self._generate_cross_category_analysis(category_results)
        
        return category_results
    
    def _evaluate_stock_category(self, category_name: str, stock_list: List[str]) -> Dict:
        """Evaluate all strategies for a specific stock category"""
        
        category_results = {}
        
        for timeframe in self.timeframes.keys():
            print(f"\n⏰ Timeframe: {timeframe}")
            
            start_date, end_date = self.timeframes[timeframe]
            
            # Fetch data for this category
            print(f"📥 Fetching data for {len(stock_list)} stocks...")
            price_data = self.backtester.fetch_real_data(stock_list, start_date, end_date)
            benchmark_data = self.backtester.fetch_sp500_benchmark(start_date, end_date)
            
            if not price_data:
                print("⚠️ No data available for this category/timeframe")
                continue
            
            print(f"✅ Successfully loaded data for {len(price_data)} stocks")
            
            timeframe_results = {}
            
            for strategy_name in self.strategies:
                try:
                    # Generate predictions
                    predictions, uncertainties = self.backtester.generate_enhanced_predictions(
                        strategy_name, price_data
                    )
                    
                    # Run backtest
                    result = self.backtester.run_enhanced_backtest(
                        strategy_name, predictions, uncertainties,
                        price_data, benchmark_data, start_date, end_date
                    )
                    
                    timeframe_results[strategy_name] = result
                    
                    # Brief result display
                    return_pct = result.total_return * 100
                    excess = result.benchmark_comparison.get('excess_return', 0) * 100 if result.benchmark_comparison else 0
                    print(f"   {strategy_name:20s}: {return_pct:6.1f}% (vs S&P: {excess:+5.1f}%)")
                    
                except Exception as e:
                    print(f"   ❌ {strategy_name}: Failed ({str(e)[:50]}...)")
            
            category_results[timeframe] = timeframe_results
        
        return category_results
    
    def _display_category_summary(self, category_name: str, results: Dict):
        """Display summary for a stock category"""
        
        if not results:
            return
        
        print(f"\n📈 {category_name.upper()} CATEGORY SUMMARY:")
        print("-" * 50)
        
        # Collect all performance data for this category
        all_performance = []
        for timeframe, timeframe_results in results.items():
            for strategy, result in timeframe_results.items():
                all_performance.append({
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'return': result.total_return,
                    'sharpe': result.sharpe_ratio,
                    'trades': result.trade_count,
                    'excess_return': result.benchmark_comparison.get('excess_return', 0) if result.benchmark_comparison else 0
                })
        
        if not all_performance:
            print("No performance data available")
            return
        
        # Best performers in this category
        best_return = max(all_performance, key=lambda x: x['return'])
        best_sharpe = max(all_performance, key=lambda x: x['sharpe'])
        best_vs_sp500 = max(all_performance, key=lambda x: x['excess_return'])
        
        print(f"🥇 Best Return: {best_return['strategy']} ({best_return['timeframe']}) - {best_return['return']:.2%}")
        print(f"📊 Best Sharpe: {best_sharpe['strategy']} ({best_sharpe['timeframe']}) - {best_sharpe['sharpe']:.2f}")
        print(f"🎯 Best vs S&P 500: {best_vs_sp500['strategy']} - {best_vs_sp500['excess_return']:+.2%}")
        
        # Bayesian vs Traditional for this category
        bayesian_perf = [p for p in all_performance if 'bayesian' in p['strategy']]
        traditional_perf = [p for p in all_performance if 'traditional' in p['strategy']]
        
        if bayesian_perf and traditional_perf:
            avg_bayesian = np.mean([p['return'] for p in bayesian_perf])
            avg_traditional = np.mean([p['return'] for p in traditional_perf])
            
            bayesian_sp500_wins = sum(1 for p in bayesian_perf if p['excess_return'] > 0)
            traditional_sp500_wins = sum(1 for p in traditional_perf if p['excess_return'] > 0)
            
            print(f"🔮 Bayesian avg: {avg_bayesian:.2%} (Beat S&P: {bayesian_sp500_wins}/{len(bayesian_perf)})")
            print(f"🏛️ Traditional avg: {avg_traditional:.2%} (Beat S&P: {traditional_sp500_wins}/{len(traditional_perf)})")
            
            winner = "Bayesian" if avg_bayesian > avg_traditional else "Traditional"
            print(f"🏆 Category Winner: {winner}")
    
    def _generate_cross_category_analysis(self, all_results: Dict):
        """Generate comprehensive analysis across all categories"""
        
        print(f"\n{'='*80}")
        print("🌍 CROSS-CATEGORY COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        # Collect all performance data
        all_performance = []
        category_summaries = {}
        
        for category, category_results in all_results.items():
            category_performance = []
            
            for timeframe, timeframe_results in category_results.items():
                for strategy, result in timeframe_results.items():
                    perf_data = {
                        'category': category,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'type': 'Bayesian' if 'bayesian' in strategy else 'Traditional',
                        'return': result.total_return,
                        'annual_return': result.annual_return,
                        'sharpe': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'trades': result.trade_count,
                        'excess_return': result.benchmark_comparison.get('excess_return', 0) if result.benchmark_comparison else 0,
                        'beat_sp500': result.benchmark_comparison.get('outperformed', False) if result.benchmark_comparison else False
                    }
                    
                    all_performance.append(perf_data)
                    category_performance.append(perf_data)
            
            # Category summary
            if category_performance:
                avg_return = np.mean([p['return'] for p in category_performance])
                avg_sharpe = np.mean([p['sharpe'] for p in category_performance])
                sp500_wins = sum(1 for p in category_performance if p['beat_sp500'])
                
                category_summaries[category] = {
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'sp500_win_rate': sp500_wins / len(category_performance) if category_performance else 0,
                    'sample_size': len(category_performance)
                }
        
        if not all_performance:
            print("❌ No performance data available for analysis")
            return
        
        # Overall best performers
        print(f"\n🏆 OVERALL BEST PERFORMERS ACROSS ALL CATEGORIES:")
        print("-" * 60)
        
        best_overall_return = max(all_performance, key=lambda x: x['return'])
        best_overall_sharpe = max(all_performance, key=lambda x: x['sharpe'])
        best_overall_excess = max(all_performance, key=lambda x: x['excess_return'])
        
        print(f"📈 Best Return: {best_overall_return['strategy']} on {best_overall_return['category']}")
        print(f"    {best_overall_return['return']:.2%} ({best_overall_return['timeframe']})")
        
        print(f"📊 Best Sharpe: {best_overall_sharpe['strategy']} on {best_overall_sharpe['category']}")
        print(f"    {best_overall_sharpe['sharpe']:.2f} ({best_overall_sharpe['timeframe']})")
        
        print(f"🎯 Best vs S&P 500: {best_overall_excess['strategy']} on {best_overall_excess['category']}")
        print(f"    {best_overall_excess['excess_return']:+.2%} ({best_overall_excess['timeframe']})")
        
        # Category performance ranking
        print(f"\n📊 CATEGORY PERFORMANCE RANKINGS:")
        print("-" * 50)
        
        sorted_categories = sorted(category_summaries.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        
        for i, (category, summary) in enumerate(sorted_categories, 1):
            print(f"{i:2d}. {category:20s}: {summary['avg_return']:6.2%} avg return, "
                  f"{summary['sp500_win_rate']:4.0%} beat S&P 500")
        
        # Model type analysis across all categories
        print(f"\n🔬 BAYESIAN vs TRADITIONAL ACROSS ALL CATEGORIES:")
        print("-" * 55)
        
        bayesian_performance = [p for p in all_performance if p['type'] == 'Bayesian']
        traditional_performance = [p for p in all_performance if p['type'] == 'Traditional']
        
        if bayesian_performance and traditional_performance:
            # Overall averages
            bayesian_avg_return = np.mean([p['return'] for p in bayesian_performance])
            traditional_avg_return = np.mean([p['return'] for p in traditional_performance])
            
            bayesian_avg_sharpe = np.mean([p['sharpe'] for p in bayesian_performance])
            traditional_avg_sharpe = np.mean([p['sharpe'] for p in traditional_performance])
            
            bayesian_sp500_wins = sum(1 for p in bayesian_performance if p['beat_sp500'])
            traditional_sp500_wins = sum(1 for p in traditional_performance if p['beat_sp500'])
            
            bayesian_win_rate = bayesian_sp500_wins / len(bayesian_performance)
            traditional_win_rate = traditional_sp500_wins / len(traditional_performance)
            
            print(f"🔮 Bayesian Models ({len(bayesian_performance)} tests):")
            print(f"   Average Return: {bayesian_avg_return:6.2%}")
            print(f"   Average Sharpe: {bayesian_avg_sharpe:6.2f}")
            print(f"   Beat S&P 500: {bayesian_win_rate:5.1%} of the time")
            
            print(f"🏛️ Traditional Models ({len(traditional_performance)} tests):")
            print(f"   Average Return: {traditional_avg_return:6.2%}")
            print(f"   Average Sharpe: {traditional_avg_sharpe:6.2f}")
            print(f"   Beat S&P 500: {traditional_win_rate:5.1%} of the time")
            
            # Determine overall winner
            bayesian_score = bayesian_avg_return + bayesian_avg_sharpe * 0.1 + bayesian_win_rate * 0.05
            traditional_score = traditional_avg_return + traditional_avg_sharpe * 0.1 + traditional_win_rate * 0.05
            
            overall_winner = "Bayesian" if bayesian_score > traditional_score else "Traditional"
            print(f"\n🏆 OVERALL WINNER ACROSS ALL STOCK CATEGORIES: {overall_winner} GNNs")
        
        # Sector-specific insights
        print(f"\n💡 KEY INSIGHTS BY SECTOR:")
        print("-" * 35)
        
        sector_insights = {
            'large_cap_tech': 'High growth, high volatility - test model robustness',
            'financials': 'Interest rate sensitive - economic cycle dependency',
            'utilities': 'Low volatility - conservative model testing',
            'energy': 'Commodity cyclical - regime change adaptability',
            'healthcare': 'Defensive sector - stability testing',
            'high_volatility': 'Extreme conditions - model stress testing',
            'value_stocks': 'Mean reversion potential - strategy effectiveness',
            'small_cap': 'High risk/reward - model scalability'
        }
        
        for category in sorted_categories[:5]:  # Top 5 performing categories
            category_name = category[0]
            if category_name in sector_insights:
                print(f"• {category_name:15s}: {sector_insights[category_name]}")
        
        print(f"\n{'='*80}")
        print("✅ COMPREHENSIVE EVALUATION COMPLETED!")
        print(f"Total Tests Conducted: {len(all_performance)}")
        print(f"Categories Analyzed: {len(all_results)}")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    
    print("🚀 Starting Comprehensive Stock Universe Evaluation...")
    print("This will test Bayesian vs Traditional GNNs across:")
    print("• 13+ stock categories (Tech, Finance, Healthcare, Energy, etc.)")
    print("• 100+ individual stocks")
    print("• 5 different timeframes") 
    print("• 6 GNN strategies")
    print("• Multiple market conditions and volatility profiles")
    print()
    
    evaluator = ComprehensiveStockEvaluator()
    
    try:
        results = evaluator.run_comprehensive_evaluation()
        print("\n🎉 Comprehensive evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()