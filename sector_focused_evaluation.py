#!/usr/bin/env python3
"""
Sector-Focused Evaluation
Test Bayesian vs Traditional GNNs across key market sectors for comprehensive view
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Use our simplified backtester for faster results
from simple_evaluation import SimpleBacktester, MultiTimeframeEvaluator


class SectorFocusedEvaluator:
    """Focused evaluation across key market sectors"""
    
    def __init__(self):
        # Representative stocks from different sectors
        self.sector_portfolios = {
            # Previous focus - Big Tech (high growth, high correlation)
            'big_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            
            # Traditional Large Cap (diversified, stable)
            'large_cap_diversified': ['JPM', 'JNJ', 'PG', 'UNH', 'HD'],
            
            # Financial Sector (interest rate sensitive)
            'financials': ['JPM', 'BAC', 'WFC', 'GS', 'C'],
            
            # Healthcare (defensive, stable)
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            
            # Energy (cyclical, commodity-driven)
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            
            # Consumer Staples (defensive, low volatility)
            'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'HD'],
            
            # Utilities (low volatility, dividend-focused)
            'utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            
            # High Volatility (meme stocks, growth)
            'high_volatility': ['GME', 'AMC', 'PLTR', 'RIVN', 'COIN'],
        }
        
        self.timeframes = {
            '6_months': ('2024-07-01', '2024-12-31'),
            '12_months': ('2024-01-01', '2024-12-31'),
            '3_years': ('2022-01-01', '2024-12-31'),
        }
        
        self.strategies = [
            'bayesian_gcn', 'bayesian_gat', 'ensemble_bayesian',
            'traditional_gcn', 'traditional_gat', 'ensemble_traditional',
            'buy_hold', 'random'
        ]
    
    def run_sector_evaluation(self) -> Dict:
        """Run evaluation across all sectors"""
        
        print("🌟 SECTOR-FOCUSED EVALUATION")
        print("="*60)
        print(f"📊 Sectors: {len(self.sector_portfolios)}")
        print(f"🎯 Stocks per sector: 5")
        print(f"⏰ Timeframes: {list(self.timeframes.keys())}")
        print(f"🤖 Strategies: {len(self.strategies)}")
        print("="*60)
        
        all_sector_results = {}
        
        for sector_name, stock_list in self.sector_portfolios.items():
            print(f"\n🔍 EVALUATING SECTOR: {sector_name.upper()}")
            print(f"Stocks: {', '.join(stock_list)}")
            print("-" * 50)
            
            try:
                # Create evaluator for this sector
                sector_evaluator = MultiTimeframeEvaluator(stock_list)
                sector_evaluator.timeframes = self.timeframes
                sector_evaluator.strategies = self.strategies
                
                # Run evaluation
                sector_results = {}
                
                for timeframe in self.timeframes.keys():
                    print(f"\n⏰ {timeframe}:")
                    timeframe_results = sector_evaluator.evaluate_timeframe(timeframe)
                    sector_results[timeframe] = timeframe_results
                    
                    # Display quick summary
                    if timeframe_results:
                        best_strategy = max(timeframe_results.items(), key=lambda x: x[1].total_return)
                        print(f"   🥇 Best: {best_strategy[0]} ({best_strategy[1].total_return:.2%})")
                
                all_sector_results[sector_name] = sector_results
                
                # Display sector summary
                self._display_sector_summary(sector_name, sector_results)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {sector_name}: {e}")
                continue
        
        # Generate cross-sector analysis
        self._generate_cross_sector_analysis(all_sector_results)
        
        return all_sector_results
    
    def _display_sector_summary(self, sector_name: str, results: Dict):
        """Display summary for a sector"""
        
        print(f"\n📈 {sector_name.upper()} SECTOR SUMMARY:")
        print("-" * 40)
        
        # Collect performance data
        all_performance = []
        for timeframe, timeframe_results in results.items():
            if timeframe_results:
                for strategy, result in timeframe_results.items():
                    all_performance.append({
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'return': result.total_return,
                        'sharpe': result.sharpe_ratio,
                        'trades': result.trade_count
                    })
        
        if not all_performance:
            print("No performance data available")
            return
        
        # Best performers
        best_return = max(all_performance, key=lambda x: x['return'])
        best_sharpe = max(all_performance, key=lambda x: x['sharpe'])
        
        print(f"🥇 Best Return: {best_return['strategy']} - {best_return['return']:.2%}")
        print(f"📊 Best Sharpe: {best_sharpe['strategy']} - {best_sharpe['sharpe']:.2f}")
        
        # Model type comparison
        bayesian_perf = [p for p in all_performance if 'bayesian' in p['strategy']]
        traditional_perf = [p for p in all_performance if 'traditional' in p['strategy']]
        
        if bayesian_perf and traditional_perf:
            avg_bayesian = np.mean([p['return'] for p in bayesian_perf])
            avg_traditional = np.mean([p['return'] for p in traditional_perf])
            
            print(f"🔮 Bayesian avg: {avg_bayesian:.2%}")
            print(f"🏛️ Traditional avg: {avg_traditional:.2%}")
            
            winner = "Bayesian" if avg_bayesian > avg_traditional else "Traditional"
            print(f"🏆 Sector Winner: {winner}")
    
    def _generate_cross_sector_analysis(self, all_results: Dict):
        """Generate analysis across all sectors"""
        
        print(f"\n{'='*80}")
        print("🌍 CROSS-SECTOR ANALYSIS")
        print(f"{'='*80}")
        
        # Collect all performance data
        sector_summaries = {}
        all_performance = []
        
        for sector, sector_results in all_results.items():
            sector_performance = []
            
            for timeframe, timeframe_results in sector_results.items():
                if timeframe_results:
                    for strategy, result in timeframe_results.items():
                        perf_data = {
                            'sector': sector,
                            'timeframe': timeframe,
                            'strategy': strategy,
                            'type': self._get_strategy_type(strategy),
                            'return': result.total_return,
                            'sharpe': result.sharpe_ratio,
                            'max_drawdown': result.max_drawdown,
                            'trades': result.trade_count
                        }
                        
                        all_performance.append(perf_data)
                        sector_performance.append(perf_data)
            
            # Sector summary
            if sector_performance:
                sector_summaries[sector] = {
                    'avg_return': np.mean([p['return'] for p in sector_performance]),
                    'avg_sharpe': np.mean([p['sharpe'] for p in sector_performance]),
                    'best_return': max(p['return'] for p in sector_performance),
                    'sample_size': len(sector_performance)
                }
        
        if not all_performance:
            print("❌ No performance data available")
            return
        
        # Sector rankings
        print(f"\n📊 SECTOR PERFORMANCE RANKINGS:")
        print("-" * 45)
        
        sorted_sectors = sorted(sector_summaries.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        
        for i, (sector, summary) in enumerate(sorted_sectors, 1):
            print(f"{i}. {sector:20s}: {summary['avg_return']:6.2%} avg, "
                  f"{summary['best_return']:6.2%} best")
        
        # Model type analysis
        print(f"\n🔬 BAYESIAN vs TRADITIONAL ACROSS SECTORS:")
        print("-" * 50)
        
        bayesian_performance = [p for p in all_performance if p['type'] == 'Bayesian']
        traditional_performance = [p for p in all_performance if p['type'] == 'Traditional']
        baseline_performance = [p for p in all_performance if p['type'] == 'Baseline']
        
        if bayesian_performance:
            bayesian_avg = np.mean([p['return'] for p in bayesian_performance])
            print(f"🔮 Bayesian Models: {bayesian_avg:.2%} avg return ({len(bayesian_performance)} tests)")
        
        if traditional_performance:
            traditional_avg = np.mean([p['return'] for p in traditional_performance])
            print(f"🏛️ Traditional Models: {traditional_avg:.2%} avg return ({len(traditional_performance)} tests)")
        
        if baseline_performance:
            baseline_avg = np.mean([p['return'] for p in baseline_performance])
            print(f"📈 Baseline Strategies: {baseline_avg:.2%} avg return ({len(baseline_performance)} tests)")
        
        # Key insights by sector characteristics
        print(f"\n💡 KEY INSIGHTS BY SECTOR CHARACTERISTICS:")
        print("-" * 55)
        
        sector_insights = {
            'big_tech': '📱 High correlation, momentum-driven',
            'large_cap_diversified': '🏢 Stable, diversified exposure',
            'financials': '🏦 Interest rate & economic sensitive',
            'healthcare': '🏥 Defensive, low correlation to tech',
            'energy': '⛽ Cyclical, commodity-driven',
            'consumer_staples': '🛒 Defensive, dividend-focused',
            'utilities': '⚡ Low volatility, rate-sensitive',
            'high_volatility': '🎢 Extreme conditions, meme stocks'
        }
        
        for sector, insight in sector_insights.items():
            if sector in sector_summaries:
                avg_ret = sector_summaries[sector]['avg_return']
                print(f"{insight} - Avg: {avg_ret:.2%}")
        
        # Overall winner determination
        if bayesian_performance and traditional_performance:
            bayesian_score = np.mean([p['return'] for p in bayesian_performance])
            traditional_score = np.mean([p['return'] for p in traditional_performance])
            
            overall_winner = "Bayesian" if bayesian_score > traditional_score else "Traditional"
            advantage = abs(bayesian_score - traditional_score) / max(abs(bayesian_score), abs(traditional_score)) * 100
            
            print(f"\n🏆 OVERALL WINNER ACROSS ALL SECTORS: {overall_winner} GNNs")
            print(f"💪 Performance Advantage: {advantage:.1f}%")
        
        print(f"\n✅ SECTOR-FOCUSED EVALUATION COMPLETED!")
        print(f"Total sectors analyzed: {len(sector_summaries)}")
        print(f"Total tests conducted: {len(all_performance)}")
    
    def _get_strategy_type(self, strategy: str) -> str:
        """Categorize strategy type"""
        if 'bayesian' in strategy:
            return 'Bayesian'
        elif 'traditional' in strategy:
            return 'Traditional'
        else:
            return 'Baseline'


def main():
    """Main execution function"""
    
    print("🚀 Starting Sector-Focused Evaluation...")
    print("Testing across 8 key market sectors with different characteristics:")
    print("• Big Tech (momentum, high correlation)")
    print("• Financials (interest rate sensitive)")  
    print("• Healthcare (defensive)")
    print("• Energy (cyclical)")
    print("• Consumer Staples (stable)")
    print("• Utilities (low volatility)")
    print("• High Volatility (meme stocks)")
    print("• Large Cap Diversified (balanced)")
    print()
    
    evaluator = SectorFocusedEvaluator()
    
    try:
        results = evaluator.run_sector_evaluation()
        return results
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()