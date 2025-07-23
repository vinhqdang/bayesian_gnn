#!/usr/bin/env python3
"""
Bayesian GNN Trading System
Main execution script with API integrations
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import StockDataLoader, MarketDataPreprocessor
from data.graph_builder import StockGraphBuilder
from models.bayesian_gnn import BayesianGNN, TemporalBayesianGNN
from trading.strategy import BayesianMomentumStrategy, BayesianMeanReversionStrategy
from trading.backtester import Backtester, BacktestResults
from trading.risk_manager import RiskManager, PositionSizer
from utils.api_client import APIClient
from utils.config import get_config
from trainer import BayesianGNNTrainer, prepare_training_data

def setup_environment():
    """Setup environment and check configuration"""
    print("=== Bayesian GNN Trading System ===")
    
    # Load and validate configuration
    config = get_config()
    config.print_status()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device, config

def train_model(symbols, start_date, end_date, device, config):
    """Train the Bayesian GNN model"""
    print("\n=== Training Phase ===")
    
    # Prepare data
    data = prepare_training_data(symbols, start_date, end_date)
    
    # Get model configuration from config
    model_config_obj = config.get_model_config()
    model_config = {
        'input_dim': data['X_train'].shape[-1],
        'hidden_dims': model_config_obj.hidden_dims,
        'output_dim': 1,
        'gnn_type': model_config_obj.gnn_type,
        'dropout': model_config_obj.dropout,
        'prior_std': model_config_obj.prior_std,
        'use_uncertainty': model_config_obj.use_uncertainty
    }
    
    # Create and train model
    model = BayesianGNN(**model_config)
    trainer = BayesianGNNTrainer(model, device)
    
    # Create simple data loaders for demonstration
    from torch.utils.data import DataLoader, TensorDataset
    
    # Flatten features for training
    X_flat = data['X_train'].reshape(-1, data['X_train'].shape[-1])
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_flat),
        data['graph_data'].edge_index.unsqueeze(0).repeat(len(X_flat), 1, 1),
        data['graph_data'].edge_attr.unsqueeze(0).repeat(len(X_flat), 1),
        torch.FloatTensor(data['y_train'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = train_loader  # Simplified for demo
    
    # Train
    print("Training model...")
    trainer.train(train_loader, val_loader, n_epochs=50, patience=10)
    
    return model, data

def generate_predictions(model, data, device):
    """Generate predictions with uncertainty"""
    print("\n=== Generating Predictions ===")
    
    model.eval()
    predictions = {}
    uncertainties = {}
    
    with torch.no_grad():
        for i, symbol in enumerate(data['symbols']):
            # Use test data for predictions
            if i < len(data['X_test']):
                x = torch.FloatTensor(data['X_test'][i:i+1]).to(device)
                
                # Multiple forward passes for uncertainty estimation
                pred_samples = []
                for _ in range(100):
                    if model.use_uncertainty:
                        mean, var = model(x, data['graph_data'].edge_index, 
                                        data['graph_data'].edge_attr, sample=True)
                        pred = torch.normal(mean, torch.sqrt(var))
                    else:
                        pred = model(x, data['graph_data'].edge_index, 
                                   data['graph_data'].edge_attr, sample=True)
                    pred_samples.append(pred.cpu().numpy())
                
                pred_samples = np.array(pred_samples)
                predictions[symbol] = pred_samples.mean()
                uncertainties[symbol] = pred_samples.std()
    
    return predictions, uncertainties

def run_backtest(strategy_class, predictions, uncertainties, symbols, start_date, end_date, config):
    """Run backtesting"""
    print(f"\n=== Backtesting {strategy_class.__name__} ===")
    
    # Get trading configuration
    trading_config = config.get_trading_config()
    
    # Load fresh data for backtesting
    data_loader = StockDataLoader(symbols, start_date, end_date)
    price_data = data_loader.fetch_data()
    
    # Convert predictions to time series format (simplified)
    pred_series = {}
    uncertainty_series = {}
    
    for symbol in symbols:
        if symbol in predictions:
            # Create dummy time series (in practice, you'd have time-indexed predictions)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = [d for d in dates if d.weekday() < 5]  # Trading days
            
            pred_series[symbol] = [predictions[symbol]] * len(dates)
            uncertainty_series[symbol] = [uncertainties.get(symbol, 0.1)] * len(dates)
    
    # Create strategy and backtester using config
    strategy = strategy_class(
        initial_capital=trading_config.initial_capital,
        transaction_cost=trading_config.transaction_cost,
        confidence_threshold=trading_config.confidence_threshold,
        max_position_size=trading_config.max_position_size
    )
    backtester = Backtester(
        initial_capital=trading_config.initial_capital,
        transaction_cost=trading_config.transaction_cost
    )
    
    # Run backtest
    results = backtester.run_backtest(
        strategy, pred_series, uncertainty_series, price_data, start_date, end_date
    )
    
    return results

def analyze_with_ai(api_client, predictions, uncertainties, market_data):
    """Get AI-powered analysis"""
    print("\n=== AI Analysis ===")
    
    # Prepare data for AI analysis
    analysis_data = {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'market_summary': {
            'n_stocks': len(predictions),
            'avg_prediction': np.mean(list(predictions.values())),
            'avg_uncertainty': np.mean(list(uncertainties.values())),
            'prediction_range': [min(predictions.values()), max(predictions.values())]
        }
    }
    
    # Get AI insights
    try:
        insights = api_client.generate_trading_insights(analysis_data, predictions, method='gemini')
        print("AI Insights:")
        print(f"- {insights.get('insights', 'No insights available')}")
        print(f"- Risk Assessment: {insights.get('risk_assessment', 'Unknown')}")
        print(f"- Market Outlook: {insights.get('market_outlook', 'Neutral')}")
        
        if insights.get('recommended_actions'):
            print("- Recommended Actions:")
            for action in insights['recommended_actions']:
                print(f"  • {action}")
                
    except Exception as e:
        print(f"AI analysis failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Bayesian GNN Trading System')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], default='full',
                       help='Run mode: train only, backtest only, or full pipeline')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                       help='Stock symbols to analyze')
    parser.add_argument('--start-date', default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--api-method', choices=['openai', 'gemini'], default='gemini',
                       help='AI API to use for analysis')
    
    args = parser.parse_args()
    
    # Setup
    device, config = setup_environment()
    
    # Initialize API client
    api_client = APIClient(config)
    
    if args.mode in ['train', 'full']:
        # Training phase
        model, data = train_model(args.symbols, args.start_date, args.end_date, device, config)
        
        # Generate predictions
        predictions, uncertainties = generate_predictions(model, data, device)
        
        print("\nPredictions Summary:")
        for symbol in predictions:
            print(f"{symbol}: {predictions[symbol]:.4f} ± {uncertainties[symbol]:.4f}")
    
    if args.mode in ['backtest', 'full']:
        # Backtesting phase
        if args.mode == 'backtest':
            # Generate dummy predictions for backtest-only mode
            predictions = {symbol: np.random.normal(0.02, 0.05) for symbol in args.symbols}
            uncertainties = {symbol: np.random.uniform(0.05, 0.15) for symbol in args.symbols}
        
        # Test multiple strategies
        strategies = [BayesianMomentumStrategy, BayesianMeanReversionStrategy]
        
        for strategy_class in strategies:
            results = run_backtest(strategy_class, predictions, uncertainties, 
                                 args.symbols, args.start_date, args.end_date, config)
            
            print(f"\nResults for {strategy_class.__name__}:")
            print(f"Total Return: {results.total_return:.2%}")
            print(f"Annual Return: {results.annual_return:.2%}")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {results.max_drawdown:.2%}")
            print(f"Win Rate: {results.win_rate:.2%}")
            print(f"Total Trades: {results.trade_count}")
    
    # AI Analysis
    if api_client and args.mode == 'full':
        analyze_with_ai(api_client, predictions, uncertainties, {})
    
    print("\n=== Analysis Complete ===")
    print("Next steps:")
    print("1. Review model performance and predictions")
    print("2. Analyze backtest results")
    print("3. Consider live trading with paper money first")
    print("4. Implement risk management controls")

if __name__ == "__main__":
    main()