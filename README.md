# Bayesian GNN Trading System

A sophisticated algorithmic trading system that uses Bayesian Graph Neural Networks to predict stock prices and optimize trading strategies with uncertainty quantification.

## Features

- **Bayesian Graph Neural Networks**: Incorporates uncertainty quantification for more robust predictions
- **Graph-based Stock Relationships**: Models correlations between stocks using graph structures
- **Multiple Trading Strategies**: Momentum, mean reversion, and risk parity strategies
- **Comprehensive Backtesting**: Walk-forward analysis with detailed performance metrics
- **Risk Management**: Advanced position sizing and portfolio risk controls
- **AI Integration**: OpenAI and Gemini API support for enhanced market analysis
- **Real-time Uncertainty**: Epistemic and aleatoric uncertainty estimation

## Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd bayesian_gnn

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Configuration

The system uses a `config.json` file for all settings:

```bash
# Copy template and edit with your API keys
cp config.json.template config.json
# Edit config.json with your actual API keys and preferences
```

**Important:** Your `config.json` file contains sensitive API keys and is automatically ignored by git for security.

Example configuration:
```json
{
  "api_keys": {
    "openai": "sk-proj-your-openai-key-here",
    "gemini": "AIzaSy-your-gemini-key-here"
  },
  "trading_config": {
    "initial_capital": 100000,
    "transaction_cost": 0.001,
    "max_position_size": 0.1
  }
}
```

You can also set API keys via environment variables (config.json takes precedence):
```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 3. Run the System

```bash
# Full pipeline (training + backtesting + AI analysis)
python main.py --mode full --symbols AAPL MSFT GOOGL AMZN TSLA

# Training only
python main.py --mode train --symbols AAPL MSFT GOOGL

# Backtesting only
python main.py --mode backtest --symbols AAPL MSFT GOOGL

# Custom date range
python main.py --start-date 2020-01-01 --end-date 2023-12-31
```

## System Architecture

### Core Components

1. **Data Pipeline** (`src/data/`)
   - `data_loader.py`: Stock data fetching and technical indicators
   - `graph_builder.py`: Correlation-based graph construction

2. **Bayesian Models** (`src/models/`)
   - `bayesian_layers.py`: Bayesian neural network components
   - `bayesian_gnn.py`: Graph neural network architectures

3. **Trading Engine** (`src/trading/`)
   - `strategy.py`: Trading strategy implementations
   - `backtester.py`: Comprehensive backtesting framework
   - `risk_manager.py`: Risk management and position sizing

4. **AI Integration** (`src/utils/`)
   - `api_client.py`: OpenAI and Gemini API integration

### Model Architecture

The system uses a Bayesian Graph Neural Network that:

- **Captures Uncertainty**: Uses variational inference to quantify prediction uncertainty
- **Models Relationships**: Graph structure captures stock correlations and market relationships
- **Temporal Dynamics**: Incorporates time series patterns with LSTM/GRU layers
- **Risk-Aware**: Uncertainty estimates guide position sizing and risk management

## Trading Strategies

### 1. Bayesian Momentum Strategy
- Uses prediction confidence and uncertainty for position sizing
- Kelly Criterion with uncertainty adjustments
- Dynamic risk management based on model confidence

### 2. Bayesian Mean Reversion Strategy
- Statistical mean reversion with Bayesian updates
- Uncertainty-adjusted entry/exit signals
- Adaptive threshold based on model predictions

### 3. Risk Parity Strategy
- Inverse volatility weighting using uncertainty estimates
- Portfolio rebalancing based on risk contributions
- Dynamic allocation optimization

## Risk Management

### Position Sizing Methods
- **Kelly Criterion**: Optimal position sizing with uncertainty adjustment
- **Fixed Risk**: Percentage-based risk per trade
- **Volatility Sizing**: Inverse volatility weighting
- **Equal Weight**: Simple equal allocation

### Risk Controls
- Maximum position size limits
- Portfolio concentration limits
- Stop-loss orders
- Correlation-based diversification
- Value at Risk (VaR) monitoring

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total, annual, and risk-adjusted returns
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Risk Measures**: VaR, CVaR, beta, correlation
- **Trade Analysis**: Win rate, profit factor, trade frequency
- **Uncertainty Metrics**: Prediction confidence, epistemic/aleatoric uncertainty

## API Integration

### OpenAI Integration
- Market sentiment analysis
- Trading insight generation
- Portfolio optimization suggestions

### Gemini Integration
- Alternative AI analysis
- Market outlook assessment
- Risk assessment

## Configuration

### Model Parameters
```python
model_config = {
    'input_dim': 13,  # Technical indicators
    'hidden_dims': [64, 32, 16],
    'output_dim': 1,
    'gnn_type': 'GCN',  # or 'GAT'
    'dropout': 0.1,
    'prior_std': 0.1,
    'use_uncertainty': True
}
```

### Trading Parameters
```python
strategy_config = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,
    'confidence_threshold': 0.6,
    'max_position_size': 0.1,
    'uncertainty_penalty': 2.0
}
```

## Example Usage

### Training a Model
```python
from src.trainer import BayesianGNNTrainer, prepare_training_data
from src.models.bayesian_gnn import BayesianGNN

# Prepare data
data = prepare_training_data(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')

# Create and train model
model = BayesianGNN(input_dim=13, hidden_dims=[64, 32], output_dim=1)
trainer = BayesianGNNTrainer(model)
trainer.train(train_loader, val_loader, n_epochs=100)
```

### Running Backtest
```python
from src.trading.strategy import BayesianMomentumStrategy
from src.trading.backtester import Backtester

strategy = BayesianMomentumStrategy()
backtester = Backtester()
results = backtester.run_backtest(strategy, predictions, uncertainties, price_data)
```

### AI Analysis
```python
from src.utils.api_client import APIClient

api_client = APIClient()
insights = api_client.generate_trading_insights(market_data, predictions)
```

## Advanced Features

### Walk-Forward Analysis
```python
from src.trading.backtester import WalkForwardAnalysis

wfa = WalkForwardAnalysis(backtester)
results = wfa.run_analysis(strategy, predictions, uncertainties, price_data)
```

### Ensemble Models
```python
from src.models.bayesian_gnn import EnsembleBayesianGNN

ensemble = EnsembleBayesianGNN(n_models=5)
predictions = ensemble.predict_with_uncertainty(x, edge_index)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Disclaimer

This system is for educational and research purposes. Always:
- Test with paper money first
- Understand the risks involved
- Consider transaction costs and market impact
- Validate model performance out-of-sample
- Implement proper risk management

**Past performance does not guarantee future results.**

## Backtest Results

### Performance Comparison (Jan 1 - Jun 30, 2025)

We conducted a comprehensive backtest comparing our Bayesian GNN strategies against S&P 500 and simple trading strategies over a 6-month period in 2025.

#### 📊 **Performance Rankings by Total Return:**
1. **🥇 Random Strategy: 6.77%** (Best absolute return)
2. **🥈 Bayesian Momentum: 5.35%** (Our AI strategy)
3. **🥉 S&P 500 Benchmark: 5.19%** (Market baseline)
4. **4th Moving Average: 2.51%**
5. **5th RSI Strategy: 1.34%**
6. **6th Buy & Hold: -0.63%**

#### 📈 **Risk-Adjusted Performance (Sharpe Ratio):**
1. **🥇 Moving Average: 1.19** (Best risk-adjusted return)
2. **🥈 Random Strategy: 1.05**
3. **🥉 Bayesian Momentum: 0.52**
4. **4th S&P 500: 0.45**
5. **5th RSI Strategy: 0.19**
6. **6th Buy & Hold: -0.05**

#### 📋 **Complete Performance Metrics:**

| Strategy | Total Return | Annual Return | Volatility | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|----------|--------------|---------------|------------|--------------|---------------|----------|---------|
| **S&P 500 (Benchmark)** | 5.19% | 11.1% | 24.5% | 0.45 | -18.9% | N/A | N/A |
| **Buy & Hold** | -0.63% | -1.3% | 24.1% | -0.05 | -20.6% | 100% | 4 |
| **Moving Average** | 2.51% | 5.2% | 4.4% | 1.19 | -2.7% | 100% | 3 |
| **RSI Strategy** | 1.34% | 2.8% | 14.9% | 0.19 | -10.6% | 53.3% | 15 |
| **Random Strategy** | 6.77% | 14.5% | 13.8% | 1.05 | -7.0% | 66.7% | 21 |
| **Bayesian Momentum** | 5.35% | 11.4% | 22.0% | 0.52 | -13.9% | 70.3% | 427 |

#### 🎯 **Key Insights:**

**Market Outperformance:**
- **2 out of 5 strategies** beat the S&P 500 benchmark
- Bayesian Momentum and Random Strategy both outperformed the market
- Our AI strategy achieved competitive returns with controlled risk

**Risk Management Excellence:**
- **Moving Average** strategy delivered the best risk-adjusted returns
- **Bayesian Momentum** maintained a solid 70.3% win rate across 427 trades
- Low volatility strategies showed superior Sharpe ratios

**Strategy Performance Analysis:**
- **Bayesian Momentum**: Solid 5.35% return with active trading (427 trades)
- **Moving Average**: Conservative approach with excellent risk control
- **Random Strategy**: Surprisingly strong performance highlighting market inefficiencies

#### 🧪 **Test Configuration:**
- **Initial Capital**: $100,000
- **Transaction Cost**: 0.1%
- **Test Period**: January 1, 2025 - June 30, 2025
- **Assets**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **Benchmark**: S&P 500 (^GSPC)

#### 🚀 **Running Your Own Backtest:**

```bash
# Run comprehensive comparison backtest
python run_comparison_backtest.py

# Test specific strategies with custom parameters
python main.py --mode backtest --symbols AAPL MSFT GOOGL AMZN TSLA \
              --start-date 2025-01-01 --end-date 2025-06-30
```

**Note**: Results are based on historical simulation and do not guarantee future performance. Always test strategies thoroughly before live trading.

## License

MIT License - see LICENSE file for details.