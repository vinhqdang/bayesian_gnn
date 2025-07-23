# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+
- Your OpenAI and/or Gemini API keys

## 5-Minute Setup

### 1. Install
```bash
pip install -r requirements.txt
python setup.py
```

### 2. Configure API Keys
```bash
# Copy template
cp config.json.template config.json

# Edit config.json and add your keys:
{
  "api_keys": {
    "openai": "sk-proj-YOUR-OPENAI-KEY-HERE",
    "gemini": "AIzaSy-YOUR-GEMINI-KEY-HERE"
  }
}
```

### 3. Test Configuration
```bash
python test_config.py
```

### 4. Run Your First Analysis
```bash
# Quick demo with 5 popular stocks
python main.py --mode full --symbols AAPL MSFT GOOGL AMZN TSLA
```

## What You'll See

1. **Configuration Status** - Verification of API keys and settings
2. **Training Phase** - Bayesian GNN model training on historical data
3. **Predictions** - Model predictions with uncertainty estimates
4. **Backtesting** - Multiple trading strategies tested
5. **AI Analysis** - AI-powered market insights (if APIs configured)

## Quick Commands

```bash
# Full pipeline (recommended first run)
python main.py --mode full

# Just train the model
python main.py --mode train --symbols AAPL MSFT GOOGL

# Just backtest (uses random predictions)
python main.py --mode backtest

# Custom date range
python main.py --start-date 2020-01-01 --end-date 2023-12-31

# Help
python main.py --help
```

## Expected Output

```
=== Bayesian GNN Trading System ===

=== Configuration Status ===
Config file exists: ✓
OpenAI API key: ✓
Gemini API key: ✓
Trading config valid: ✓
Model config valid: ✓

✓ Configuration looks good!
Using device: cpu

=== Training Phase ===
Loading stock data...
Successfully fetched data for AAPL
Successfully fetched data for MSFT
...
Training model...
Epoch 1/50: Train Loss: 0.1234, Val Loss: 0.1456
...

=== Predictions ===
AAPL: 0.0234 ± 0.0123
MSFT: 0.0187 ± 0.0098
...

=== Backtesting BayesianMomentumStrategy ===
Total Return: 12.34%
Annual Return: 15.67%
Sharpe Ratio: 1.23
...

=== AI Analysis ===
AI Insights:
- The model shows high confidence in tech stocks
- Risk Assessment: medium
- Market Outlook: cautiously optimistic
```

## Configuration Options

### Trading Parameters
```json
"trading_config": {
  "initial_capital": 100000,      // Starting capital
  "transaction_cost": 0.001,      // 0.1% transaction cost
  "max_position_size": 0.1,       // Max 10% per position
  "risk_per_trade": 0.02,         // 2% risk per trade
  "confidence_threshold": 0.6     // Minimum confidence for trades
}
```

### Model Parameters
```json
"model_config": {
  "hidden_dims": [64, 32, 16],    // Neural network architecture
  "dropout": 0.1,                 // Dropout rate
  "gnn_type": "GCN",             // Graph neural network type
  "use_uncertainty": true         // Enable uncertainty quantification
}
```

## Next Steps

1. **Paper Trading**: Start with small amounts or paper trading
2. **Backtest Analysis**: Review the backtest results carefully
3. **Risk Management**: Adjust position sizes and risk parameters
4. **Model Tuning**: Experiment with different model architectures
5. **Strategy Development**: Create custom trading strategies

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
```

**API Errors**
- Check your API keys in `config.json`
- Verify internet connection
- Check API quotas/limits

**Data Loading Issues**
- Check internet connection
- Try different date ranges
- Some stocks may not have data for certain periods

**Memory Issues**
- Reduce batch size or number of stocks
- Use CPU instead of GPU for smaller datasets

### Getting Help

1. Check the full README.md for detailed documentation
2. Run `python test_config.py` to diagnose issues
3. Review the logs for error messages
4. Start with fewer stocks and shorter date ranges

## Safety Reminders ⚠️

- **Start with paper trading or very small amounts**
- **Past performance doesn't guarantee future results**
- **Always implement proper risk management**
- **Understand the model's limitations and uncertainties**
- **Consider transaction costs and market impact**

Happy trading! 📈