#!/usr/bin/env python3
"""
Test script to verify configuration and API connectivity
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import get_config
from utils.api_client import APIClient

def test_configuration():
    """Test configuration loading"""
    print("=== Testing Configuration ===")
    
    try:
        config = get_config()
        config.print_status()
        
        # Test configuration objects
        api_config = config.get_api_config()
        trading_config = config.get_trading_config()
        model_config = config.get_model_config()
        data_config = config.get_data_config()
        
        print(f"\n📊 Trading Config:")
        print(f"   Initial Capital: ${trading_config.initial_capital:,.2f}")
        print(f"   Transaction Cost: {trading_config.transaction_cost:.3f}")
        print(f"   Max Position Size: {trading_config.max_position_size:.1%}")
        
        print(f"\n🧠 Model Config:")
        print(f"   Hidden Dimensions: {model_config.hidden_dims}")
        print(f"   GNN Type: {model_config.gnn_type}")
        print(f"   Use Uncertainty: {model_config.use_uncertainty}")
        
        print(f"\n📈 Data Config:")
        print(f"   Lookback Window: {data_config.lookback_window}")
        print(f"   Correlation Threshold: {data_config.correlation_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_api_connectivity():
    """Test API connectivity"""
    print("\n=== Testing API Connectivity ===")
    
    try:
        config = get_config()
        api_client = APIClient(config)
        
        # Test with sample data
        test_data = ["The market is showing positive sentiment today."]
        
        print("Testing Gemini API...")
        if api_client.gemini_model:
            try:
                result = api_client.analyze_market_sentiment(test_data, method='gemini')
                print(f"✅ Gemini API working - Sentiment: {result.get('sentiment_score', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Gemini API error: {e}")
        else:
            print("ℹ️  Gemini API not configured")
        
        print("Testing OpenAI API...")
        if api_client.openai_client:
            try:
                result = api_client.analyze_market_sentiment(test_data, method='openai')
                print(f"✅ OpenAI API working - Sentiment: {result.get('sentiment_score', 'N/A')}")
            except Exception as e:
                print(f"⚠️  OpenAI API error: {e}")
        else:
            print("ℹ️  OpenAI API not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_data_loading():
    """Test basic data loading functionality"""
    print("\n=== Testing Data Loading ===")
    
    try:
        from data.data_loader import StockDataLoader
        
        # Test with a single stock for quick testing
        print("Testing stock data loading...")
        loader = StockDataLoader(['AAPL'], '2023-01-01', '2023-01-31')
        data = loader.fetch_data()
        
        if 'AAPL' in data and not data['AAPL'].empty:
            print(f"✅ Stock data loaded successfully - {len(data['AAPL'])} rows")
            return True
        else:
            print("⚠️  No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    print("🧪 Bayesian GNN Trading System - Configuration Test\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("API Connectivity", test_api_connectivity),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 Test Summary:")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
        print("Run: python main.py --mode full")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please check configuration.")
        
        if not any(passed for _, passed in results):
            print("\n💡 Quick fixes:")
            print("   1. Install dependencies: pip install -r requirements.txt")
            print("   2. Add API keys to config.json")
            print("   3. Check internet connection for data loading")

if __name__ == "__main__":
    main()