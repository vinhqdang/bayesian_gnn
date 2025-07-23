#!/usr/bin/env python3
"""
Setup script for Bayesian GNN Trading System
"""

import os
import shutil
import json

def create_config_from_template():
    """Create config.json from template if it doesn't exist"""
    
    config_path = "config.json"
    template_path = "config.json.template"
    
    if os.path.exists(config_path):
        print(f"✓ {config_path} already exists")
        return
    
    if not os.path.exists(template_path):
        print(f"✗ Template file {template_path} not found")
        return
    
    # Copy template to config
    shutil.copy(template_path, config_path)
    print(f"✓ Created {config_path} from template")
    
    # Prompt user to add API keys
    print("\n📝 Please edit config.json and add your API keys:")
    print("   - OpenAI API key (optional)")
    print("   - Gemini API key (optional)")
    print("   - Adjust trading and model parameters as needed")

def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'torch',
        'torch_geometric', 
        'numpy',
        'pandas',
        'matplotlib',
        'yfinance',
        'openai',
        'google-generativeai',
        'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed")
        return True

def validate_config():
    """Validate configuration file"""
    
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"✗ {config_path} not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check structure
        required_sections = ['api_keys', 'trading_config', 'model_config', 'data_config']
        
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing section: {section}")
                return False
            else:
                print(f"✓ {section}")
        
        # Check API keys
        api_keys = config.get('api_keys', {})
        has_openai = api_keys.get('openai') and len(api_keys['openai']) > 10
        has_gemini = api_keys.get('gemini') and len(api_keys['gemini']) > 10
        
        if has_openai:
            print("✓ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not configured")
            
        if has_gemini:
            print("✓ Gemini API key configured")
        else:
            print("⚠️  Gemini API key not configured")
        
        if not has_openai and not has_gemini:
            print("⚠️  No AI API keys configured - AI features will be disabled")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in {config_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading {config_path}: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    
    directories = [
        'models',
        'data',
        'plots',
        'logs',
        'checkpoints'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def main():
    print("=== Bayesian GNN Trading System Setup ===\n")
    
    print("1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Setting up configuration...")
    create_config_from_template()
    
    print("\n3. Validating configuration...")
    config_ok = validate_config()
    
    print("\n4. Creating directories...")
    create_directories()
    
    print("\n=== Setup Summary ===")
    
    if deps_ok and config_ok:
        print("✅ Setup completed successfully!")
        print("\n🚀 Ready to run:")
        print("   python main.py --mode full")
        print("   python main.py --mode train")
        print("   python main.py --mode backtest")
    else:
        print("⚠️  Setup completed with issues:")
        if not deps_ok:
            print("   - Install missing dependencies: pip install -r requirements.txt")
        if not config_ok:
            print("   - Fix configuration issues in config.json")
    
    print("\n📚 Next steps:")
    print("   1. Add your API keys to config.json")
    print("   2. Adjust trading parameters in config.json")
    print("   3. Run: python main.py --help for usage options")
    print("   4. Start with small amounts and paper trading")

if __name__ == "__main__":
    main()