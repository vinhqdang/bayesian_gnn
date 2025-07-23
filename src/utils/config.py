import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    openai: Optional[str] = None
    gemini: Optional[str] = None

@dataclass 
class TradingConfig:
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    confidence_threshold: float = 0.6

@dataclass
class ModelConfig:
    hidden_dims: list = None
    dropout: float = 0.1
    prior_std: float = 0.1
    use_uncertainty: bool = True
    gnn_type: str = "GCN"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32, 16]

@dataclass
class DataConfig:
    lookback_window: int = 30
    correlation_threshold: float = 0.3
    top_k_edges: int = 5

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config_data = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            print(f"Config file {self.config_path} not found.")
            print("Please copy config.json.template to config.json and add your API keys.")
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            print(f"Configuration loaded from {self.config_path}")
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            self._create_default_config()
        except Exception as e:
            print(f"Error loading config file: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config_data = {
            "api_keys": {
                "openai": None,
                "gemini": None
            },
            "trading_config": {
                "initial_capital": 100000,
                "transaction_cost": 0.001,
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
                "confidence_threshold": 0.6
            },
            "model_config": {
                "hidden_dims": [64, 32, 16],
                "dropout": 0.1,
                "prior_std": 0.1,
                "use_uncertainty": True,
                "gnn_type": "GCN"
            },
            "data_config": {
                "lookback_window": 30,
                "correlation_threshold": 0.3,
                "top_k_edges": 5
            }
        }
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        api_data = self.config_data.get("api_keys", {})
        return APIConfig(
            openai=api_data.get("openai"),
            gemini=api_data.get("gemini")
        )
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        trading_data = self.config_data.get("trading_config", {})
        return TradingConfig(
            initial_capital=trading_data.get("initial_capital", 100000.0),
            transaction_cost=trading_data.get("transaction_cost", 0.001),
            max_position_size=trading_data.get("max_position_size", 0.1),
            risk_per_trade=trading_data.get("risk_per_trade", 0.02),
            confidence_threshold=trading_data.get("confidence_threshold", 0.6)
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        model_data = self.config_data.get("model_config", {})
        return ModelConfig(
            hidden_dims=model_data.get("hidden_dims", [64, 32, 16]),
            dropout=model_data.get("dropout", 0.1),
            prior_std=model_data.get("prior_std", 0.1),
            use_uncertainty=model_data.get("use_uncertainty", True),
            gnn_type=model_data.get("gnn_type", "GCN")
        )
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        data_config = self.config_data.get("data_config", {})
        return DataConfig(
            lookback_window=data_config.get("lookback_window", 30),
            correlation_threshold=data_config.get("correlation_threshold", 0.3),
            top_k_edges=data_config.get("top_k_edges", 5)
        )
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def update_api_key(self, provider: str, key: str):
        """Update API key for a provider"""
        if "api_keys" not in self.config_data:
            self.config_data["api_keys"] = {}
        
        self.config_data["api_keys"][provider] = key
        print(f"Updated {provider} API key")
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation = {
            "config_file_exists": os.path.exists(self.config_path),
            "openai_key_present": bool(self.get_api_config().openai),
            "gemini_key_present": bool(self.get_api_config().gemini),
            "trading_config_valid": True,
            "model_config_valid": True
        }
        
        # Validate trading config
        trading_config = self.get_trading_config()
        if trading_config.initial_capital <= 0:
            validation["trading_config_valid"] = False
        if not (0 < trading_config.max_position_size <= 1):
            validation["trading_config_valid"] = False
        
        # Validate model config
        model_config = self.get_model_config()
        if not model_config.hidden_dims or not all(isinstance(x, int) and x > 0 for x in model_config.hidden_dims):
            validation["model_config_valid"] = False
        
        return validation
    
    def print_status(self):
        """Print configuration status"""
        validation = self.validate_config()
        
        print("\n=== Configuration Status ===")
        print(f"Config file exists: {'✓' if validation['config_file_exists'] else '✗'}")
        print(f"OpenAI API key: {'✓' if validation['openai_key_present'] else '✗'}")
        print(f"Gemini API key: {'✓' if validation['gemini_key_present'] else '✗'}")
        print(f"Trading config valid: {'✓' if validation['trading_config_valid'] else '✗'}")
        print(f"Model config valid: {'✓' if validation['model_config_valid'] else '✗'}")
        
        if not any(validation.values()):
            print("\n⚠️  Please check your configuration!")
        elif validation['openai_key_present'] or validation['gemini_key_present']:
            print("\n✓ Configuration looks good!")
        else:
            print("\n⚠️  No API keys configured. AI features will be disabled.")

# Global config instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config_manager