"""
Configuration management for the options trading system.
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for the trading system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'tick_data_path': 'data/tick_data',
                'candles_path': 'data/candles',
                'instruments_path': 'data/instruments'
            },
            'candle_builder': {
                'timeframes': ['1s', '1m', '5m', '15m', '30m', '1h', '1d'],
                'default_timeframe': '1m'
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'max_file_size': 10485760,  # 10MB
                'backup_count': 5
            },
            'trading': {
                'max_positions': 10,
                'max_daily_loss': 10000,
                'default_lot_size': 1
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self):
        """Save current configuration to file."""
        os.makedirs(self.config_path.parent, exist_ok=True)
        with open(self.config_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False)


# Global config instance
config = Config()
