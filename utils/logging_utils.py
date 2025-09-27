"""
Logging utilities for the options trading system.
"""
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
from .config import config


class DebugLogger:
    """Custom logger for debug information with separate log files."""
    
    def __init__(self, name: str, log_dir: str = None):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path(config.get('logging.log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up file and console handlers."""
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.get('logging.max_file_size', 10485760),
            backupCount=config.get('logging.backup_count', 5)
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set levels
        file_handler.setLevel(logging.DEBUG)
        console_level = config.get('logging.level', 'INFO')
        console_handler.setLevel(getattr(logging, console_level.upper()))
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


def get_logger(name: str) -> DebugLogger:
    """Get a logger instance for the given name."""
    return DebugLogger(name)


# Global loggers for different components
tick_logger = get_logger('tick_data')
candle_logger = get_logger('candle_builder')
strategy_logger = get_logger('strategy')
backtest_logger = get_logger('backtest')
trade_logger = get_logger('trading')
error_logger = get_logger('errors')
