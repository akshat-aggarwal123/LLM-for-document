"""Logging configuration and utilities"""

import sys
import os
from loguru import logger
from app.core.config import settings


def setup_logging():
    """Configure logging for the application"""

    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(settings.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # File handler
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )

    return logger


def log_function_call(func_name: str, **kwargs):
    """Log function calls with parameters"""
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_performance(func_name: str, duration: float):
    """Log performance metrics"""
    logger.info(f"Function {func_name} completed in {duration:.3f}s")


def log_error(error: Exception, context: str = ""):
    """Log errors with context"""
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
def get_logger(name: str = None):
    """Returns the pre-configured Loguru logger."""
    # The logger is imported from loguru and configured by setup_logging()
    return logger

# Initialize logger
app_logger = setup_logging()
