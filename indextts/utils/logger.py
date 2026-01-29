import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

class CentralLogger:
    """Centralized logging utility for IndexTTS2.
    
    Provides both console output (via Rich) and file output (via RotatingFileHandler).
    """
    
    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CentralLogger, cls).__new__(cls)
        return cls._instance

    def setup(
        self,
        name: str = "indextts",
        log_file: str = "app.log",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        rich_console: Optional[Console] = None
    ) -> logging.Logger:
        """Initialize and configure the logger.
        
        Args:
            name: Logger name.
            log_file: Path to the log file.
            level: Logging level.
            max_bytes: Max size of log file before rotation.
            backup_count: Number of backup log files to keep.
            rich_console: Optional Rich Console instance.
            
        Returns:
            Configured logger instance.
        """
        if self._logger:
            return self._logger

        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()

        # 1. File Handler (Rotating)
        try:
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

        # 2. Console Handler (Rich)
        console = rich_console or Console()
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False
        )
        # Use a simpler format for console to keep it clean
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)

        # 3. Suppress external loggers
        self._suppress_external_loggers()

        self._logger = logger
        return logger

    def _suppress_external_loggers(self):
        """Suppress noisy logs from 3rd party libraries."""
        noisy_loggers = [
            "urllib3",
            "transformers",
            "huggingface_hub",
            "matplotlib",
            "PIL",
            "asyncio"
        ]
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    @property
    def logger(self) -> logging.Logger:
        if not self._logger:
            # Fallback to a default setup if not initialized
            return self.setup()
        return self._logger

# Global instance for easy access
logger_manager = CentralLogger()

def get_logger() -> logging.Logger:
    return logger_manager.logger
