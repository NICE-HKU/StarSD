#!/usr/bin/env python3
"""
Simple Auto Client logger
Clears the corresponding log file on each start and uses a fixed filename.
Redirects all `print` output to the log file.
"""
import logging
import traceback
import sys
from pathlib import Path
from datetime import datetime


class SimpleAutoLogger:
    """Simple logger: uses a fixed filename, clears file on each start, and redirects print output."""
    
    def __init__(self, client_id):
        """
        Initialize the logger.

        Args:
            client_id: Client ID (e.g., 0, 1, 2)
        """
        self.client_id = client_id
        log_dir = Path("client_error_logs")
        log_dir.mkdir(exist_ok=True)
        
        # fixed log filename
        self.log_file = log_dir / f"auto_client_{client_id}.log"
        
        #  create logger
        self.logger = logging.getLogger(f"AutoClient{client_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # file handler - overwrite old file on each start
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # redirect stdout and stderr to the log file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file_handle = open(self.log_file, 'a', encoding='utf-8')
        
        # create a class that writes to both console and file
        class TeeOutput:
            def __init__(self, file_handle, original_stream):
                self.file = file_handle
                self.original = original_stream
                
            def write(self, message):
                self.file.write(message)
                self.file.flush()
                self.original.write(message)
                self.original.flush()
                
            def flush(self):
                self.file.flush()
                self.original.flush()
        
        # redirect print to both file and console
        sys.stdout = TeeOutput(self.log_file_handle, self.original_stdout)
        sys.stderr = TeeOutput(self.log_file_handle, self.original_stderr)
        
        self.logger.info("="*70)
        self.logger.info(f"Auto Base Client {client_id} Started")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)
        
        print(f"\n{'='*70}")
        print(f"📝 Auto Client {client_id} - All output redirected to: {self.log_file}")
        print(f"{'='*70}\n")
    
    def log_error(self, message, exception=None):
        """Log an error message (includes traceback if exception provided)."""
        self.logger.error(f"❌ {message}")
        if exception:
            self.logger.error(f"Exception type: {type(exception).__name__}")
            self.logger.error(f"Exception message: {str(exception)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
    
    def log_warning(self, message):
        """Log a warning message."""
        self.logger.warning(f"⚠️ {message}")
    
    def log_critical(self, message, exception=None):
        """Log a critical error (includes traceback if exception provided)."""
        self.logger.critical(f"💥 CRITICAL: {message}")
        if exception:
            self.logger.critical(f"Exception type: {type(exception).__name__}")
            self.logger.critical(f"Exception message: {str(exception)}")
            self.logger.critical("Full traceback:")
            self.logger.critical(traceback.format_exc())
    
    def log_info(self, message):
        """Log an informational message."""
        self.logger.info(message)
    
    def close(self):
        """Close the logger and restore stdout/stderr."""
        self.logger.info("="*70)
        self.logger.info(f"Client {self.client_id} session ended")
        self.logger.info("="*70)
        
        # restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # close file handle
        if hasattr(self, 'log_file_handle'):
            self.log_file_handle.close()
        
        # close logger handlers
        for handler in self.logger.handlers:
            handler.close()
        
        print(f"\n{'='*70}")
        print(f"📝 Client {self.client_id} log saved to: {self.log_file}")
        print(f"{'='*70}\n")
