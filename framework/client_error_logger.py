#!/usr/bin/env python3
"""
Client Error Logger - records client error and warning messages

Features:
- Create a separate error log file for each client
- Only record errors (❌) and warnings (⚠️)
- Log files are stored under the fixed directory client_error_logs/
- Automatically create directories and files as needed
- Thread-safe
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path


class ClientErrorLogger:
    """Client error logger"""
    
    LOG_DIR = Path(__file__).parent / "client_error_logs"
    
    def __init__(self, client_identifier=None):
        """
        Initialize the error logger

        Args:
            client_identifier: Client identifier (used to generate the log filename).
                               If None, the process ID will be used.
        """
        # Create the log directory
        self.LOG_DIR.mkdir(exist_ok=True)
        
        if client_identifier is None:
            client_identifier = f"pid_{os.getpid()}"
        
        # Use the client identifier as the filename (ensures the same file is used across runs)
        self.log_file = self.LOG_DIR / f"{client_identifier}_errors.log"
        
        # Thread lock to ensure thread safety
        self._lock = threading.Lock()
        
        # Write log header
        self._write_log_header()
        
        print(f"📝 Error log file: {self.log_file}")
    
    def _write_log_header(self):
        """Write log header information"""
        with self._lock:
            mode = 'a' if self.log_file.exists() else 'w'
            with open(self.log_file, mode, encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Client started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Process ID: {os.getpid()}\n")
                f.write(f"{'='*80}\n\n")
    
    def log_error(self, message, exception=None):
        """
        Record an error message

        Args:
            message: Error message
            exception: Exception object (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] ❌ ERROR: {message}\n")
                
                if exception:
                    import traceback
                    f.write(f"Exception type: {type(exception).__name__}\n")
                    f.write(f"Exception message: {str(exception)}\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                
                f.flush()
    
    def log_warning(self, message):
        """
        Record a warning message

        Args:
            message: Warning message
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] ⚠️  WARNING: {message}\n")
                f.flush()
    
    def log_critical(self, message, exception=None):
        """
        Record a critical error

        Args:
            message: Error message
            exception: Exception object (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'!'*80}\n")
                f.write(f"[{timestamp}] 🔥 CRITICAL ERROR: {message}\n")
                
                if exception:
                    import traceback
                    f.write(f"Exception type: {type(exception).__name__}\n")
                    f.write(f"Exception message: {str(exception)}\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                
                f.write(f"{'!'*80}\n\n")
                f.flush()
    
    def log_timeout(self, operation, timeout_value):
        """
        Record a timeout event

        Args:
            operation: Description of the timed-out operation
            timeout_value: Timeout value (seconds)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] ⏰ TIMEOUT: {operation} (timeout: {timeout_value}s)\n")
                f.flush()
    
    def close(self):
        """Close the logger"""
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Client ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")


_global_logger = None

def get_error_logger(client_identifier=None):
    """
    Get a global error logger instance

    Args:
        client_identifier: Client identifier

    Returns:
        A ClientErrorLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ClientErrorLogger(client_identifier)
    return _global_logger


if __name__ == "__main__":
    # Test code
    logger = ClientErrorLogger("test_client")
    
    logger.log_warning("This is a test warning")
    logger.log_error("This is a test error")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.log_error("Caught an exception", e)
    
    logger.log_timeout("waiting for server response", 30)
    logger.log_critical("Critical failure detected")
    logger.close()

    print(f"✅ Test completed. Check log file: {logger.log_file}")
