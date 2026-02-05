"""
Terminal logging utilities with progress bars and buffer clearing.

Provides enhanced terminal output with progress tracking, buffer clearing,
and formatted logging for long-running QAOA optimization tasks.
"""
import sys
import os
from typing import Optional, Union
from datetime import datetime
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class TerminalLogger:
    """
    Terminal logger with progress bars and buffer clearing.
    """
    
    def __init__(
        self,
        use_progress_bars: bool = True,
        clear_on_update: bool = True,
        flush_immediately: bool = True
    ):
        """
        Initialize terminal logger.
        
        Args:
            use_progress_bars: Whether to use tqdm progress bars
            clear_on_update: Whether to clear terminal on major updates
            flush_immediately: Whether to flush output immediately
        """
        self.use_progress_bars = use_progress_bars and TQDM_AVAILABLE
        self.clear_on_update = clear_on_update
        self.flush_immediately = flush_immediately
        self.start_time = time.time()
        self.last_update_time = time.time()
    
    def clear_terminal(self):
        """Clear terminal screen."""
        if self.clear_on_update:
            os.system('cls' if os.name == 'nt' else 'clear')
    
    def flush(self):
        """Flush output buffer."""
        if self.flush_immediately:
            sys.stdout.flush()
            sys.stderr.flush()
    
    def log(self, message: str, level: str = "INFO", clear: bool = False):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
            clear: Whether to clear terminal before logging
        """
        if clear:
            self.clear_terminal()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color codes for different levels
        colors = {
            "INFO": "\033[0m",      # Reset
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",    # Red
            "SUCCESS": "\033[92m",   # Green
            "PROGRESS": "\033[94m"  # Blue
        }
        reset = "\033[0m"
        
        color = colors.get(level, colors["INFO"])
        formatted_message = f"{color}[{timestamp}] [{level}] {message}{reset}"
        
        print(formatted_message)
        self.flush()
    
    def info(self, message: str, clear: bool = False):
        """Log info message."""
        self.log(message, "INFO", clear)
    
    def warning(self, message: str, clear: bool = False):
        """Log warning message."""
        self.log(message, "WARNING", clear)
    
    def error(self, message: str, clear: bool = False):
        """Log error message."""
        self.log(message, "ERROR", clear)
    
    def success(self, message: str, clear: bool = False):
        """Log success message."""
        self.log(message, "SUCCESS", clear)
    
    def progress(self, message: str, clear: bool = False):
        """Log progress message."""
        self.log(message, "PROGRESS", clear)
    
    def section(self, title: str, clear: bool = True):
        """
        Print a section header.
        
        Args:
            title: Section title
            clear: Whether to clear terminal before printing
        """
        if clear:
            self.clear_terminal()
        
        width = 80
        print("=" * width)
        print(f" {title}".ljust(width - 1) + "=")
        print("=" * width)
        self.flush()
    
    def create_progress_bar(
        self,
        total: int,
        desc: str = "",
        unit: str = "it",
        leave: bool = True
    ):
        """
        Create a progress bar.
        
        Args:
            total: Total number of iterations
            desc: Description
            unit: Unit name
            leave: Whether to leave progress bar after completion
            
        Returns:
            Progress bar object or None if not available
        """
        if not self.use_progress_bars:
            return None
        
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update_progress_bar(self, pbar, n: int = 1, desc: Optional[str] = None):
        """
        Update progress bar.
        
        Args:
            pbar: Progress bar object
            n: Number of steps to advance
            desc: Optional description update
        """
        if pbar is not None:
            if desc is not None:
                pbar.set_description(desc)
            pbar.update(n)
            self.flush()
    
    def close_progress_bar(self, pbar):
        """Close progress bar."""
        if pbar is not None:
            pbar.close()
    
    def elapsed_time(self) -> float:
        """Get elapsed time since initialization."""
        return time.time() - self.start_time
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def summary(self, stats: dict):
        """
        Print summary statistics.
        
        Args:
            stats: Dictionary with statistics
        """
        self.section("Summary", clear=False)
        
        for key, value in stats.items():
            if isinstance(value, float):
                if value < 1.0:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:,.2f}")
            elif isinstance(value, int):
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
        
        elapsed = self.elapsed_time()
        print(f"\n  Total elapsed time: {self.format_time(elapsed)}")
        self.flush()


# Global logger instance
_global_logger: Optional[TerminalLogger] = None


def get_logger(
    use_progress_bars: bool = True,
    clear_on_update: bool = True,
    flush_immediately: bool = True
) -> TerminalLogger:
    """
    Get or create global logger instance.
    
    Args:
        use_progress_bars: Whether to use progress bars
        clear_on_update: Whether to clear on update
        flush_immediately: Whether to flush immediately
        
    Returns:
        TerminalLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = TerminalLogger(
            use_progress_bars=use_progress_bars,
            clear_on_update=clear_on_update,
            flush_immediately=flush_immediately
        )
    
    return _global_logger


def reset_logger():
    """Reset global logger instance."""
    global _global_logger
    _global_logger = None
