"""
Enhanced debug logging system for PyATRAC1 signal processing analysis.
Provides comprehensive metadata and source tracking for cross-comparison with atracdenc.
"""

import time
import inspect
import numpy as np
from typing import List, Union, Optional, Any
import os


class AtracDebugLogger:
    """
    Comprehensive debug logger for ATRAC1 signal processing stages.
    Logs with full metadata including source location, data statistics, and context.
    """
    
    def __init__(self, log_file: str = "pytrac_debug.log", enabled: bool = True):
        self.log_file = log_file
        self.enabled = enabled
        if enabled:
            # Clear log file and write header
            with open(log_file, 'w') as f:
                f.write(f"# PyATRAC1 Debug Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Format: [TIMESTAMP][IMPL][FILE:LINE][FUNC][CH{n}][FR{nnn}][BAND_{name}] STAGE: data_type=values |META: ... |SRC: ...\n")
                f.write("#\n")
    
    def log_stage(self, stage: str, data_type: str, values: Union[List, np.ndarray, float, int], 
                  channel: int = 0, frame: int = 0, band: str = "", 
                  **context) -> None:
        """
        Log a processing stage with comprehensive metadata.
        
        Args:
            stage: Processing stage name (e.g., 'QMF_OUTPUT', 'MDCT_INPUT')
            data_type: Type of data being logged (e.g., 'samples', 'coeffs', 'bits')
            values: The actual data values
            channel: Channel index (0, 1)
            frame: Frame index 
            band: Band name ('LOW', 'MID', 'HIGH', '')
            **context: Additional context (algorithm, window_type, bfu_idx, etc.)
        """
        if not self.enabled:
            return
            
        # Auto-detect source location
        frame_info = inspect.currentframe().f_back
        filename = os.path.basename(frame_info.f_code.co_filename)
        line_no = frame_info.f_lineno
        func_name = frame_info.f_code.co_name
        
        # Handle scalar values
        if isinstance(values, (int, float)):
            values_array = np.array([values], dtype=np.float32)
            is_scalar = True
        else:
            # Convert to numpy array for consistent handling
            if isinstance(values, (list, tuple)):
                values_array = np.array(values, dtype=np.float32)
            else:
                values_array = values
            is_scalar = False
            
        # Calculate metadata
        size = len(values_array) if hasattr(values_array, '__len__') else 1
        
        if size > 0:
            min_val = float(np.min(values_array))
            max_val = float(np.max(values_array))
            sum_val = float(np.sum(values_array))
            mean_val = float(np.mean(values_array))
            nonzero_count = int(np.count_nonzero(values_array))
        else:
            min_val = max_val = sum_val = mean_val = 0.0
            nonzero_count = 0
            
        # Format values (truncate if too long for readability)
        if is_scalar:
            values_str = f"{values:.6f}"
        elif size <= 10:
            values_str = f"[{','.join(f'{v:.6f}' for v in values_array)}]"
        else:
            # Show first 5 and last 5 values
            first_5 = ','.join(f'{v:.6f}' for v in values_array[:5])
            last_5 = ','.join(f'{v:.6f}' for v in values_array[-5:])
            values_str = f"[{first_5}...{last_5}]"
        
        # Build context string
        context_parts = []
        for key, value in context.items():
            context_parts.append(f"{key}={value}")
        context_str = " ".join(context_parts)
        
        # Generate timestamp with microsecond precision
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.') + f"{int(time.time() * 1000000) % 1000000:06d}"
        
        # Format band string
        band_str = f"[BAND_{band}]" if band else ""
        
        # Build complete log entry
        log_entry = (
            f"[{timestamp}][PYTRAC][{filename}:{line_no}][{func_name}]"
            f"[CH{channel}][FR{frame:03d}]{band_str} {stage}: "
            f"{data_type}={values_str} "
            f"|META: size={size} range=[{min_val:.6f},{max_val:.6f}] "
            f"sum={sum_val:.6f} mean={mean_val:.6f} nonzero={nonzero_count} "
            f"|SRC: {context_str}\n"
        )
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_array_detailed(self, stage: str, data_type: str, values: Union[List, np.ndarray],
                          channel: int = 0, frame: int = 0, band: str = "",
                          max_elements: int = 50, **context) -> None:
        """
        Log array data with more detailed output (useful for small critical arrays).
        
        Args:
            max_elements: Maximum number of elements to log in full detail
        """
        if not self.enabled:
            return
            
        # Convert to numpy for consistent handling
        if isinstance(values, (list, tuple)):
            values_array = np.array(values, dtype=np.float32)
        else:
            values_array = values
            
        if len(values_array) <= max_elements:
            # Log all elements with full precision
            values_str = f"[{','.join(f'{v:.8f}' for v in values_array)}]"
        else:
            # Use standard truncation
            values_str = f"[{','.join(f'{v:.8f}' for v in values_array[:max_elements//2])}...truncated...{','.join(f'{v:.8f}' for v in values_array[-max_elements//2:])}]"
        
        # Use regular logging but with detailed values
        self.log_stage(f"{stage}_DETAILED", data_type, values_array, channel, frame, band, **context)
    
    def log_bitstream(self, stage: str, bitstream_bytes: bytes, 
                     channel: int = 0, frame: int = 0, **context) -> None:
        """
        Special logging for bitstream data in hex format.
        """
        if not self.enabled:
            return
            
        hex_str = bitstream_bytes.hex()
        size = len(bitstream_bytes)
        
        # Auto-detect source location
        frame_info = inspect.currentframe().f_back
        filename = os.path.basename(frame_info.f_code.co_filename)
        line_no = frame_info.f_lineno
        func_name = frame_info.f_code.co_name
        
        # Build context string
        context_parts = []
        for key, value in context.items():
            context_parts.append(f"{key}={value}")
        context_str = " ".join(context_parts)
        
        # Generate timestamp
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.') + f"{int(time.time() * 1000000) % 1000000:06d}"
        
        # Format log entry
        log_entry = (
            f"[{timestamp}][PYTRAC][{filename}:{line_no}][{func_name}]"
            f"[CH{channel}][FR{frame:03d}] {stage}: "
            f"hex={hex_str} "
            f"|META: size={size} bytes "
            f"|SRC: {context_str}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def enable(self):
        """Enable logging."""
        self.enabled = True
        
    def disable(self):
        """Disable logging."""
        self.enabled = False


# Global logger instance
debug_logger = AtracDebugLogger()


def log_debug(stage: str, data_type: str, values: Any, **kwargs) -> None:
    """
    Convenience function for logging with global logger instance.
    
    Usage:
        log_debug("QMF_OUTPUT", "samples", qmf_samples, 
                  channel=0, frame=1, band="LOW", algorithm="qmf_analysis")
    """
    debug_logger.log_stage(stage, data_type, values, **kwargs)


def log_debug_detailed(stage: str, data_type: str, values: Any, **kwargs) -> None:
    """
    Convenience function for detailed array logging.
    """
    debug_logger.log_array_detailed(stage, data_type, values, **kwargs)


def log_bitstream(stage: str, bitstream_bytes: bytes, **kwargs) -> None:
    """
    Convenience function for bitstream logging.
    """
    debug_logger.log_bitstream(stage, bitstream_bytes, **kwargs)


def enable_debug_logging(log_file: str = "pytrac_debug.log") -> None:
    """
    Enable debug logging with specified log file.
    """
    global debug_logger
    debug_logger = AtracDebugLogger(log_file, enabled=True)


def disable_debug_logging() -> None:
    """
    Disable debug logging.
    """
    debug_logger.disable()