#!/usr/bin/env python3
"""
Memory management utilities for the QR ID printing application.
Provides tools for monitoring and optimizing memory usage.
"""

import gc
import os
import sys
import psutil
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management utilities"""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information"""
        try:
            vm = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'system': {
                    'total_gb': round(vm.total / (1024**3), 2),
                    'available_gb': round(vm.available / (1024**3), 2),
                    'used_percent': vm.percent,
                    'available_mb': round(vm.available / (1024**2), 1)
                },
                'process': {
                    'rss_mb': round(process.memory_info().rss / (1024**2), 1),
                    'vms_mb': round(process.memory_info().vms / (1024**2), 1),
                    'percent': round(process.memory_percent(), 2)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
    
    def check_memory_status(self) -> str:
        """Check current memory status"""
        try:
            vm = psutil.virtual_memory()
            if vm.percent >= self.critical_threshold:
                return "critical"
            elif vm.percent >= self.warning_threshold:
                return "warning"
            else:
                return "normal"
        except Exception:
            return "unknown"
    
    def force_cleanup(self) -> bool:
        """Force garbage collection and memory cleanup"""
        try:
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            # Clear any cached data
            if hasattr(gc, 'set_threshold'):
                gc.set_threshold(700, 10, 10)  # More aggressive GC
                
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False

def memory_limit_check(min_mb: int = 50):
    """Decorator to check memory before function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                vm = psutil.virtual_memory()
                available_mb = vm.available / (1024**2)
                
                if available_mb < min_mb:
                    logger.warning(f"Insufficient memory for {func.__name__}: {available_mb:.1f}MB < {min_mb}MB")
                    gc.collect()  # Try cleanup
                    
                    # Check again after cleanup
                    vm = psutil.virtual_memory()
                    available_mb = vm.available / (1024**2)
                    
                    if available_mb < min_mb:
                        raise MemoryError(f"Insufficient memory: {available_mb:.1f}MB available, {min_mb}MB required")
                
                return func(*args, **kwargs)
                
            except MemoryError:
                raise
            except Exception as e:
                logger.error(f"Memory check failed for {func.__name__}: {e}")
                return func(*args, **kwargs)  # Continue anyway
                
        return wrapper
    return decorator

def optimize_pandas_memory():
    """Optimize pandas for low memory usage"""
    try:
        import pandas as pd
        
        # Configure pandas for memory efficiency
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('compute.use_bottleneck', False)
        pd.set_option('compute.use_numexpr', False)
        pd.set_option('mode.copy_on_write', True)
        
        logger.info("Pandas memory optimization applied")
        return True
        
    except ImportError:
        logger.warning("Pandas not available for optimization")
        return False
    except Exception as e:
        logger.error(f"Pandas optimization failed: {e}")
        return False

def set_memory_limits():
    """Set system memory limits for the application"""
    try:
        if sys.platform != "win32":
            import resource
            
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Set conservative memory limit (180MB for starter plan)
            memory_limit = int(os.getenv('MEMORY_LIMIT_MB', '180')) * 1024 * 1024
            
            if soft == resource.RLIM_INFINITY or soft > memory_limit:
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
                logger.info(f"Memory limit set to {memory_limit // (1024*1024)}MB")
                return True
            else:
                logger.info(f"Memory limit already set to {soft // (1024*1024)}MB")
                return True
                
    except ImportError:
        logger.info("Resource module not available (Windows)")
        return False
    except Exception as e:
        logger.error(f"Failed to set memory limits: {e}")
        return False

def emergency_cleanup():
    """Emergency memory cleanup when system is under pressure"""
    try:
        logger.warning("Performing emergency memory cleanup")
        
        # Force multiple garbage collection cycles
        for i in range(5):
            collected = gc.collect()
            logger.info(f"GC cycle {i+1}: collected {collected} objects")
        
        # Try to free up system memory
        if hasattr(gc, 'set_debug'):
            gc.set_debug(0)  # Disable debug mode to save memory
            
        # Log memory status after cleanup
        vm = psutil.virtual_memory()
        logger.info(f"Memory after cleanup: {vm.percent:.1f}% used, {vm.available/(1024**2):.1f}MB available")
        
        return vm.percent < 90  # Return True if cleanup was successful
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")
        return False

if __name__ == "__main__":
    # Test memory utilities
    monitor = MemoryMonitor()
    print("Memory Info:", monitor.get_memory_info())
    print("Memory Status:", monitor.check_memory_status())
    
    # Test optimization functions
    optimize_pandas_memory()
    set_memory_limits()
