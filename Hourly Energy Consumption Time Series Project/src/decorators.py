import functools
import logging
import time

#Configure logging to putput INFO-level messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

def log_execution(func):
    """Decorator to log function execution details."""
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        logging.info(f"Executing {func.__name__} with args:{args} kwargs:{kwargs}")
        result = func(*args,**kwargs)
        logging.info(f"Finished {func.__name__}")
        return result
    return wrapper

def time_execution(func):
    """Decorator to measure and log execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        result = func(*args,**kwargs)
        end_time = time.perf_counter()
        logging.info(f"{func.__name__} executed in {end_time-start_time:.4f} seconds")
        return result
    return wrapper