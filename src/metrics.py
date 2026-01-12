"""
Metrics and evaluation functions
"""
import numpy as np
from typing import Dict, List, Tuple
import time


class Metrics:
    """Collection of evaluation metrics"""
    
    @staticmethod
    def compute_error(true_values: Dict[str, float], 
                     estimated_values: Dict[str, float]) -> Dict[str, float]:
        """Compute relative error for each metric"""
        errors = {}
        for key in true_values:
            if true_values[key] != 0:
                rel_error = abs(true_values[key] - estimated_values[key]) / abs(true_values[key])
                errors[key] = rel_error * 100  # Percentage
            else:
                errors[key] = abs(true_values[key] - estimated_values[key])
        return errors
    
    @staticmethod
    def mape(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def throughput_test(aggregator, generator, n_samples: int = 100000) -> float:
        """Measure throughput in events per second"""
        start_time = time.time()
        
        count = 0
        for timestamp, value in generator.generate_stream(n_samples):
            aggregator.update(timestamp, value)
            count += 1
            if count >= n_samples:
                break
        
        elapsed = time.time() - start_time
        return count / elapsed
    
    @staticmethod
    def latency_test(aggregator, n_queries: int = 1000) -> Tuple[float, float]:
        """Measure query latency"""
        timeframes = [1, 10, 60, 300, 600, 1800]  # 1s to 30min
        latencies = []
        
        for _ in range(n_queries):
            timeframe = np.random.choice(timeframes)
            current_time = time.time()
            
            start = time.perf_counter()
            aggregator.query(timeframe, current_time)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1e6)  # Convert to microseconds
        
        return np.mean(latencies), np.std(latencies)
    
    @staticmethod
    def memory_efficiency(aggregator, n_updates: int) -> float:
        """Compute memory efficiency (bytes per event)"""
        stats = aggregator.get_stats()
        memory_blocks = stats['memory_blocks']
        
        # Estimate bytes per block (aggregate + metadata)
        bytes_per_block = 100  # Approximate
        total_bytes = memory_blocks * bytes_per_block
        
        return total_bytes / n_updates  # bytes per event