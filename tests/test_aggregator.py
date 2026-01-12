"""
Unit tests for the hierarchical aggregator
"""
import unittest
import numpy as np
import time
from src.hierarchical_aggregator import HierarchicalAggregator, Aggregate
from src.data_generator import FinancialDataGenerator


class TestAggregator(unittest.TestCase):
    
    def setUp(self):
        self.agg = HierarchicalAggregator(base_delta=1.0, max_timeframe=60.0)
        self.generator = FinancialDataGenerator(seed=42)
    
    def test_aggregate_operations(self):
        """Test basic aggregate operations"""
        a1 = Aggregate()
        a2 = Aggregate()
        
        # Update with values
        for val in [10, 20, 30]:
            a1.update(val)
        
        for val in [40, 50]:
            a2.update(val)
        
        # Test individual aggregates
        self.assertEqual(a1.count, 3)
        self.assertEqual(a1.sum, 60)
        self.assertEqual(a1.min, 10)
        self.assertEqual(a1.max, 30)
        self.assertEqual(a1.first, 10)
        self.assertEqual(a1.last, 30)
        
        # Test merge
        merged = a1.merge(a2)
        self.assertEqual(merged.count, 5)
        self.assertEqual(merged.sum, 150)
        self.assertEqual(merged.min, 10)
        self.assertEqual(merged.max, 50)
        self.assertEqual(merged.first, 10)
        self.assertEqual(merged.last, 50)
    
    def test_update_throughput(self):
        """Test that updates are O(1) amortized"""
        times = []
        
        # Measure update times
        for i in range(10000):
            timestamp = i * 0.01  # 100 Hz data
            value = self.generator._generate_value(timestamp)
            
            start = time.perf_counter()
            self.agg.update(timestamp, value)
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)  # microseconds
        
        # Average should be very small (sub-microsecond in optimized code)
        avg_time = np.mean(times)
        self.assertLess(avg_time, 100)  # Less than 100 microseconds per update
        
        print(f"Average update time: {avg_time:.2f} µs")
    
    def test_query_accuracy(self):
        """Test query accuracy against exact computation"""
        # Generate data
        values = []
        timestamps = []
        
        for i in range(1000):
            timestamp = i * 0.1  # 10 Hz
            value = self.generator._generate_value(timestamp)
            self.agg.update(timestamp, value)
            values.append(value)
            timestamps.append(timestamp)
        
        # Test queries at different timeframes
        test_timeframes = [1.0, 5.0, 10.0]
        current_time = timestamps[-1]
        
        for tf in test_timeframes:
            # Get exact values
            mask = [t >= current_time - tf for t in timestamps]
            exact_values = [v for v, m in zip(values, mask) if m]
            
            if exact_values:
                exact_stats = {
                    'open': exact_values[0],
                    'high': max(exact_values),
                    'low': min(exact_values),
                    'close': exact_values[-1],
                    'volume': len(exact_values)
                }
                
                # Get estimated values
                estimated_stats = self.agg.query(tf, current_time)
                
                # Check that error is reasonable
                for key in ['high', 'low']:
                    error = abs(exact_stats[key] - estimated_stats[key]) / abs(exact_stats[key])
                    self.assertLess(error, 0.05)  # Less than 5% error
    
    def test_memory_usage(self):
        """Test that memory usage is O(Δ_max/δ)"""
        # Process many events
        for i in range(100000):
            timestamp = i * 0.001  # 1 KHz data
            value = self.generator._generate_value(timestamp)
            self.agg.update(timestamp, value)
        
        # Memory should be bounded
        memory = self.agg.memory_usage
        max_expected = 4 * (self.agg.max_timeframe / self.agg.base_delta)  # Theoretical bound
        
        self.assertLess(memory, max_expected)
        print(f"Memory usage: {memory} blocks (bound: {max_expected:.0f})")
    
    def test_multiple_queries(self):
        """Test that multiple queries don't affect performance"""
        # Process data
        for i in range(10000):
            timestamp = i * 0.01
            value = self.generator._generate_value(timestamp)
            self.agg.update(timestamp, value)
        
        # Run many queries
        query_times = []
        current_time = 10000 * 0.01
        
        for _ in range(1000):
            tf = np.random.choice([1.0, 5.0, 10.0, 30.0])
            
            start = time.perf_counter()
            result = self.agg.query(tf, current_time)
            end = time.perf_counter()
            
            query_times.append((end - start) * 1e6)  # microseconds
            
            # Verify result has expected keys
            self.assertIn('open', result)
            self.assertIn('high', result)
            self.assertIn('low', result)
            self.assertIn('close', result)
        
        avg_query_time = np.mean(query_times)
        self.assertLess(avg_query_time, 1000)  # Less than 1 ms per query
        print(f"Average query time: {avg_query_time:.2f} µs")


if __name__ == '__main__':
    unittest.main(verbosity=2)