"""
Main experiments from the paper
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
import yaml

from .hierarchical_aggregator import HierarchicalAggregator
from .data_generator import FinancialDataGenerator, IoTDataGenerator, NetworkDataGenerator
from .metrics import Metrics


class Experiments:
    """Run all experiments from the paper"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        default_config = {
            'base_delta': 1.0,  # 1 second
            'max_timeframe': 3600.0,  # 1 hour
            'n_samples': 1000000,
            'test_queries': 1000,
            'random_seed': 42
        }
        
        if config_path:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def experiment_financial(self) -> Dict[str, Any]:
        """Financial data experiment (OHLCV)"""
        print("Running financial experiment...")
        
        # Setup
        np.random.seed(self.config['random_seed'])
        base_delta = self.config['base_delta']
        max_timeframe = self.config['max_timeframe']
        
        # Create aggregator
        aggregator = HierarchicalAggregator(base_delta, max_timeframe)
        
        # Generate data
        generator = FinancialDataGenerator(
            initial_price=100.0,
            volatility=0.02,
            trend=0.0001
        )
        
        # Ground truth calculation (exact)
        exact_values = []
        timestamps = []
        values = []
        
        # Process stream
        start_time = time.time()
        count = 0
        for timestamp, value in generator.generate_stream(self.config['n_samples']):
            aggregator.update(timestamp, value)
            exact_values.append(value)
            timestamps.append(timestamp)
            values.append(value)
            count += 1
            
            if count >= self.config['n_samples']:
                break
        
        processing_time = time.time() - start_time
        throughput = count / processing_time
        
        # Test queries at different timeframes
        timeframes = [1, 10, 60, 300, 600, 1800]  # 1s to 30min
        errors = {tf: [] for tf in timeframes}
        
        for tf in timeframes:
            for _ in range(self.config['test_queries']):
                # Random query time
                query_time = np.random.uniform(timestamps[0], timestamps[-1])
                
                # Get exact values for comparison
                mask = (timestamps >= query_time - tf) & (timestamps <= query_time)
                if mask.any():
                    exact_ohlcv = {
                        'open': values[mask.argmax()] if mask.any() else 0,
                        'high': np.max(values[mask]),
                        'low': np.min(values[mask]),
                        'close': values[mask.argmax()] if mask.any() else 0,
                        'volume': len(values[mask])
                    }
                    
                    # Get estimated values
                    estimated_ohlcv = aggregator.query(tf, query_time)
                    
                    # Compute error
                    error = Metrics.compute_error(exact_ohlcv, estimated_ohlcv)
                    for key in error:
                        errors[tf].append(error[key])
        
        # Calculate average errors
        avg_errors = {tf: np.mean(errors[tf]) for tf in timeframes if errors[tf]}
        
        result = {
            'throughput': throughput,
            'memory_blocks': aggregator.memory_usage,
            'avg_errors': avg_errors,
            'update_count': aggregator.update_count,
            'merge_count': aggregator.merge_count
        }
        
        self.results['financial'] = result
        return result
    
    def experiment_iot(self) -> Dict[str, Any]:
        """IoT sensor data experiment"""
        print("Running IoT experiment...")
        
        np.random.seed(self.config['random_seed'] + 1)
        
        # Setup for anomaly detection
        aggregator = HierarchicalAggregator(
            base_delta=0.1,  # 100ms resolution
            max_timeframe=3600.0  # 1 hour
        )
        
        generator = IoTDataGenerator('temperature')
        
        # Simulate anomalies
        anomalies = []
        detection_results = []
        
        count = 0
        anomaly_threshold = 3.0  # Standard deviations
        
        for timestamp, value in generator.generate_stream(self.config['n_samples'] // 10):
            # Inject occasional anomalies
            if count % 10000 == 0:
                value += 10.0  # Spike
                anomalies.append((timestamp, value))
            
            aggregator.update(timestamp, value)
            
            # Detect anomalies using multi-scale queries
            current_time = timestamp
            
            # Check short-term (1s) deviation
            short_stats = aggregator.query(1.0, current_time)
            short_mean = (short_stats['high'] + short_stats['low']) / 2
            
            # Check long-term (60s) baseline
            long_stats = aggregator.query(60.0, current_time)
            long_mean = (long_stats['high'] + long_stats['low']) / 2
            
            # Detect if short-term deviates significantly from long-term
            deviation = abs(short_mean - long_mean)
            is_anomaly = deviation > anomaly_threshold
            
            if is_anomaly:
                detection_results.append({
                    'timestamp': timestamp,
                    'value': value,
                    'detected': True,
                    'deviation': deviation
                })
            
            count += 1
        
        # Calculate detection metrics
        true_positives = len([a for a in anomalies 
                            if any(abs(a[0] - d['timestamp']) < 1.0 
                                  for d in detection_results)])
        false_positives = len([d for d in detection_results 
                             if not any(abs(a[0] - d['timestamp']) < 1.0 
                                       for a in anomalies)])
        
        detection_rate = true_positives / max(1, len(anomalies)) * 100
        false_positive_rate = false_positives / max(1, len(detection_results)) * 100
        
        result = {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'anomalies_detected': len(detection_results),
            'true_anomalies': len(anomalies),
            'true_positives': true_positives,
            'false_positives': false_positives
        }
        
        self.results['iot'] = result
        return result
    
    def experiment_scalability(self) -> Dict[str, Any]:
        """Scalability with hierarchy depth"""
        print("Running scalability experiment...")
        
        base_delta = 1.0
        depths = [8, 10, 12, 14, 16]
        
        results = []
        
        for L in depths:
            max_timeframe = base_delta * (2 ** (L - 1))
            
            aggregator = HierarchicalAggregator(base_delta, max_timeframe)
            generator = FinancialDataGenerator()
            
            # Measure update time
            update_times = []
            query_times = []
            
            count = 0
            for timestamp, value in generator.generate_stream(100000):
                start = time.perf_counter()
                aggregator.update(timestamp, value)
                end = time.perf_counter()
                update_times.append((end - start) * 1e6)  # microseconds
                
                # Occasionally measure query time
                if count % 1000 == 0:
                    q_start = time.perf_counter()
                    aggregator.query(60.0, timestamp)  # 1-minute query
                    q_end = time.perf_counter()
                    query_times.append((q_end - q_start) * 1e6)
                
                count += 1
            
            results.append({
                'depth': L,
                'max_timeframe': max_timeframe,
                'avg_update_us': np.mean(update_times),
                'avg_query_us': np.mean(query_times) if query_times else 0,
                'memory_blocks': aggregator.memory_usage,
                'throughput': 1e6 / np.mean(update_times) if update_times else 0
            })
        
        self.results['scalability'] = results
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run all experiments"""
        print("Starting all experiments...")
        
        results = {
            'financial': self.experiment_financial(),
            'iot': self.experiment_iot(),
            'scalability': self.experiment_scalability(),
            'config': self.config
        }
        
        # Save results
        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]):
        """Generate markdown summary of results"""
        summary = "# Experiment Results\n\n"
        
        # Financial results
        if 'financial' in results:
            fin = results['financial']
            summary += "## Financial Data (OHLCV)\n"
            summary += f"- Throughput: {fin['throughput']:,.0f} events/sec\n"
            summary += f"- Memory usage: {fin['memory_blocks']} blocks\n"
            summary += "- Average errors by timeframe:\n"
            for tf, err in fin['avg_errors'].items():
                summary += f"  - {tf}s: {err:.2f}%\n"
            summary += "\n"
        
        # IoT results
        if 'iot' in results:
            iot = results['iot']
            summary += "## IoT Anomaly Detection\n"
            summary += f"- Detection rate: {iot['detection_rate']:.1f}%\n"
            summary += f"- False positive rate: {iot['false_positive_rate']:.1f}%\n"
            summary += f"- True anomalies: {iot['true_anomalies']}\n"
            summary += f"- Detected anomalies: {iot['anomalies_detected']}\n"
            summary += "\n"
        
        # Scalability results
        if 'scalability' in results:
            summary += "## Scalability Analysis\n"
            summary += "| Depth | Max Timeframe | Update (µs) | Query (µs) | Memory (blocks) |\n"
            summary += "|-------|---------------|-------------|------------|-----------------|\n"
            for r in results['scalability']:
                summary += f"| {r['depth']} | {r['max_timeframe']:.0f}s | {r['avg_update_us']:.2f} | {r['avg_query_us']:.2f} | {r['memory_blocks']} |\n"
        
        with open('results_summary.md', 'w') as f:
            f.write(summary)
        
        print("Results saved to experiment_results.json and results_summary.md")


def run_demo():
    """Quick demo of the hierarchical aggregator"""
    print("=== Hierarchical Streaming Aggregator Demo ===\n")
    
    # Create aggregator
    agg = HierarchicalAggregator(base_delta=1.0, max_timeframe=300.0)  # 5 minutes max
    
    # Simulate some data
    generator = FinancialDataGenerator(initial_price=100.0)
    
    print("Processing 100,000 financial data points...")
    count = 0
    for timestamp, price in generator.generate_stream(100000):
        agg.update(timestamp, price)
        count += 1
        
        # Show progress
        if count % 20000 == 0:
            print(f"  Processed {count:,} events")
        
        if count >= 100000:
            break
    
    print(f"\nProcessed {count:,} events")
    print(f"Memory usage: {agg.memory_usage} blocks")
    
    # Demo queries
    current_time = timestamp
    print("\nSample queries:")
    
    timeframes = [1, 10, 60, 300]  # 1s, 10s, 1min, 5min
    for tf in timeframes:
        result = agg.query(tf, current_time)
        print(f"  Last {tf}s: O={result['open']:.2f}, H={result['high']:.2f}, "
              f"L={result['low']:.2f}, C={result['close']:.2f}")
    
    # Show stats
    stats = agg.get_stats()
    print(f"\nPerformance statistics:")
    print(f"  Update rate: {stats['update_rate']:,.0f} events/sec")
    print(f"  Merge operations: {stats['merge_count']}")
    print(f"  Hierarchy depth: {stats['levels']}")
    
    return agg


if __name__ == "__main__":
    # Run demo by default
    run_demo()