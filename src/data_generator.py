"""
Data generators for financial, IoT, and network data streams
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple, Dict, Any
import time


class DataGenerator:
    """Base class for data generators"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.time_offset = 0.0
    
    def generate_stream(self, n_samples: int = 1000000, 
                       rate: float = 1000.0) -> Generator[Tuple[float, float], None, None]:
        """
        Generate stream of (timestamp, value) pairs
        
        Args:
            n_samples: Number of samples to generate
            rate: Events per second
        """
        for i in range(n_samples):
            timestamp = self.time_offset + i / rate
            value = self._generate_value(timestamp)
            yield timestamp, value
    
    def _generate_value(self, timestamp: float) -> float:
        raise NotImplementedError


class FinancialDataGenerator(DataGenerator):
    """Generate realistic financial time series (prices)"""
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.02, 
                 trend: float = 0.0001, seed: int = 42):
        super().__init__(seed)
        self.price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.spread = 0.01
        
    def _generate_value(self, timestamp: float) -> float:
        # Geometric Brownian motion with drift
        dt = 1.0 / 252 / 6.5 / 3600  # Assume 1-second intervals
        dw = self.rng.normal(0, np.sqrt(dt))
        
        # Update price
        self.price *= np.exp((self.trend - 0.5 * self.volatility**2) * dt + 
                           self.volatility * dw)
        
        # Add micro-structure noise
        noise = self.rng.normal(0, self.spread * np.sqrt(dt))
        return self.price + noise


class IoTDataGenerator(DataGenerator):
    """Generate IoT sensor data (temperature, pressure, etc.)"""
    
    def __init__(self, sensor_type: str = 'temperature', seed: int = 42):
        super().__init__(seed)
        self.sensor_type = sensor_type
        self.cycle_period = 3600.0  # 1-hour cycle
        self.base_values = {
            'temperature': 20.0,  # °C
            'pressure': 1013.25,  # hPa
            'humidity': 50.0,     # %
            'vibration': 0.0      # g
        }
        self.noise_levels = {
            'temperature': 0.1,
            'pressure': 0.5,
            'humidity': 1.0,
            'vibration': 0.01
        }
    
    def _generate_value(self, timestamp: float) -> float:
        base = self.base_values[self.sensor_type]
        noise_level = self.noise_levels[self.sensor_type]
        
        # Add diurnal cycle
        cycle = np.sin(2 * np.pi * timestamp / self.cycle_period)
        
        # Add random walk
        if not hasattr(self, 'walk'):
            self.walk = base
        self.walk += self.rng.normal(0, noise_level * 0.1)
        
        # Add noise
        noise = self.rng.normal(0, noise_level)
        
        return self.walk + cycle + noise


class NetworkDataGenerator(DataGenerator):
    """Generate network traffic data"""
    
    def __init__(self, base_rate: float = 1000.0, burst_prob: float = 0.01, 
                 seed: int = 42):
        super().__init__(seed)
        self.base_rate = base_rate
        self.burst_prob = burst_prob
        self.burst_active = False
        self.burst_end = 0.0
        
    def _generate_value(self, timestamp: float) -> float:
        # Check for burst events
        if not self.burst_active and self.rng.random() < self.burst_prob:
            self.burst_active = True
            self.burst_end = timestamp + self.rng.exponential(0.1)  # 100ms bursts
        
        if self.burst_active and timestamp > self.burst_end:
            self.burst_active = False
        
        # Generate traffic volume
        if self.burst_active:
            # Burst traffic: 10-100x normal rate
            rate_multiplier = 10 + 90 * self.rng.random()
        else:
            # Normal traffic with some variation
            rate_multiplier = 0.5 + self.rng.random()
        
        # Add Poisson-like variation
        volume = self.base_rate * rate_multiplier
        volume *= 1 + 0.1 * self.rng.normal()
        
        return max(0, volume)


class MixedDataGenerator:
    """Generate mixed data streams for comprehensive testing"""
    
    @staticmethod
    def create_dataset(n_samples: int = 100000, 
                      data_types: list = None) -> pd.DataFrame:
        """
        Create mixed dataset with multiple data types
        
        Returns:
            DataFrame with columns: timestamp, value, data_type
        """
        if data_types is None:
            data_types = ['financial', 'iot_temp', 'iot_pressure', 'network']
        
        generators = []
        for dtype in data_types:
            if dtype == 'financial':
                gen = FinancialDataGenerator()
            elif dtype == 'iot_temp':
                gen = IoTDataGenerator('temperature')
            elif dtype == 'iot_pressure':
                gen = IoTDataGenerator('pressure')
            elif dtype == 'network':
                gen = NetworkDataGenerator()
            else:
                continue
            generators.append((dtype, gen))
        
        data = []
        rate = 1000.0  # events per second
        
        for i in range(n_samples):
            timestamp = time.time() + i / rate
            for dtype, gen in generators:
                value = gen._generate_value(timestamp)
                data.append({
                    'timestamp': timestamp,
                    'value': value,
                    'data_type': dtype
                })
        
        return pd.DataFrame(data)