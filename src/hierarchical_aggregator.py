"""
Hierarchical Streaming Aggregator
Implements the O(1) update, O(log n) query hierarchical algorithm
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class Aggregate:
    """Base class for all aggregates"""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = -float('inf')
    first: Optional[float] = None
    last: Optional[float] = None
    sum_squares: float = 0.0  # For variance
    
    def update(self, value: float):
        """Update aggregate with new value"""
        self.count += 1
        self.sum += value
        self.sum_squares += value * value
        
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
            
        if self.first is None:
            self.first = value
        self.last = value
    
    def merge(self, other: 'Aggregate') -> 'Aggregate':
        """Merge two aggregates (decomposable operation)"""
        result = Aggregate()
        result.count = self.count + other.count
        result.sum = self.sum + other.sum
        result.sum_squares = self.sum_squares + other.sum_squares
        result.min = min(self.min, other.min)
        result.max = max(self.max, other.max)
        result.first = self.first if self.first is not None else other.first
        result.last = other.last if other.last is not None else self.last
        return result
    
    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return (self.sum_squares - (self.sum * self.sum) / self.count) / (self.count - 1)
    
    def to_ohlcv(self) -> Dict[str, float]:
        """Convert to OHLCV format"""
        return {
            'open': self.first if self.first is not None else 0.0,
            'high': self.max,
            'low': self.min,
            'close': self.last if self.last is not None else 0.0,
            'volume': self.sum if self.sum > 0 else self.count  # Use sum as volume proxy
        }


class Block:
    """Represents a time block in the hierarchy"""
    def __init__(self, start_time: float, duration: float):
        self.start_time = start_time
        self.end_time = start_time + duration
        self.duration = duration
        self.aggregate = Aggregate()
        self.is_complete = False
        
    def update(self, timestamp: float, value: float):
        """Add value to block"""
        if timestamp >= self.start_time and timestamp < self.end_time:
            self.aggregate.update(value)
            return True
        return False
    
    def finalize(self):
        """Mark block as complete"""
        self.is_complete = True
        return self.aggregate


class Level:
    """Represents a level in the hierarchy"""
    def __init__(self, level_idx: int, base_delta: float):
        self.level_idx = level_idx
        self.block_duration = base_delta * (2 ** level_idx)
        self.current_block: Optional[Block] = None
        self.previous_blocks: List[Block] = []
        self.blocks_pool: List[Block] = []  # For memory reuse
        
    def get_block(self, start_time: float) -> Block:
        """Get or create a block"""
        if self.blocks_pool:
            block = self.blocks_pool.pop()
            block.start_time = start_time
            block.end_time = start_time + self.block_duration
            block.aggregate = Aggregate()
            block.is_complete = False
        else:
            block = Block(start_time, self.block_duration)
        return block
    
    def recycle_block(self, block: Block):
        """Recycle block for reuse"""
        self.blocks_pool.append(block)


class HierarchicalAggregator:
    """
    Main hierarchical streaming aggregator
    
    Args:
        base_delta: Smallest timeframe resolution
        max_timeframe: Largest timeframe to support
    """
    def __init__(self, base_delta: float = 1.0, max_timeframe: float = 3600.0):
        self.base_delta = base_delta
        self.max_timeframe = max_timeframe
        
        # Calculate hierarchy depth
        self.L = int(np.ceil(np.log2(max_timeframe / base_delta))) + 1
        
        # Initialize levels
        self.levels: List[Level] = []
        for l in range(self.L):
            self.levels.append(Level(l, base_delta))
        
        # Statistics
        self.update_count = 0
        self.merge_count = 0
        self.last_update_time = time.time()
        
    def update(self, timestamp: float, value: float) -> None:
        """Process new element with amortized O(1) complexity"""
        self.update_count += 1
        
        for l, level in enumerate(self.levels):
            # Check if we need to create new block
            if (level.current_block is None or 
                timestamp >= level.current_block.end_time):
                
                # Finalize current block if exists
                if level.current_block is not None:
                    level.current_block.finalize()
                    level.previous_blocks.append(level.current_block)
                    
                    # Keep only last 2 blocks for merging
                    if len(level.previous_blocks) > 2:
                        old_block = level.previous_blocks.pop(0)
                        level.recycle_block(old_block)
                    
                    # Check if we can merge blocks from previous level
                    if l > 0 and len(level.previous_blocks) >= 2:
                        self._try_merge_blocks(l)
                
                # Create new block aligned to block boundaries
                block_start = timestamp - (timestamp % level.block_duration)
                level.current_block = level.get_block(block_start)
            
            # Update current block
            level.current_block.update(timestamp, value)
        
        # Amortized cleanup
        if self.update_count % 1000 == 0:
            self._cleanup()
    
    def _try_merge_blocks(self, level_idx: int) -> None:
        """Try to merge two blocks from lower level"""
        level = self.levels[level_idx]
        if len(level.previous_blocks) < 2:
            return
        
        # Check if two consecutive blocks form a complete parent block
        block1 = level.previous_blocks[-2]
        block2 = level.previous_blocks[-1]
        
        if (block2.start_time - block1.start_time == level.block_duration and
            block1.is_complete and block2.is_complete):
            
            # Merge aggregates
            merged_agg = block1.aggregate.merge(block2.aggregate)
            
            # Create merged block in parent level
            parent_level = self.levels[level_idx + 1]
            merged_block = Block(block1.start_time, parent_level.block_duration)
            merged_block.aggregate = merged_agg
            merged_block.is_complete = True
            
            # Add to parent level
            parent_level.previous_blocks.append(merged_block)
            self.merge_count += 1
            
            # Recycle old blocks
            level.recycle_block(block1)
            level.recycle_block(block2)
            level.previous_blocks = level.previous_blocks[:-2]
    
    def query(self, timeframe: float, current_time: float) -> Dict[str, float]:
        """
        Query aggregates for given timeframe with O(log n) complexity
        
        Args:
            timeframe: Time window to query
            current_time: Current timestamp
            
        Returns:
            Dictionary with aggregated statistics
        """
        if timeframe < self.base_delta or timeframe > self.max_timeframe:
            raise ValueError(f"Timeframe must be between {self.base_delta} and {self.max_timeframe}")
        
        start_time = current_time - timeframe
        blocks = self._decompose_interval(start_time, current_time)
        
        # Merge aggregates from all blocks
        result = Aggregate()
        for block in blocks:
            result = result.merge(block.aggregate)
        
        return result.to_ohlcv()
    
    def _decompose_interval(self, start: float, end: float) -> List[Block]:
        """
        Decompose time interval into minimal set of blocks
        Uses at most 2*log2(Δ/δ) blocks
        """
        blocks = []
        remaining_start = start
        
        # Try to use largest blocks first
        for level in reversed(self.levels):
            if level.block_duration > (end - start):
                continue
            
            # Find blocks covering the interval
            for block in level.previous_blocks:
                if (block.start_time >= remaining_start and 
                    block.end_time <= end and 
                    block.is_complete):
                    
                    blocks.append(block)
                    remaining_start = block.end_time
            
            # Check current block
            if (level.current_block and 
                level.current_block.start_time >= remaining_start and
                level.current_block.end_time <= end):
                
                blocks.append(level.current_block)
                remaining_start = level.current_block.end_time
        
        return blocks
    
    def _cleanup(self):
        """Periodic cleanup of old blocks"""
        current_time = time.time()
        cutoff_time = current_time - self.max_timeframe
        
        for level in self.levels:
            level.previous_blocks = [
                b for b in level.previous_blocks 
                if b.end_time > cutoff_time
            ]
    
    @property
    def memory_usage(self) -> int:
        """Estimate memory usage in blocks"""
        total = 0
        for level in self.levels:
            total += len(level.previous_blocks)
            if level.current_block:
                total += 1
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'update_count': self.update_count,
            'merge_count': self.merge_count,
            'memory_blocks': self.memory_usage,
            'levels': self.L,
            'update_rate': self.update_count / max(1, time.time() - self.last_update_time)
        }