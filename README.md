# science-
Algorithm Overview This implementation provides: 
Amortized O(1) update time; 
O(log n) query time for arbitrary timeframes; 
Θ(Δ_max/δ) space complexity; 
Provable error bounds for decomposable aggregates

# Hierarchical Streaming Aggregation

A Python implementation of the optimal hierarchical aggregation algorithm for multi-scale time series with provable approximation guarantees.

## Quick Start

```bash
# Clone repository
git clone https://github.com/nikita-zmanovskii/science
cd science

# Install dependencies
pip install -r requirements.txt

# Run demo
python -m src.experiments run_demo

# Run all experiments
python scripts/run_experiments.py
