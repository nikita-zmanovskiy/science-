#!/usr/bin/env python3
"""
Main script to run all experiments
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from src.experiments import Experiments


def main():
    parser = argparse.ArgumentParser(description='Run hierarchical streaming experiments')
    parser.add_argument('--config', type=str, default='configs/financial.yaml',
                       help='Configuration file')
    parser.add_argument('--experiment', type=str, choices=['all', 'financial', 'iot', 'scalability'],
                       default='all', help='Experiment to run')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run experiments
    exp = Experiments(args.config)
    
    if args.experiment == 'all':
        results = exp.run_all()
    elif args.experiment == 'financial':
        results = exp.experiment_financial()
    elif args.experiment == 'iot':
        results = exp.experiment_iot()
    elif args.experiment == 'scalability':
        results = exp.experiment_scalability()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return
    
    print(f"\nResults saved to {args.output}/")
    
    # Generate visualizations
    if args.experiment in ['financial', 'all']:
        _plot_financial_results(results.get('financial', {}))


def _plot_financial_results(results):
    """Plot financial experiment results"""
    import matplotlib.pyplot as plt
    
    if not results:
        return
    
    # Plot errors by timeframe
    if 'avg_errors' in results:
        timeframes = list(results['avg_errors'].keys())
        errors = list(results['avg_errors'].values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(timeframes, errors, 'o-', linewidth=2, markersize=8)
        plt.xscale('log')
        plt.xlabel('Timeframe (seconds)')
        plt.ylabel('Average Error (%)')
        plt.title('Query Error vs Timeframe')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/financial_errors.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Generated plots in results/")


if __name__ == '__main__':
    main()