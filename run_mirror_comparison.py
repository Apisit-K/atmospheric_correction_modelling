"""
Run Mirror Comparison - Execution Script

This script provides a command-line interface for running mirror comparisons
with configurable parameters. It implements the full statistical methodology
from the comparative performance analysis plan.

Usage:
    python run_mirror_comparison.py --Q1 10 --Q2 20 --runs 30
    python run_mirror_comparison.py --config comparison_config.json
    python run_mirror_comparison.py --sweep-Q --Q-range 5,25 --runs 50

Examples:
    # Compare two specific mirrors
    python run_mirror_comparison.py --mirror-A "Mirror_X_Q20" --Q1 20 --mirror-B "Mirror_Y_Q10" --Q2 10

    # Sweep Q factor range
    python run_mirror_comparison.py --sweep-Q --Q-range 5,30 --step 5 --runs 30

    # Compare under different turbulence conditions
    python run_mirror_comparison.py --Q1 20 --Q2 21 --turbulence-sweep 50,100,200 --runs 30

    # Generate report with plots
    python run_mirror_comparison.py --Q1 20 --Q2 21 --report --output-dir ./results
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional
import numpy as np

from mirror_comparison import (
    MirrorConfig,
    SimulationConfig,
    compare_mirrors,
    compare_q_factors,
    print_comparison_report,
    calculate_sample_size,
    MirrorComparisonResult
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='Run mirror comparison simulations with statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python run_mirror_comparison.py --Q1 10 --Q2 20

  # With custom names and more runs
  python run_mirror_comparison.py --mirror-A "High_Q" --Q1 20 --mirror-B "Low_Q" --Q2 10 --runs 50

  # Sweep Q factor range
  python run_mirror_comparison.py --sweep-Q --Q-range 5,30 --step 5

  # Compare under strong turbulence
  python run_mirror_comparison.py --Q1 20 --Q2 21 --f-corner 200 --runs 30

  # Generate full report
  python run_mirror_comparison.py --Q1 20 --Q2 21 --report --output-dir ./results
        """
    )

    # Mirror configurations
    mirror_group = parser.add_argument_group('Mirror Configuration')
    mirror_group.add_argument('--mirror-A', type=str, default='Mirror_A',
                             help='Name for mirror A (default: Mirror_A)')
    mirror_group.add_argument('--Q1', type=float, default=None,
                             help='Quality factor for mirror A')
    mirror_group.add_argument('--mirror-B', type=str, default='Mirror_B',
                             help='Name for mirror B (default: Mirror_B)')
    mirror_group.add_argument('--Q2', type=float, default=None,
                             help='Quality factor for mirror B')
    mirror_group.add_argument('--f-res', type=float, default=437.0,
                             help='Resonant frequency in Hz (default: 437)')

    # PID gains (optional - will auto-tune if not specified)
    pid_group = parser.add_argument_group('PID Gains (optional, auto-tuned if not specified)')
    pid_group.add_argument('--kp1', type=float, help='Proportional gain for mirror A')
    pid_group.add_argument('--ki1', type=float, help='Integral gain for mirror A')
    pid_group.add_argument('--kd1', type=float, help='Derivative gain for mirror A')
    pid_group.add_argument('--kp2', type=float, help='Proportional gain for mirror B')
    pid_group.add_argument('--ki2', type=float, help='Integral gain for mirror B')
    pid_group.add_argument('--kd2', type=float, help='Derivative gain for mirror B')

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--runs', type=int, default=30,
                          help='Number of simulation runs (default: 30)')
    sim_group.add_argument('--dt', type=float, default=0.0001,
                          help='Sampling time in seconds (default: 0.0001)')
    sim_group.add_argument('--duration', type=float, default=0.5,
                          help='Simulation duration in seconds (default: 0.5)')
    sim_group.add_argument('--f0', type=float, default=10.0,
                          help='Inner scale frequency in Hz (default: 10)')
    sim_group.add_argument('--f-corner', type=float, default=100.0,
                          help='Corner frequency in Hz (default: 100)')
    sim_group.add_argument('--scaling', type=float, default=8.0,
                          help='Turbulence amplitude scaling in mrad (default: 8)')
    sim_group.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')

    # Sweep options
    sweep_group = parser.add_argument_group('Parameter Sweep Options')
    sweep_group.add_argument('--sweep-Q', action='store_true',
                            help='Perform Q factor sweep comparison')
    sweep_group.add_argument('--Q-range', type=str,
                            help='Q factor range as min,max (e.g., "5,30")')
    sweep_group.add_argument('--step', type=float, default=5.0,
                            help='Step size for Q sweep (default: 5)')
    sweep_group.add_argument('--turbulence-sweep', type=str,
                            help='Comma-separated corner frequencies (e.g., "50,100,200")')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--report', action='store_true',
                             help='Generate detailed report with plots')
    output_group.add_argument('--output-dir', type=str, default='.',
                             help='Output directory for results (default: current)')
    output_group.add_argument('--save-json', action='store_true',
                             help='Save results to JSON file')
    output_group.add_argument('--quiet', action='store_true',
                             help='Minimal output')

    # Statistical options
    stat_group = parser.add_argument_group('Statistical Options')
    stat_group.add_argument('--alpha', type=float, default=0.05,
                           help='Significance level (default: 0.05)')
    stat_group.add_argument('--calculate-sample-size', action='store_true',
                           help='Calculate required sample size for given effect size')
    stat_group.add_argument('--effect-size', type=float, default=0.5,
                           help='Expected effect size for sample size calculation')

    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')

    return parser


def load_config_from_file(filepath: str) -> Dict:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results_to_json(result: MirrorComparisonResult, filepath: str):
    """Save comparison results to JSON file."""
    data = {
        'mirror_a': result.mirror_a_name,
        'mirror_b': result.mirror_b_name,
        'metrics_a': {
            name: {
                'mean': float(m.mean),
                'std': float(m.std),
                'median': float(m.median),
                'cv': float(m.cv),
                'values': m.values.tolist()
            }
            for name, m in result.metrics_a.items()
        },
        'metrics_b': {
            name: {
                'mean': float(m.mean),
                'std': float(m.std),
                'median': float(m.median),
                'cv': float(m.cv),
                'values': m.values.tolist()
            }
            for name, m in result.metrics_b.items()
        },
        'statistical_tests': {
            name: {
                'test_name': t.test_name,
                'statistic': float(t.statistic),
                'p_value': float(t.p_value),
                'significant': bool(t.significant),
                'effect_size': float(t.effect_size) if t.effect_size else None,
                'effect_interpretation': t.effect_interpretation
            }
            for name, t in result.statistical_tests.items()
        },
        'recommendation': result.recommendation
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def run_single_comparison(args) -> MirrorComparisonResult:
    """Run a single mirror comparison."""
    # Create mirror configurations
    mirror_a = MirrorConfig(
        name=args.mirror_A,
        Q=args.Q1,
        f_res=args.f_res,
        kp=args.kp1,
        ki=args.ki1,
        kd=args.kd1
    )

    mirror_b = MirrorConfig(
        name=args.mirror_B,
        Q=args.Q2,
        f_res=args.f_res,
        kp=args.kp2,
        ki=args.ki2,
        kd=args.kd2
    )

    # Create simulation configuration
    sim_config = SimulationConfig(
        dt=args.dt,
        duration=args.duration,
        f0=args.f0,
        f_corner=args.f_corner,
        scaling=args.scaling,
        seed=args.seed
    )

    # Run comparison
    if not args.quiet:
        print(f"\nRunning comparison: {mirror_a.name} (Q={mirror_a.Q}) vs {mirror_b.name} (Q={mirror_b.Q})")
        print(f"Simulation runs: {args.runs}")
        print(f"Turbulence: f0={args.f0}Hz, f_corner={args.f_corner}Hz")
        print("-" * 60)

    result = compare_mirrors(mirror_a, mirror_b, sim_config, n_runs=args.runs)

    return result


def run_q_sweep(args) -> List[MirrorComparisonResult]:
    """Run Q factor sweep comparison."""
    if not args.Q_range:
        print("Error: --Q-range required for sweep (e.g., --Q-range 5,30)")
        sys.exit(1)

    q_min, q_max = map(float, args.Q_range.split(','))
    q_values = np.arange(q_min, q_max + args.step, args.step)

    if not args.quiet:
        print(f"\nRunning Q factor sweep: {q_min} to {q_max} (step={args.step})")
        print(f"Reference Q: {args.Q1}")
        print("-" * 60)

    results = []
    for q in q_values:
        if abs(q - args.Q1) < 0.1:
            continue  # Skip if same as reference

        # Temporarily modify args
        original_q2 = args.Q2
        args.Q2 = q
        args.mirror_B = f"Mirror_Q{q:.0f}"

        result = run_single_comparison(args)
        results.append(result)

        args.Q2 = original_q2

    return results


def run_turbulence_sweep(args) -> Dict[str, MirrorComparisonResult]:
    """Run comparison under different turbulence conditions."""
    if not args.turbulence_sweep:
        print("Error: --turbulence-sweep requires comma-separated values")
        sys.exit(1)

    f_corners = [float(f) for f in args.turbulence_sweep.split(',')]

    if not args.quiet:
        print(f"\nRunning turbulence sweep: corner frequencies = {f_corners}")
        print(f"Comparing Q={args.Q1} vs Q={args.Q2}")
        print("-" * 60)

    results = {}
    base_f_corner = args.f_corner

    for f_corner in f_corners:
        args.f_corner = f_corner

        if not args.quiet:
            print(f"\n--- Testing with f_corner = {f_corner}Hz ---")

        result = run_single_comparison(args)
        results[f"f_corner_{f_corner}"] = result

    args.f_corner = base_f_corner
    return results


def generate_report(result: MirrorComparisonResult, output_dir: str, args):
    """Generate detailed report with visualizations."""
    try:
        from visualize_comparison import (
            create_comparison_plots,
            create_summary_table,
            save_report
        )

        os.makedirs(output_dir, exist_ok=True)

        # Create plots
        if not args.quiet:
            print(f"\nGenerating visualizations...")

        fig = create_comparison_plots(result)
        plot_path = os.path.join(output_dir, f"comparison_{result.mirror_a_name}_vs_{result.mirror_b_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        if not args.quiet:
            print(f"  Saved plot: {plot_path}")

        # Create summary table
        table_path = os.path.join(output_dir, f"summary_{result.mirror_a_name}_vs_{result.mirror_b_name}.txt")
        create_summary_table(result, table_path)
        if not args.quiet:
            print(f"  Saved table: {table_path}")

        # Save full report
        report_path = os.path.join(output_dir, f"report_{result.mirror_a_name}_vs_{result.mirror_b_name}.md")
        save_report(result, report_path)
        if not args.quiet:
            print(f"  Saved report: {report_path}")

    except ImportError:
        print("Warning: visualize_comparison module not available, skipping plots")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load config file if specified
    if args.config:
        config = load_config_from_file(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)

    # Calculate sample size if requested
    if args.calculate_sample_size:
        n = calculate_sample_size(args.effect_size)
        print(f"\nSample Size Calculation")
        print(f"Expected effect size (Cohen's d): {args.effect_size}")
        print(f"Significance level (alpha): {args.alpha}")
        print(f"Statistical power: 80%")
        print(f"Required sample size per group: {n}")
        print(f"Total runs needed: {2 * n}")
        return

    # Validate required arguments for comparison runs
    if args.Q1 is None or args.Q2 is None:
        parser.error("--Q1 and --Q2 are required for comparison runs")

    # Run comparison(s)
    if args.sweep_Q:
        results = run_q_sweep(args)

        # Print summary
        print("\n" + "="*80)
        print("Q FACTOR SWEEP SUMMARY")
        print("="*80)
        for result in results:
            rms_a = result.metrics_a['rms_error'].mean
            rms_b = result.metrics_b['rms_error'].mean
            improvement = (rms_a - rms_b) / rms_a * 100
            print(f"{result.mirror_b_name}: RMS error = {rms_b:.4f} mrad "
                  f"({improvement:+.1f}% vs reference)")

    elif args.turbulence_sweep:
        results = run_turbulence_sweep(args)

        # Print summary
        print("\n" + "="*80)
        print("TURBULENCE SWEEP SUMMARY")
        print("="*80)
        for condition, result in results.items():
            rms_a = result.metrics_a['rms_error'].mean
            rms_b = result.metrics_b['rms_error'].mean
            print(f"{condition}: {result.mirror_a_name} RMS={rms_a:.4f}, "
                  f"{result.mirror_b_name} RMS={rms_b:.4f}")

    else:
        # Single comparison
        result = run_single_comparison(args)

        # Print report
        if not args.quiet:
            print_comparison_report(result)

        # Generate report with plots
        if args.report:
            generate_report(result, args.output_dir, args)

        # Save to JSON
        if args.save_json:
            json_path = os.path.join(
                args.output_dir,
                f"results_{result.mirror_a_name}_vs_{result.mirror_b_name}.json"
            )
            save_results_to_json(result, json_path)
            if not args.quiet:
                print(f"\nSaved results to: {json_path}")


if __name__ == "__main__":
    main()
