"""
Mirror Comparison: S040512 vs S72026

This script compares two Physik Instrumente FSM mirrors:
- S040512: Lower frequency (437 Hz), moderate Q (20-21)
- S72026: Higher frequency (889 Hz), high Q (52-56)

Specs:
========
S040512:
  - Resonant frequency: 437 Hz
  - Q: 21 (X-axis), 20 (Y-axis)
  - Recommended LPF: 160 Hz (6th Order Bessel)

S72026:
  - Resonant frequency: 889 Hz
  - Q: 52 (X-axis), 56 (Y-axis)
  - Recommended LPF: 380 Hz (6th Order Bessel)

The S72026 has:
  - 2x higher resonant frequency (better bandwidth)
  - 2.5x higher Q (sharper resonance, requires more conservative PID)
  - 2.4x higher LPF cutoff (can handle faster disturbances)
"""

from mirror_comparison import (
    MirrorConfig, SimulationConfig, compare_mirrors,
    print_comparison_report, compare_q_factors
)
import numpy as np


def compare_s040512_vs_s72026(n_runs=30):
    """
    Compare S040512 vs S72026 mirrors.
    Uses X-axis specs for 1D comparison.
    """
    print("="*80)
    print("MIRROR COMPARISON: S040512 vs S72026")
    print("="*80)
    print()
    print("S040512 Specs:")
    print("  - Resonant frequency: 437 Hz")
    print("  - Q: 21 (X-axis), 20 (Y-axis)")
    print("  - Recommended LPF: 160 Hz")
    print()
    print("S72026 Specs:")
    print("  - Resonant frequency: 889 Hz")
    print("  - Q: 52 (X-axis), 56 (Y-axis)")
    print("  - Recommended LPF: 380 Hz")
    print()
    print("="*80)
    print()

    # S040512 - Lower frequency, moderate Q
    # Optimal gains from tuning: kp=0.9595, ki=2.1000, kd=0.000070
    s040512 = MirrorConfig(
        name="S040512",
        Q=21.0,              # X-axis Q
        f_res=437.0,         # Resonant frequency
        kp=0.9595,           # Tuned P gain
        ki=2.1000,           # Tuned I gain
        kd=0.000070,         # Tuned D gain
    )

    # S72026 - Higher frequency, high Q
    # Optimal gains from tuning: kp=0.9667, ki=0.7000, kd=0.000035
    s72026 = MirrorConfig(
        name="S72026",
        Q=52.0,              # X-axis Q
        f_res=889.0,         # Resonant frequency
        kp=0.9667,           # Tuned P gain
        ki=0.7000,           # Tuned I gain
        kd=0.000035,         # Tuned D gain
    )

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.0001,           # 10 kHz sampling
        duration=0.5,        # 0.5 second
        f0=10.0,             # Inner scale frequency
        f_corner=100.0,      # Moderate turbulence
        scaling=8.0,         # 8 mrad amplitude
        seed=42
    )

    # Run comparison
    result = compare_mirrors(s040512, s72026, sim_config, n_runs=n_runs)
    print_comparison_report(result)

    return result


def compare_under_different_turbulence(n_runs=30):
    """
    Compare mirrors under different turbulence conditions.
    """
    print("\n" + "="*80)
    print("TURBULENCE SWEEP: S040512 vs S72026")
    print("="*80)

    # Mirror configs with optimal gains
    s040512 = MirrorConfig(
        name="S040512", Q=21.0, f_res=437.0,
        kp=0.9595, ki=2.1000, kd=0.000070
    )
    s72026 = MirrorConfig(
        name="S72026", Q=52.0, f_res=889.0,
        kp=0.9667, ki=0.7000, kd=0.000035
    )

    # Test different corner frequencies
    f_corners = [50, 100, 200, 300]

    print(f"\n{'Turbulence':<15} {'S040512 RMS':<15} {'S72026 RMS':<15} {'Winner':<15}")
    print("-"*60)

    results = []
    for f_corner in f_corners:
        sim_config = SimulationConfig(
            dt=0.0001,
            duration=0.5,
            f0=10.0,
            f_corner=f_corner,
            scaling=8.0,
            seed=42
        )

        result = compare_mirrors(s040512, s72026, sim_config, n_runs=n_runs)
        rms_040512 = result.metrics_a['rms_error'].mean
        rms_72026 = result.metrics_b['rms_error'].mean
        winner = "S72026" if rms_72026 < rms_040512 else "S040512"

        print(f"{f_corner} Hz{'':<10} {rms_040512:<15.4f} {rms_72026:<15.4f} {winner:<15}")
        results.append((f_corner, rms_040512, rms_72026))

    print()
    return results


def compare_axis_performance(n_runs=30):
    """
    Compare X vs Y axis performance for each mirror.
    """
    print("\n" + "="*80)
    print("AXIS COMPARISON: X vs Y")
    print("="*80)

    # S040512 X vs Y
    print("\nS040512:")
    print("-"*40)
    s040512_x = MirrorConfig(name="S040512_X", Q=21.0, f_res=437.0,
                              kp=0.9595, ki=2.1000, kd=0.000070)
    s040512_y = MirrorConfig(name="S040512_Y", Q=20.0, f_res=437.0,
                              kp=1.0095, ki=2.2050, kd=0.000073)  # Slightly adjusted for Q=20

    sim_config = SimulationConfig(seed=42)
    result = compare_mirrors(s040512_x, s040512_y, sim_config, n_runs=n_runs)
    rms_x = result.metrics_a['rms_error'].mean
    rms_y = result.metrics_b['rms_error'].mean
    print(f"  X-axis (Q=21): RMS = {rms_x:.4f} mrad")
    print(f"  Y-axis (Q=20): RMS = {rms_y:.4f} mrad")
    print(f"  Difference: {abs(rms_x - rms_y):.4f} mrad ({abs(rms_x - rms_y)/rms_x*100:.1f}%)")

    # S72026 X vs Y
    print("\nS72026:")
    print("-"*40)
    s72026_x = MirrorConfig(name="S72026_X", Q=52.0, f_res=889.0,
                             kp=0.9667, ki=0.7000, kd=0.000035)
    s72026_y = MirrorConfig(name="S72026_Y", Q=56.0, f_res=889.0,
                             kp=0.9333, ki=0.6500, kd=0.000033)  # Slightly adjusted for Q=56

    result = compare_mirrors(s72026_x, s72026_y, sim_config, n_runs=n_runs)
    rms_x = result.metrics_a['rms_error'].mean
    rms_y = result.metrics_b['rms_error'].mean
    print(f"  X-axis (Q=52): RMS = {rms_x:.4f} mrad")
    print(f"  Y-axis (Q=56): RMS = {rms_y:.4f} mrad")
    print(f"  Difference: {abs(rms_x - rms_y):.4f} mrad ({abs(rms_x - rms_y)/rms_x*100:.1f}%)")


def generate_full_report(n_runs=50, output_dir="./s040512_vs_s72026"):
    """
    Generate comprehensive report with visualizations.
    """
    import os
    from visualize_comparison import create_comparison_plots, save_report, create_summary_table

    print("\nGenerating comprehensive report...")
    os.makedirs(output_dir, exist_ok=True)

    # Run main comparison
    result = compare_s040512_vs_s72026(n_runs=n_runs)

    # Create plots
    fig = create_comparison_plots(result)
    plot_path = os.path.join(output_dir, "comparison_S040512_vs_S72026.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")

    # Save report
    report_path = os.path.join(output_dir, "report_S040512_vs_S72026.md")
    save_report(result, report_path)
    print(f"  Saved report: {report_path}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary_S040512_vs_S72026.txt")
    create_summary_table(result, summary_path)
    print(f"  Saved summary: {summary_path}")

    print(f"\nReport generated in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare S040512 vs S72026 mirrors')
    parser.add_argument('--runs', type=int, default=30, help='Number of simulation runs')
    parser.add_argument('--turbulence-sweep', action='store_true', help='Run turbulence sweep')
    parser.add_argument('--axis-comparison', action='store_true', help='Compare X vs Y axis')
    parser.add_argument('--full-report', action='store_true', help='Generate full report with plots')

    args = parser.parse_args()

    if args.full_report:
        generate_full_report(n_runs=args.runs)
    elif args.turbulence_sweep:
        compare_under_different_turbulence(n_runs=args.runs)
    elif args.axis_comparison:
        compare_axis_performance(n_runs=args.runs)
    else:
        # Default: basic comparison
        compare_s040512_vs_s72026(n_runs=args.runs)
