"""
Visualization and Reporting for Mirror Comparison

This module provides plotting and report generation functions for the
mirror comparison framework, following the reporting guidelines from
the comparative performance analysis methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os

from mirror_comparison import MirrorComparisonResult, MetricResult


def create_box_plot_comparison(
    result: MirrorComparisonResult,
    metric_name: str,
    ax: Optional[plt.Axes] = None,
    show_stats: bool = True
) -> plt.Axes:
    """
    Create box plot comparing a metric between two mirrors.

    Args:
        result: Comparison result
        metric_name: Name of metric to plot
        ax: Optional axes to plot on
        show_stats: Whether to show statistics on plot

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    data_a = result.metrics_a[metric_name].values
    data_b = result.metrics_b[metric_name].values

    bp = ax.boxplot([data_a, data_b], patch_artist=True, showmeans=True)
    ax.set_xticklabels([result.mirror_a_name, result.mirror_b_name])

    # Color boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
    ax.grid(True, alpha=0.3)

    # Add statistics
    if show_stats:
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        ax.text(0.5, 0.95, f'Mean: {mean_a:.4f} vs {mean_b:.4f}',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def create_distribution_plot(
    result: MirrorComparisonResult,
    metric_name: str,
    ax: Optional[plt.Axes] = None,
    bins: int = 20
) -> plt.Axes:
    """
    Create distribution/histogram comparison.

    Args:
        result: Comparison result
        metric_name: Name of metric to plot
        ax: Optional axes to plot on
        bins: Number of histogram bins

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    data_a = result.metrics_a[metric_name].values
    data_b = result.metrics_b[metric_name].values

    ax.hist(data_a, bins=bins, alpha=0.5, label=result.mirror_a_name, color='blue', density=True)
    ax.hist(data_b, bins=bins, alpha=0.5, label=result.mirror_b_name, color='green', density=True)

    # Add mean lines
    ax.axvline(np.mean(data_a), color='blue', linestyle='--', linewidth=2, label=f'{result.mirror_a_name} mean')
    ax.axvline(np.mean(data_b), color='green', linestyle='--', linewidth=2, label=f'{result.mirror_b_name} mean')

    ax.set_xlabel(metric_name.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.set_title(f'{metric_name.replace("_", " ").title()} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def create_bland_altman_plot(
    result: MirrorComparisonResult,
    metric_name: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create Bland-Altman plot for agreement analysis.

    Args:
        result: Comparison result
        metric_name: Name of metric to plot
        ax: Optional axes to plot on

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    data_a = result.metrics_a[metric_name].values
    data_b = result.metrics_b[metric_name].values

    # Calculate differences and means
    differences = data_a - data_b
    means = (data_a + data_b) / 2

    # Calculate limits of agreement
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Scatter plot
    ax.scatter(means, differences, alpha=0.6)

    # Reference lines
    ax.axhline(mean_diff, color='red', linestyle='--', label=f'Mean diff: {mean_diff:.4f}')
    ax.axhline(loa_upper, color='orange', linestyle=':', label=f'+1.96 SD: {loa_upper:.4f}')
    ax.axhline(loa_lower, color='orange', linestyle=':', label=f'-1.96 SD: {loa_lower:.4f}')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Mean of Two Mirrors')
    ax.set_ylabel(f'Difference ({result.mirror_a_name} - {result.mirror_b_name})')
    ax.set_title(f'Bland-Altman Plot: {metric_name.replace("_", " ").title()}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def create_metrics_summary_plot(
    result: MirrorComparisonResult,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create summary plot with all metrics.

    Args:
        result: Comparison result
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    metrics = list(result.metrics_a.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i < len(axes):
            create_box_plot_comparison(result, metric, ax=axes[i], show_stats=True)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Mirror Comparison: {result.mirror_a_name} vs {result.mirror_b_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def create_comparison_plots(
    result: MirrorComparisonResult,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive comparison visualization.

    Args:
        result: Comparison result
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Primary metric: RMS Error
    ax1 = fig.add_subplot(gs[0, 0])
    create_box_plot_comparison(result, 'rms_error', ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    create_distribution_plot(result, 'rms_error', ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    create_bland_altman_plot(result, 'rms_error', ax=ax3)

    # Secondary metrics
    ax4 = fig.add_subplot(gs[1, 0])
    create_box_plot_comparison(result, 'max_error', ax=ax4)
    ax4.set_title('Max Error')

    ax5 = fig.add_subplot(gs[1, 1])
    create_box_plot_comparison(result, 'std_error', ax=ax5)
    ax5.set_title('Std Error')

    ax6 = fig.add_subplot(gs[1, 2])
    create_box_plot_comparison(result, 'mean_abs_error', ax=ax6)
    ax6.set_title('Mean Absolute Error')

    # Additional metrics
    ax7 = fig.add_subplot(gs[2, 0])
    create_box_plot_comparison(result, 'settling_time', ax=ax7)
    ax7.set_title('Settling Time')

    ax8 = fig.add_subplot(gs[2, 1])
    create_box_plot_comparison(result, 'rejection_ratio', ax=ax8)
    ax8.set_title('Rejection Ratio')

    # Statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Create statistics table
    table_data = []
    table_data.append(['Metric', f'{result.mirror_a_name}', f'{result.mirror_b_name}', 'Winner'])
    table_data.append(['-'*15, '-'*15, '-'*15, '-'*15])

    for metric_name in result.metrics_a.keys():
        ma = result.metrics_a[metric_name]
        mb = result.metrics_b[metric_name]

        # Lower is better for most metrics
        if metric_name == 'rejection_ratio':
            winner = result.mirror_a_name if ma.mean > mb.mean else result.mirror_b_name
        else:
            winner = result.mirror_a_name if ma.mean < mb.mean else result.mirror_b_name

        table_data.append([
            metric_name.replace('_', ' ').title()[:14],
            f'{ma.mean:.4f}',
            f'{mb.mean:.4f}',
            winner[:14]
        ])

    table = ax9.table(cellText=table_data, loc='center', cellLoc='left',
                       colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)

    plt.suptitle(f'Mirror Comparison Report: {result.mirror_a_name} vs {result.mirror_b_name}',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig


def create_sweep_visualization(
    results: List[MirrorComparisonResult],
    sweep_parameter: str = 'Q',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualization for parameter sweep results.

    Args:
        results: List of comparison results
        sweep_parameter: Name of swept parameter
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data
    sweep_values = []
    rms_errors = []
    max_errors = []
    rejection_ratios = []

    for result in results:
        # Extract Q value from mirror name
        try:
            q_val = float(result.mirror_b_name.split('Q')[-1])
            sweep_values.append(q_val)
            rms_errors.append(result.metrics_b['rms_error'].mean)
            max_errors.append(result.metrics_b['max_error'].mean)
            rejection_ratios.append(result.metrics_b['rejection_ratio'].mean)
        except (ValueError, IndexError):
            continue

    # Sort by sweep parameter
    sorted_indices = np.argsort(sweep_values)
    sweep_values = np.array(sweep_values)[sorted_indices]
    rms_errors = np.array(rms_errors)[sorted_indices]
    max_errors = np.array(max_errors)[sorted_indices]
    rejection_ratios = np.array(rejection_ratios)[sorted_indices]

    # Plot RMS Error
    ax = axes[0, 0]
    ax.plot(sweep_values, rms_errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(f'{sweep_parameter} Factor')
    ax.set_ylabel('RMS Error (mrad)')
    ax.set_title('RMS Error vs Q Factor')
    ax.grid(True, alpha=0.3)

    # Plot Max Error
    ax = axes[0, 1]
    ax.plot(sweep_values, max_errors, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel(f'{sweep_parameter} Factor')
    ax.set_ylabel('Max Error (mrad)')
    ax.set_title('Max Error vs Q Factor')
    ax.grid(True, alpha=0.3)

    # Plot Rejection Ratio
    ax = axes[1, 0]
    ax.plot(sweep_values, rejection_ratios, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel(f'{sweep_parameter} Factor')
    ax.set_ylabel('Rejection Ratio')
    ax.set_title('Disturbance Rejection vs Q Factor')
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    Parameter Sweep Summary
    ========================
    Parameter: {sweep_parameter}
    Range: {sweep_values[0]:.1f} to {sweep_values[-1]:.1f}
    Steps: {len(sweep_values)}

    Best RMS Error:
      Q = {sweep_values[np.argmin(rms_errors)]:.1f}
      RMS = {np.min(rms_errors):.4f} mrad

    Worst RMS Error:
      Q = {sweep_values[np.argmax(rms_errors)]:.1f}
      RMS = {np.max(rms_errors):.4f} mrad

    Recommendation:
      Optimal Q range: {sweep_values[np.argmin(rms_errors)]:.0f}
    """

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{sweep_parameter} Factor Sweep Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def create_summary_table(result: MirrorComparisonResult, filepath: str):
    """
    Create text summary table and save to file.

    Args:
        result: Comparison result
        filepath: Output file path
    """
    lines = []
    lines.append("="*80)
    lines.append(f"MIRROR COMPARISON SUMMARY: {result.mirror_a_name} vs {result.mirror_b_name}")
    lines.append("="*80)
    lines.append("")

    # Metrics table
    lines.append("Performance Metrics")
    lines.append("-"*80)
    lines.append(f"{'Metric':<25} {'Mirror A (mean±SD)':<25} {'Mirror B (mean±SD)':<25}")
    lines.append("-"*80)

    for metric_name in result.metrics_a.keys():
        ma = result.metrics_a[metric_name]
        mb = result.metrics_b[metric_name]
        lines.append(f"{metric_name:<25} {ma.mean:.4f}±{ma.std:.4f}          {mb.mean:.4f}±{mb.std:.4f}")

    lines.append("")
    lines.append("Statistical Tests")
    lines.append("-"*80)
    lines.append(f"{'Test':<40} {'p-value':<15} {'Significant':<12} {'Effect Size'}")
    lines.append("-"*80)

    for test_name, test in result.statistical_tests.items():
        sig = "Yes" if test.significant else "No"
        effect = f"{test.effect_size:.3f}" if test.effect_size else "N/A"
        lines.append(f"{test_name:<40} {test.p_value:<15.6f} {sig:<12} {effect}")

    lines.append("")
    lines.append("Recommendation")
    lines.append("-"*80)
    lines.append(result.recommendation)
    lines.append("="*80)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def save_report(result: MirrorComparisonResult, filepath: str):
    """
    Generate comprehensive markdown report.

    Args:
        result: Comparison result
        filepath: Output file path
    """
    lines = []
    lines.append(f"# Mirror Comparison Report: {result.mirror_a_name} vs {result.mirror_b_name}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(result.recommendation)
    lines.append("")

    # Performance metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Mirror A (mean±SD) | Mirror B (mean±SD) | Winner |")
    lines.append("|--------|-------------------|-------------------|--------|")

    for metric_name in result.metrics_a.keys():
        ma = result.metrics_a[metric_name]
        mb = result.metrics_b[metric_name]

        if metric_name == 'rejection_ratio':
            winner = result.mirror_a_name if ma.mean > mb.mean else result.mirror_b_name
        else:
            winner = result.mirror_a_name if ma.mean < mb.mean else result.mirror_b_name

        lines.append(f"| {metric_name.replace('_', ' ').title()} | "
                    f"{ma.mean:.4f}±{ma.std:.4f} | {mb.mean:.4f}±{mb.std:.4f} | {winner} |")

    lines.append("")

    # Statistical tests
    lines.append("## Statistical Analysis")
    lines.append("")
    lines.append("| Test | Statistic | p-value | Significant | Effect Size |")
    lines.append("|------|-----------|---------|-------------|-------------|")

    for test_name, test in result.statistical_tests.items():
        sig = "Yes" if test.significant else "No"
        effect = f"{test.effect_size:.3f} ({test.effect_interpretation})" if test.effect_size else "N/A"
        lines.append(f"| {test_name} | {test.statistic:.4f} | {test.p_value:.6f} | {sig} | {effect} |")

    lines.append("")

    # Detailed statistics
    lines.append("## Detailed Statistics")
    lines.append("")

    for metric_name in result.metrics_a.keys():
        ma = result.metrics_a[metric_name]
        mb = result.metrics_b[metric_name]

        lines.append(f"### {metric_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"**{result.mirror_a_name}:**")
        lines.append(f"- Mean: {ma.mean:.6f}")
        lines.append(f"- Std Dev: {ma.std:.6f}")
        lines.append(f"- Median: {ma.median:.6f}")
        lines.append(f"- CV: {ma.cv:.2f}%")
        lines.append(f"- 95% CI: ({ma.confidence_interval(0.95)[0]:.6f}, {ma.confidence_interval(0.95)[1]:.6f})")
        lines.append("")
        lines.append(f"**{result.mirror_b_name}:**")
        lines.append(f"- Mean: {mb.mean:.6f}")
        lines.append(f"- Std Dev: {mb.std:.6f}")
        lines.append(f"- Median: {mb.median:.6f}")
        lines.append(f"- CV: {mb.cv:.2f}%")
        lines.append(f"- 95% CI: ({mb.confidence_interval(0.95)[0]:.6f}, {mb.confidence_interval(0.95)[1]:.6f})")
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append(result.recommendation)
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by Mirror Comparison Framework*")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def create_time_series_plot(
    time: np.ndarray,
    disturbance: np.ndarray,
    mirror_a_pos: np.ndarray,
    mirror_b_pos: np.ndarray,
    mirror_a_name: str,
    mirror_b_name: str,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Create time series comparison plot.

    Args:
        time: Time array
        disturbance: Atmospheric disturbance
        mirror_a_pos: Mirror A position
        mirror_b_pos: Mirror B position
        mirror_a_name: Name of mirror A
        mirror_b_name: Name of mirror B
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top plot: All signals
    ax = axes[0]
    ax.plot(time, disturbance, 'k-', label='Atmospheric Disturbance', alpha=0.7, linewidth=1)
    ax.plot(time, mirror_a_pos, 'b--', label=f'{mirror_a_name} Response', linewidth=1.5)
    ax.plot(time, mirror_b_pos, 'g--', label=f'{mirror_b_name} Response', linewidth=1.5)
    ax.set_ylabel('Position (mrad)')
    ax.set_title('Mirror Response Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom plot: Residual errors
    ax = axes[1]
    error_a = disturbance - mirror_a_pos
    error_b = disturbance - mirror_b_pos
    ax.plot(time, error_a, 'b-', label=f'{mirror_a_name} Error', linewidth=1)
    ax.plot(time, error_b, 'g-', label=f'{mirror_b_name} Error', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual Error (mrad)')
    ax.set_title('Tracking Error Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization module for Mirror Comparison Framework")
    print("Import this module to use visualization functions")
