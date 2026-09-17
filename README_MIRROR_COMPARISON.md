# Mirror Comparison Framework

A rigorous statistical framework for comparing mirror models in atmospheric correction systems, implementing the methodology from the Comparative Performance Analysis Plan.

## Overview

This framework provides:
- **Statistical hypothesis testing** (t-test, Welch's t-test, Mann-Whitney U)
- **Effect size calculation** (Cohen's d)
- **Power analysis** for sample size determination
- **Outlier detection** and normality testing
- **Comprehensive reporting** with visualizations

## Files

| File | Description |
|------|-------------|
| `mirror_comparison.py` | Core framework with simulation and statistical analysis |
| `run_mirror_comparison.py` | Command-line interface for running comparisons |
| `visualize_comparison.py` | Plotting and report generation utilities |

## Quick Start

### Basic Comparison

Compare two mirrors with different Q factors:

```bash
python run_mirror_comparison.py --Q1 10 --Q2 20 --runs 30
```

### With Custom Names

```bash
python run_mirror_comparison.py --mirror-A "High_Q_Mirror" --Q1 20 --mirror-B "Low_Q_Mirror" --Q2 10 --runs 50
```

### Generate Full Report with Plots

```bash
python run_mirror_comparison.py --Q1 20 --Q2 21 --runs 30 --report --output-dir ./results --save-json
```

### Q Factor Sweep

Compare across a range of Q factors:

```bash
python run_mirror_comparison.py --sweep-Q --Q-range 5,30 --step 5 --runs 30
```

### Turbulence Condition Sweep

Test under different turbulence strengths:

```bash
python run_mirror_comparison.py --Q1 20 --Q2 21 --turbulence-sweep "50,100,200" --runs 30
```

### Calculate Required Sample Size

```bash
python run_mirror_comparison.py --calculate-sample-size --effect-size 0.5
```

## Python API

### Basic Usage

```python
from mirror_comparison import compare_q_factors, print_comparison_report

# Compare Q=10 vs Q=20 mirrors
result = compare_q_factors(Q1=10.0, Q2=20.0, n_runs=30)
print_comparison_report(result)
```

### Advanced Usage

```python
from mirror_comparison import (
    MirrorConfig, SimulationConfig, compare_mirrors
)

# Define mirror configurations
mirror_a = MirrorConfig(name="Mirror_A", Q=20.0)
mirror_b = MirrorConfig(name="Mirror_B", Q=21.0)

# Configure simulation
sim_config = SimulationConfig(
    dt=0.0001,           # 10 kHz sampling
    duration=0.5,        # 0.5 second simulation
    f0=10.0,             # Inner scale frequency
    f_corner=100.0,      # Corner frequency
    scaling=8.0,         # Turbulence amplitude
    seed=42              # Random seed
)

# Run comparison
result = compare_mirrors(
    mirror_a, mirror_b, sim_config,
    n_runs=30,
    paired=True          # Use same turbulence for both mirrors
)
```

## Performance Metrics

The framework calculates the following metrics:

| Metric | Description |
|--------|-------------|
| `rms_error` | Root-mean-square tracking error |
| `max_error` | Maximum absolute error |
| `std_error` | Standard deviation of error |
| `mean_abs_error` | Mean absolute error |
| `settling_time` | Time to settle within threshold |
| `rejection_ratio` | Disturbance rejection ratio |

## Statistical Tests

The framework automatically selects appropriate tests:

1. **Normality Test** (Shapiro-Wilk): Determines if data is normally distributed
2. **Variance Test** (Levene's): Checks for equal variances
3. **Hypothesis Test**:
   - Student's t-test (normal, equal variances)
   - Welch's t-test (normal, unequal variances)
   - Mann-Whitney U (non-normal)
4. **Effect Size** (Cohen's d): Measures practical significance

## Output Files

When using `--report`, the following files are generated:

| File | Description |
|------|-------------|
| `comparison_*.png` | Comprehensive visualization with all metrics |
| `summary_*.txt` | Text summary table |
| `report_*.md` | Full Markdown report with statistics |
| `results_*.json` | Machine-readable JSON results |

## Key Features

### Auto-Tuned PID Gains

PID gains are automatically adjusted based on Q factor:
- Higher Q → Lower gains (more conservative)
- Lower Q → Higher gains (more aggressive)

### Paired Comparisons

Use `paired=True` to run both mirrors with the same turbulence realizations:
- Reduces variability from random turbulence
- Enables paired t-test for increased statistical power

### Von Kármán Turbulence

The simulation uses physics-based atmospheric turbulence modeling:
- Inner scale frequency (f₀): 10 Hz
- Corner frequency (f_corner): Configurable (50-200 Hz)
- Proper spectral roll-off characteristics

## Command-Line Options

### Mirror Configuration
- `--mirror-A`, `--mirror-B`: Mirror names
- `--Q1`, `--Q2`: Quality factors
- `--f-res`: Resonant frequency (default: 437 Hz)
- `--kp1`, `--ki1`, `--kd1`: PID gains for mirror A
- `--kp2`, `--ki2`, `--kd2`: PID gains for mirror B

### Simulation Parameters
- `--runs`: Number of simulation runs (default: 30)
- `--dt`: Sampling time (default: 0.0001 s)
- `--duration`: Simulation duration (default: 0.5 s)
- `--f0`: Inner scale frequency (default: 10 Hz)
- `--f-corner`: Corner frequency (default: 100 Hz)
- `--scaling`: Turbulence amplitude (default: 8 mrad)
- `--seed`: Random seed (default: 42)

### Sweep Options
- `--sweep-Q`: Perform Q factor sweep
- `--Q-range`: Range as "min,max"
- `--step`: Step size for sweep
- `--turbulence-sweep`: Comma-separated corner frequencies

### Output Options
- `--report`: Generate detailed report with plots
- `--output-dir`: Output directory
- `--save-json`: Save results to JSON
- `--quiet`: Minimal output

## Interpretation Guide

### Statistical Significance (p-value)
- **p < 0.05**: Statistically significant difference
- **p >= 0.05**: No significant difference detected

### Effect Size (Cohen's d)
- **|d| < 0.2**: Negligible effect
- **0.2 <= |d| < 0.5**: Small effect
- **0.5 <= |d| < 0.8**: Medium effect
- **|d| >= 0.8**: Large effect

### Decision Criteria
1. **p < 0.05 AND |d| >= 0.8**: Strong evidence for difference
2. **p < 0.05 AND 0.2 <= |d| < 0.8**: Moderate evidence
3. **p >= 0.05 OR |d| < 0.2**: No practical difference

## Example Workflows

### Compare Your Mirror (Q=20) vs Reference (Q=10)

```bash
python run_mirror_comparison.py \
    --mirror-A "Your_Mirror_Q20" --Q1 20 \
    --mirror-B "Reference_Q10" --Q2 10 \
    --runs 50 --report --output-dir ./comparison
```

### Find Optimal Q Factor

```bash
python run_mirror_comparison.py \
    --sweep-Q --Q-range 5,30 --step 2.5 \
    --runs 30 --report --output-dir ./sweep
```

### Test Robustness Under Strong Turbulence

```bash
python run_mirror_comparison.py \
    --Q1 20 --Q2 21 \
    --turbulence-sweep "50,100,150,200" \
    --runs 30 --report
```

## References

Based on methodology from:
- ISO 10110: Optical elements and systems
- ISO 14999: Interferometric measurement
- NIST Technical Note 1297: Guidelines for uncertainty measurement
- Montgomery, D.C. (2017). Design and Analysis of Experiments
- Cohen, J. (1988). Statistical Power Analysis
