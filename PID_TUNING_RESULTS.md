# PID Tuning Results: S040512 vs S72026

## Executive Summary

After systematic PID optimization, the **S72026 significantly outperforms** the S040512:

- **S72026 RMS Error: 0.87 ± 0.38 mrad** (53% better)
- **S040512 RMS Error: 1.87 ± 1.16 mrad**

## Optimal PID Gains

### S040512 (Q=21, f_res=437 Hz)
```python
kp = 0.9595
ki = 2.1000
kd = 0.000070
```

### S72026 (Q=52, f_res=889 Hz)
```python
kp = 0.9667
ki = 0.7000
kd = 0.000035
```

## Key Insights

### Why S72026 Wins
1. **Higher resonant frequency** (889 Hz vs 437 Hz) → 2x better bandwidth
2. **Higher Q** requires more conservative gains, but the higher bandwidth compensates
3. **Better disturbance rejection** (10.8x vs 5.7x)

### Gain Tuning Observations
- **S72026 needs lower integral gain** (0.7 vs 2.1) due to higher Q
- **Derivative gain is halved** for S72026 (0.000035 vs 0.000070)
- **Proportional gains are similar** (~0.96) - primarily determined by system dynamics

## Performance Comparison

| Metric | S040512 | S72026 | Improvement |
|--------|---------|--------|-------------|
| RMS Error | 1.87 mrad | 0.87 mrad | **53% better** |
| Max Error | 7.84 mrad | 7.10 mrad | 9% better |
| Std Error | 1.23 mrad | 0.74 mrad | **40% better** |
| Settling Time | 0.33 s | 0.23 s | **30% faster** |
| Rejection Ratio | 5.7x | 10.8x | **90% better** |

## Usage in Code

### Using Optimal Gains

```python
from mirror_comparison import MirrorConfig

# S040512 with optimal gains
s040512 = MirrorConfig(
    name="S040512",
    Q=21.0,
    f_res=437.0,
    kp=0.9595,
    ki=2.1000,
    kd=0.000070
)

# S72026 with optimal gains
s72026 = MirrorConfig(
    name="S72026",
    Q=52.0,
    f_res=889.0,
    kp=0.9667,
    ki=0.7000,
    kd=0.000035
)
```

### Running Comparison

```bash
# Compare with optimal gains
python compare_s040512_vs_s72026.py --runs 30

# Generate full report
python compare_s040512_vs_s72026.py --runs 50 --full-report
```

## Files Generated

| File | Description |
|------|-------------|
| `tune_pid_gains.py` | PID optimization script |
| `optimal_pid_gains.json` | Saved optimal gains |
| `compare_s040512_vs_s72026.py` | Comparison script using optimal gains |

## Recommendation

**Select S72026** for applications requiring:
- Best tracking performance
- Fast settling time
- High disturbance rejection

The S72026's higher bandwidth and resonant frequency, combined with properly tuned conservative PID gains, deliver superior performance despite its higher Q factor.
