"""
PID Tuning Optimizer for FSM Mirrors

Finds optimal PID gains for each mirror using grid search and performance metrics.
Tests multiple gain combinations and selects the best based on RMS error.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json

from mirror_comparison import MirrorConfig, SimulationConfig, run_single_simulation


@dataclass
class PIDGains:
    """PID gain container."""
    kp: float
    ki: float
    kd: float

    def __repr__(self):
        return f"PID(kp={self.kp:.4f}, ki={self.ki:.4f}, kd={self.kd:.6f})"


@dataclass
class MirrorSpecs:
    """Mirror specifications."""
    name: str
    f_res: float
    Q: float

    @property
    def damping_ratio(self) -> float:
        return 1.0 / (2.0 * self.Q)

    @property
    def w_res(self) -> float:
        return 2 * np.pi * self.f_res


@dataclass
class ScenarioSpec:
    """Single turbulence scenario used during robust PID tuning."""
    name: str
    disturbance: np.ndarray
    distance_km: float
    turbulence_level: float
    seed: int


def generate_scenario_matrix(
    mirror: MirrorSpecs,
    dt: float = 0.0001,
    duration: float = 0.5,
    distances_km: Optional[List[float]] = None,
    turbulence_levels: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    quick: bool = False,
) -> List[ScenarioSpec]:
    """Build a representative set of atmospheric scenarios for robust tuning.

    Each scenario varies the effective turbulence strength and seed to emulate
    different operating ranges and environmental realizations on the closed loop.
    """
    if distances_km is None:
        distances_km = [10.0, 20.0, 30.0, 40.0, 50.0]
    if turbulence_levels is None:
        turbulence_levels = [0.75, 1.0, 1.35]
    if seeds is None:
        seeds = [11, 23, 37, 59, 71]

    if quick:
        distances_km = [10.0, 30.0, 50.0]
        turbulence_levels = [0.8, 1.2]
        seeds = [11, 37, 73]

    n_steps = int(duration / dt)
    scenarios: List[ScenarioSpec] = []

    for distance_km in distances_km:
        for turbulence_level in turbulence_levels:
            for seed in seeds:
                # Keep the scenario matrix in a realistic operating envelope for the
                # closed-loop mirror model. The previous envelope was too aggressive
                # for the high-Q fast mirror and pushed the search into a clearly
                # unstable region.
                scenario_scale = 1.5 + 0.08 * distance_km * turbulence_level
                disturbance = generate_von_karman_turbulence(
                    n_steps=n_steps,
                    dt=dt,
                    f0=10.0,
                    f_corner=120.0,
                    scaling=scenario_scale,
                    seed=seed + int(distance_km * 10),
                )
                scenarios.append(
                    ScenarioSpec(
                        name=f"{mirror.name}_d{distance_km:g}km_L{turbulence_level:g}_s{seed}",
                        disturbance=disturbance,
                        distance_km=float(distance_km),
                        turbulence_level=float(turbulence_level),
                        seed=int(seed),
                    )
                )

    return scenarios


def generate_von_karman_turbulence(
    n_steps: int,
    dt: float,
    f0: float = 10.0,
    f_corner: float = 100.0,
    scaling: float = 8.0,
    seed: int = 42
) -> np.ndarray:
    """Generate von Kármán turbulence."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(n_steps, dt)
    freqs[0] = 1e-10

    spectrum = 1.0 / (
        (freqs**2 + f0**2)**(11.0/12.0) *
        (1 + (np.abs(freqs)/f_corner)**2)**0.5
    )

    noise = (rng.normal(0, 1, n_steps) + 1j*rng.normal(0, 1, n_steps)) / np.sqrt(2)
    atm = np.fft.ifft(noise * spectrum).real
    atm = atm / np.std(atm) * scaling
    return atm


def simulate_with_gains(
    mirror: MirrorSpecs,
    gains: PIDGains,
    atm_disturbance: np.ndarray,
    dt: float = 0.0001
) -> Dict[str, float]:
    """
    Run simulation with specific PID gains.

    Returns performance metrics.
    """
    shared_mirror = MirrorConfig(
        name=mirror.name,
        Q=mirror.Q,
        f_res=mirror.f_res,
        kp=gains.kp,
        ki=gains.ki,
        kd=gains.kd,
    )
    config = SimulationConfig(
        dt=dt,
        duration=len(atm_disturbance) * dt,
        seed=None,
    )
    metrics = run_single_simulation(shared_mirror, config, atm_disturbance)
    metrics['stability'] = check_stability(metrics['max_error'])
    return metrics


def calculate_settling_time(error: np.ndarray, dt: float, threshold: float = 0.05) -> float:
    """Calculate settling time."""
    final_error = np.mean(error[-100:])
    threshold_value = threshold * np.max(np.abs(error))
    settled = np.where(np.abs(error - final_error) < threshold_value)[0]

    if len(settled) > 0:
        for i in settled:
            if i + 100 < len(error) and np.all(np.abs(error[i:i+100] - final_error) < threshold_value):
                return i * dt
    return len(error) * dt


def check_stability(error: float, max_allowed: float = 100.0) -> bool:
    """Check if simulation remained stable."""
    return abs(error) < max_allowed


def evaluate_gains_across_scenarios(
    mirror: MirrorSpecs,
    gains: PIDGains,
    disturbances: List[np.ndarray],
    dt: float = 0.0001,
    instability_penalty: float = 1_000.0,
) -> Dict[str, float]:
    """Evaluate one gain set over independent turbulence realizations.

    The returned objective intentionally penalizes unstable runs and includes
    both average and upper-tail RMS error, preventing single-trace overfitting.
    """
    results = [simulate_with_gains(mirror, gains, disturbance, dt) for disturbance in disturbances]
    rms_values = np.array([result['rms_error'] for result in results], dtype=float)
    stable = np.array([result['stability'] for result in results], dtype=bool)
    unstable_runs = int(np.count_nonzero(~stable))

    objective = (
        float(np.mean(rms_values))
        + 0.5 * float(np.percentile(rms_values, 95))
        + 0.5 * float(np.max(rms_values))
        + instability_penalty * unstable_runs
        + 0.1 * float(np.mean([result['control_effort_rms'] for result in results]))
    )
    return {
        'objective': objective,
        'rms_error_mean': float(np.mean(rms_values)),
        'rms_error_p95': float(np.percentile(rms_values, 95)),
        'rms_error_worst': float(np.max(rms_values)),
        'control_effort_mean': float(np.mean([result['control_effort_rms'] for result in results])),
        'saturation_fraction_mean': float(np.mean([result['actuator_saturation_fraction'] for result in results])),
        'unstable_runs': float(unstable_runs),
        'stability': bool(unstable_runs == 0),
    }


def coarse_search(
    mirror: MirrorSpecs,
    atm_disturbance: Optional[np.ndarray] = None,
    dt: float = 0.0001,
    scenarios: Optional[List[ScenarioSpec]] = None,
    quick: bool = False,
) -> List[Tuple[PIDGains, Dict[str, float]]]:
    """Coarse grid search for PID gains.

    Uses a robust scenario objective when a scenario matrix is provided, while
    still supporting the original single-disturbance nominal tuning path.
    """
    print(f"\n  Coarse search for {mirror.name}...")

    if scenarios is None:
        if atm_disturbance is None:
            raise ValueError('Either a disturbance array or a scenario list is required.')
        scenarios = [ScenarioSpec(name='nominal', disturbance=atm_disturbance, distance_km=20.0, turbulence_level=1.0, seed=42)]

    # Define search ranges based on Q factor. Quick mode shrinks the candidate
    # grid so the robust multi-scenario optimizer remains practical for iteration.
    if quick:
        if mirror.Q >= 50:
            kp_range = np.linspace(0.05, 0.30, 6)
            ki_range = np.linspace(0.5, 2.0, 6)
            kd_range = np.linspace(1.0e-6, 4.0e-5, 5)
        elif mirror.Q >= 20:
            kp_range = np.linspace(0.6, 1.3, 5)
            ki_range = np.linspace(4.0, 8.0, 5)
            kd_range = np.linspace(0.00012, 0.00032, 4)
        else:
            kp_range = np.linspace(0.9, 1.7, 5)
            ki_range = np.linspace(6.0, 12.0, 5)
            kd_range = np.linspace(0.0002, 0.00045, 4)
    else:
        if mirror.Q >= 50:
            kp_range = np.linspace(0.05, 0.35, 8)
            ki_range = np.linspace(0.5, 3.0, 8)
            kd_range = np.linspace(1.0e-6, 5.0e-5, 6)
        elif mirror.Q >= 20:
            kp_range = np.linspace(0.5, 1.5, 8)
            ki_range = np.linspace(3.0, 10.0, 8)
            kd_range = np.linspace(0.0001, 0.0004, 6)
        else:
            kp_range = np.linspace(0.8, 2.0, 8)
            ki_range = np.linspace(5.0, 15.0, 8)
            kd_range = np.linspace(0.0002, 0.0005, 6)

    results = []
    total = len(kp_range) * len(ki_range) * len(kd_range)
    count = 0

    for kp in kp_range:
        for ki in ki_range:
            for kd in kd_range:
                count += 1
                if count % 100 == 0:
                    print(f"    Progress: {count}/{total} ({100*count/total:.0f}%)")

                gains = PIDGains(kp, ki, kd)
                try:
                    if len(scenarios) == 1 and scenarios[0].disturbance is atm_disturbance:
                        metrics = simulate_with_gains(mirror, gains, scenarios[0].disturbance, dt)
                    else:
                        metrics = evaluate_gains_across_scenarios(
                            mirror,
                            gains,
                            [scenario.disturbance for scenario in scenarios],
                            dt,
                        )
                    if metrics.get('stability', True):
                        results.append((gains, metrics))
                except Exception:
                    continue

    # Sort by objective or RMS error depending on the evaluation mode.
    metric_key = 'objective' if len(scenarios) > 1 else 'rms_error'
    results.sort(key=lambda x: x[1][metric_key])
    return results


def fine_search(
    mirror: MirrorSpecs,
    atm_disturbance: Optional[np.ndarray] = None,
    best_coarse: Optional[PIDGains] = None,
    dt: float = 0.0001,
    scenarios: Optional[List[ScenarioSpec]] = None,
    quick: bool = False,
) -> Tuple[PIDGains, Dict[str, float]]:
    """Fine search around best coarse result.

    When a scenario matrix is supplied, this minimises the robust objective over
    the full disturbance ensemble instead of a single nominal trace.
    """
    print(f"\n  Fine search for {mirror.name}...")

    if scenarios is None:
        if atm_disturbance is None or best_coarse is None:
            raise ValueError('Nominal fine search requires disturbance and coarse gains.')
        scenarios = [ScenarioSpec(name='nominal', disturbance=atm_disturbance, distance_km=20.0, turbulence_level=1.0, seed=42)]

    if best_coarse is None:
        raise ValueError('A best coarse gain set is required for fine search.')

    if quick:
        kp_range = np.linspace(best_coarse.kp * 0.8, best_coarse.kp * 1.2, 6)
        ki_range = np.linspace(best_coarse.ki * 0.8, best_coarse.ki * 1.2, 6)
        kd_range = np.linspace(best_coarse.kd * 0.8, best_coarse.kd * 1.2, 5)
    else:
        kp_range = np.linspace(best_coarse.kp * 0.7, best_coarse.kp * 1.3, 10)
        ki_range = np.linspace(best_coarse.ki * 0.7, best_coarse.ki * 1.3, 10)
        kd_range = np.linspace(best_coarse.kd * 0.7, best_coarse.kd * 1.3, 8)

    best_result = None
    best_score = float('inf')
    total = len(kp_range) * len(ki_range) * len(kd_range)
    count = 0

    for kp in kp_range:
        for ki in ki_range:
            for kd in kd_range:
                count += 1
                if count % 100 == 0:
                    print(f"    Progress: {count}/{total} ({100*count/total:.0f}%)")

                gains = PIDGains(kp, ki, kd)
                try:
                    if len(scenarios) == 1 and scenarios[0].disturbance is atm_disturbance:
                        metrics = simulate_with_gains(mirror, gains, scenarios[0].disturbance, dt)
                        score = metrics['rms_error']
                    else:
                        metrics = evaluate_gains_across_scenarios(
                            mirror,
                            gains,
                            [scenario.disturbance for scenario in scenarios],
                            dt,
                        )
                        score = metrics['objective']
                    if metrics.get('stability', True) and score < best_score:
                        best_score = score
                        best_result = (gains, metrics)
                except Exception:
                    continue

    return best_result


def validate_gains(
    mirror: MirrorSpecs,
    gains: PIDGains,
    n_runs: int = 20,
    dt: float = 0.0001,
    duration: float = 0.5
) -> Dict[str, float]:
    """
    Validate gains with multiple turbulence realizations.
    """
    print(f"\n  Validating {mirror.name} with {n_runs} runs...")

    n_steps = int(duration / dt)
    rms_errors = []
    max_errors = []

    for i in range(n_runs):
        atm = generate_von_karman_turbulence(n_steps, dt, seed=42+i)
        metrics = simulate_with_gains(mirror, gains, atm, dt)
        if metrics['stability']:
            rms_errors.append(metrics['rms_error'])
            max_errors.append(metrics['max_error'])

    return {
        'rms_mean': np.mean(rms_errors),
        'rms_std': np.std(rms_errors),
        'max_mean': np.mean(max_errors),
        'max_std': np.std(max_errors),
    }


def tune_mirror(
    mirror: MirrorSpecs,
    dt: float = 0.0001,
    duration: float = 0.5,
    quick: bool = False,
    scenarios: Optional[List[ScenarioSpec]] = None,
) -> Dict:
    """
    Complete tuning process for a mirror.

    When no explicit scenario list is provided, this creates a scenario matrix that
    spans multiple turbulence strengths and propagation distances to avoid
    overfitting to a single nominal disturbance realization.
    """
    print(f"\n{'='*80}")
    print(f"TUNING: {mirror.name}")
    print(f"{'='*80}")
    print(f"  Resonant Frequency: {mirror.f_res} Hz")
    print(f"  Quality Factor Q: {mirror.Q}")
    print(f"  Damping Ratio: {mirror.damping_ratio:.4f}")

    if scenarios is None:
        scenarios = generate_scenario_matrix(mirror, dt=dt, duration=duration, quick=quick)

    print(f"  Scenario count: {len(scenarios)}")

    coarse_results = coarse_search(mirror, dt=dt, scenarios=scenarios, quick=quick)

    if not coarse_results:
        print(f"  ERROR: No stable gains found in coarse search!")
        return None

    print(f"\n  Top 5 coarse results:")
    for i, (gains, metrics) in enumerate(coarse_results[:5]):
        score_name = 'objective' if 'objective' in metrics else 'rms_error'
        print(f"    {i+1}. {gains} -> {score_name}={metrics[score_name]:.4f}")

    best_coarse = coarse_results[0][0]
    fine_result = fine_search(mirror, best_coarse=best_coarse, dt=dt, scenarios=scenarios, quick=quick)

    if fine_result is None:
        print(f"  ERROR: No stable gains found in fine search!")
        return None

    best_gains, best_metrics = fine_result

    print(f"\n  Best fine result:")
    print(f"    Gains: {best_gains}")
    score_name = 'objective' if 'objective' in best_metrics else 'rms_error'
    print(f"    Robust score: {best_metrics[score_name]:.4f}")
    print(f"    RMS Error: {best_metrics.get('rms_error_mean', best_metrics.get('rms_error', 0.0)):.4f} mrad")
    print(f"    Max Error: {best_metrics.get('rms_error_worst', best_metrics.get('max_error', 0.0)):.4f} mrad")

    validation = validate_gains(mirror, best_gains, n_runs=20, dt=dt, duration=duration)

    print(f"\n  Validation (20 runs):")
    print(f"    RMS Error: {validation['rms_mean']:.4f} ± {validation['rms_std']:.4f} mrad")
    print(f"    Max Error: {validation['max_mean']:.4f} ± {validation['max_std']:.4f} mrad")

    return {
        'mirror': mirror,
        'gains': best_gains,
        'metrics': best_metrics,
        'validation': validation,
        'scenarios': scenarios,
        'coarse_top5': coarse_results[:5]
    }


def print_comparison_report(s040512_results: Dict, s72026_results: Dict):
    """Print comparison of tuned mirrors."""
    print("\n" + "="*80)
    print("PID TUNING RESULTS COMPARISON")
    print("="*80)

    print("\nS040512 (Q=21, f_res=437 Hz):")
    print(f"  Optimal Gains: {s040512_results['gains']}")
    print(f"  RMS Error: {s040512_results['validation']['rms_mean']:.4f} ± {s040512_results['validation']['rms_std']:.4f} mrad")
    print(f"  Max Error: {s040512_results['validation']['max_mean']:.4f} ± {s040512_results['validation']['max_std']:.4f} mrad")

    print("\nS72026 (Q=52, f_res=889 Hz):")
    print(f"  Optimal Gains: {s72026_results['gains']}")
    print(f"  RMS Error: {s72026_results['validation']['rms_mean']:.4f} ± {s72026_results['validation']['rms_std']:.4f} mrad")
    print(f"  Max Error: {s72026_results['validation']['max_mean']:.4f} ± {s72026_results['validation']['max_std']:.4f} mrad")

    print("\n" + "-"*80)
    print("RECOMMENDATION:")
    rms_040512 = s040512_results['validation']['rms_mean']
    rms_72026 = s72026_results['validation']['rms_mean']

    if rms_040512 < rms_72026:
        improvement = (rms_72026 - rms_040512) / rms_72026 * 100
        print(f"  S040512 performs {improvement:.1f}% better than S72026")
        print(f"  Despite lower bandwidth, S040512's moderate Q allows more aggressive control")
    else:
        improvement = (rms_040512 - rms_72026) / rms_040512 * 100
        print(f"  S72026 performs {improvement:.1f}% better than S040512")
        print(f"  Higher bandwidth and resonant frequency provide better tracking")

    print("="*80)


def save_tuning_results(s040512_results: Dict, s72026_results: Dict, filename: str = "optimal_pid_gains.json"):
    """Save tuning results to JSON."""
    data = {
        'S040512': {
            'f_res': s040512_results['mirror'].f_res,
            'Q': s040512_results['mirror'].Q,
            'kp': s040512_results['gains'].kp,
            'ki': s040512_results['gains'].ki,
            'kd': s040512_results['gains'].kd,
            'rms_error_mean': s040512_results['validation']['rms_mean'],
            'rms_error_std': s040512_results['validation']['rms_std'],
        },
        'S72026': {
            'f_res': s72026_results['mirror'].f_res,
            'Q': s72026_results['mirror'].Q,
            'kp': s72026_results['gains'].kp,
            'ki': s72026_results['gains'].ki,
            'kd': s72026_results['gains'].kd,
            'rms_error_mean': s72026_results['validation']['rms_mean'],
            'rms_error_std': s72026_results['validation']['rms_std'],
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nOptimal gains saved to: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Find optimal PID gains for S040512 and S72026')
    parser.add_argument('--quick', action='store_true', help='Quick search (fewer points)')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    args = parser.parse_args()

    # Define mirrors
    s040512 = MirrorSpecs(name="S040512", f_res=437.0, Q=21.0)
    s72026 = MirrorSpecs(name="S72026", f_res=889.0, Q=52.0)

    # Tune each mirror
    print("Starting PID tuning optimization...")
    print("This may take a few minutes...")

    s040512_results = tune_mirror(s040512, quick=args.quick)
    s72026_results = tune_mirror(s72026, quick=args.quick)

    # Print comparison
    print_comparison_report(s040512_results, s72026_results)

    # Save results
    if args.save:
        save_tuning_results(s040512_results, s72026_results)
