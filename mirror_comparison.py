"""
Mirror Comparison Framework for Atmospheric Correction Systems

This module implements the statistical methodology for comparing two mirror models
in the context of atmospheric correction systems, following the rigorous
methodology outlined in the comparative performance analysis plan.

Key Features:
- Statistical hypothesis testing (t-test, Welch's t-test, Mann-Whitney U)
- Effect size calculation (Cohen's d)
- Power analysis for sample size determination
- Outlier detection and normality testing
- Uncertainty propagation
"""

import numpy as np
from scipy import stats
from scipy.signal import lti, tf2ss
from scipy.special import ndtr
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import warnings


@dataclass
class MirrorConfig:
    """Configuration for a mirror model."""
    name: str
    Q: float                      # Quality factor
    f_res: float = 437.0          # Resonant frequency (Hz)
    # PID gains - will be tuned based on Q factor
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    # Notch filter Q (should match mirror Q)
    Q_notch: Optional[float] = None

    def __post_init__(self):
        """Auto-tune PID gains based on Q factor if not specified."""
        if self.Q_notch is None:
            self.Q_notch = self.Q

        if self.kp is None:
            # Conservative tuning: higher Q requires lower gains
            # Q=10: kp=1.2, ki=7.0, kd=0.00025
            # Q=20: kp=1.0, ki=5.0, kd=0.00015
            damping_factor = 10.0 / self.Q  # Normalized to Q=10
            self.kp = 1.2 * damping_factor
            self.ki = 7.0 * damping_factor
            self.kd = 0.00025 * damping_factor

    @property
    def damping_ratio(self) -> float:
        """Calculate damping ratio from Q factor: zeta = 1/(2Q)."""
        return 1.0 / (2.0 * self.Q)


@dataclass
class ActuatorConfig:
    """FSM command and rate limits in the simulation's mrad command units."""
    command_limit: Optional[float] = None
    rate_limit: Optional[float] = None
    anti_windup_gain: float = 0.0


@dataclass
class SimulationConfig:
    """Configuration for atmospheric correction simulation."""
    dt: float = 0.0001            # Sampling rate (10 kHz)
    duration: float = 0.5         # Simulation duration (seconds)
    # Turbulence parameters
    f0: float = 10.0              # Inner scale frequency (Hz)
    f_corner: float = 100.0       # Corner frequency (Hz)
    scaling: float = 8.0          # Amplitude scaling (mrad)
    seed: Optional[int] = None    # Random seed for reproducibility
    use_qpd: bool = False         # Preserve the ideal-feedback baseline by default
    qpd: 'QPDConfig' = field(default_factory=lambda: QPDConfig())
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    measurement_delay_steps: int = 0
    command_delay_steps: int = 0

    @property
    def n_steps(self) -> int:
        if self.measurement_delay_steps < 0 or self.command_delay_steps < 0:
            raise ValueError('Delay steps must be non-negative')
        return int(self.duration / self.dt)


@dataclass
class QPDConfig:
    """Parameterized one-dimensional QPD centroid model.

    Position units are mrad. Optical power is normalized so measured power
    can be introduced later without changing the controller interface.
    """
    active_half_width_mrad: float = 10.0
    spot_sigma_mrad: float = 8.0
    received_power: float = 1.0
    read_noise_rms: float = 0.0
    shot_noise_scale: float = 0.0
    bandwidth_hz: Optional[float] = None
    imbalance: float = 0.0
    dropout_power: float = 0.0


class QPDModel:
    """Convert residual beam position into a noisy QPD centroid estimate."""

    def __init__(self, config: QPDConfig, dt: float, rng: np.random.Generator):
        if config.active_half_width_mrad <= 0 or config.spot_sigma_mrad <= 0:
            raise ValueError('QPD dimensions must be positive')
        if config.received_power < 0 or config.dropout_power < 0:
            raise ValueError('QPD power values must be non-negative')
        if not 0 <= config.imbalance < 1:
            raise ValueError('QPD imbalance must be in [0, 1)')
        if config.bandwidth_hz is not None and config.bandwidth_hz <= 0:
            raise ValueError('QPD bandwidth must be positive')

        self.config = config
        self.dt = dt
        self.rng = rng
        self.filtered_position = 0.0
        edge_probability = ndtr(config.active_half_width_mrad / config.spot_sigma_mrad)
        total_at_zero = 2.0 * edge_probability - 1.0
        gaussian_zero = 1.0 / np.sqrt(2.0 * np.pi)
        gaussian_edge = np.exp(-0.5 * (config.active_half_width_mrad / config.spot_sigma_mrad) ** 2) * gaussian_zero
        normalized_slope = (
            2.0 * (gaussian_zero - gaussian_edge)
            / config.spot_sigma_mrad
            / total_at_zero
        )
        self.centroid_scale = 1.0 / normalized_slope
        self.alpha = None
        if config.bandwidth_hz is not None:
            self.alpha = 1.0 - np.exp(-2.0 * np.pi * config.bandwidth_hz * dt)

    def measure(self, residual_mrad: float) -> Tuple[float, bool]:
        """Return measured centroid error and whether the QPD has signal."""
        config = self.config
        if config.received_power <= config.dropout_power:
            return 0.0, False

        clipped = float(np.clip(
            residual_mrad,
            -config.active_half_width_mrad,
            config.active_half_width_mrad,
        ))
        half_width = config.active_half_width_mrad
        sigma = config.spot_sigma_mrad

        # Integrate a Gaussian spot over left/right halves of the active area.
        left = ndtr((-0.0 - clipped) / sigma) - ndtr((-half_width - clipped) / sigma)
        right = ndtr((half_width - clipped) / sigma) - ndtr((-0.0 - clipped) / sigma)
        if config.imbalance:
            left *= 1.0 + config.imbalance
            right *= 1.0 - config.imbalance

        total = left + right
        normalized_error = (right - left) / total if total > 1e-12 else 0.0
        measured = normalized_error * self.centroid_scale

        noise_rms = config.read_noise_rms
        if config.shot_noise_scale > 0:
            noise_rms += config.shot_noise_scale / np.sqrt(config.received_power)
        if noise_rms > 0:
            measured += self.rng.normal(0.0, noise_rms)

        measured = float(np.clip(measured, -half_width, half_width))
        if self.alpha is not None:
            self.filtered_position += self.alpha * (measured - self.filtered_position)
            measured = self.filtered_position

        return measured, True


@dataclass
class MetricResult:
    """Results for a single performance metric."""
    name: str
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1))

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def cv(self) -> float:
        """Coefficient of variation (%)."""
        return 100.0 * self.std / self.mean if self.mean != 0 else 0.0

    @property
    def skewness(self) -> float:
        return float(stats.skew(self.values))

    @property
    def kurtosis(self) -> float:
        return float(stats.kurtosis(self.values))

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        n = len(self.values)
        se = self.std / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * se
        return (self.mean - margin, self.mean + margin)


@dataclass
class StatisticalTestResult:
    """Results from statistical hypothesis testing."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None


@dataclass
class MirrorComparisonResult:
    """Complete comparison results between two mirrors."""
    mirror_a_name: str
    mirror_b_name: str
    metrics_a: Dict[str, MetricResult]
    metrics_b: Dict[str, MetricResult]
    statistical_tests: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    recommendation: Optional[str] = None


def generate_von_karman_turbulence(config: SimulationConfig) -> np.ndarray:
    """
    Generate atmospheric turbulence using von Karman spectrum.

    The von Karman model provides a physics-based representation of
    atmospheric turbulence with proper spectral characteristics.

    Args:
        config: Simulation configuration

    Returns:
        Array of turbulence values in mrad
    """
    rng = np.random.default_rng(config.seed)

    n_steps = config.n_steps
    dt = config.dt

    # Frequency axis for FFT
    freqs = np.fft.fftfreq(n_steps, dt)
    freqs[0] = 1e-10  # Avoid division by zero at DC

    # von Karman power spectrum: P(f) = 1 / (f^2 + f0^2)^(11/6)
    # With high-frequency roll-off from corner frequency
    spectrum_magnitude = 1.0 / (
        (freqs**2 + config.f0**2)**(11.0/12.0) *
        (1 + (np.abs(freqs)/config.f_corner)**2)**0.5
    )

    # Generate complex Gaussian noise in frequency domain
    white_noise_complex = (
        rng.normal(0, 1, n_steps) +
        1j * rng.normal(0, 1, n_steps)
    ) / np.sqrt(2)

    # Apply spectrum and transform to time domain
    atm_fft = white_noise_complex * spectrum_magnitude
    atm_disturbance = np.fft.ifft(atm_fft).real

    # Normalize and scale
    atm_disturbance = atm_disturbance / np.std(atm_disturbance) * config.scaling

    return atm_disturbance


def create_notch_filter(f_res: float, Q_notch: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a discrete-time notch filter.

    The notch filter suppresses the resonant frequency to prevent
    excitation of the mirror's natural resonance.

    Args:
        f_res: Resonant frequency (Hz)
        Q_notch: Notch filter Q factor (should match mirror Q)
        dt: Sampling time (s)

    Returns:
        Filter coefficients (b, a)
    """
    w_notch = 2 * np.pi * f_res

    # Continuous notch: H(s) = (s^2 + w_n^2) / (s^2 + (w_n/Q)*s + w_n^2)
    num_notch = [1, 0, w_notch**2]
    den_notch = [1, w_notch / Q_notch, w_notch**2]

    notch_cont = lti(num_notch, den_notch)
    notch_disc = notch_cont.to_discrete(dt, method='bilinear')

    return np.array(notch_disc.num), np.array(notch_disc.den)


def create_plant_model(f_res: float, zeta: float, dt: float) -> Tuple:
    """
    Create state-space model of the FSM plant.

    Args:
        f_res: Resonant frequency (Hz)
        zeta: Damping ratio
        dt: Sampling time (s)

    Returns:
        State-space matrices (A, B, C, D)
    """
    w_res = 2 * np.pi * f_res

    num_p = [w_res**2]
    den_p = [1, 2 * zeta * w_res, w_res**2]

    plant_cont = lti(num_p, den_p)
    plant_disc = plant_cont.to_discrete(dt, method='bilinear')

    A, B, C, D = tf2ss(plant_disc.num, plant_disc.den)
    return A, B, C, D


def run_single_simulation(
    mirror: MirrorConfig,
    config: SimulationConfig,
    atm_disturbance: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Run a single atmospheric correction simulation.

    Args:
        mirror: Mirror configuration
        config: Simulation configuration
        atm_disturbance: Pre-generated turbulence (optional)

    Returns:
        Dictionary of performance metrics
    """
    dt = config.dt
    n_steps = config.n_steps

    # Generate turbulence if not provided
    if atm_disturbance is None:
        atm_disturbance = generate_von_karman_turbulence(config)

    # Create filters and plant
    b_n, a_n = create_notch_filter(mirror.f_res, mirror.Q_notch, dt)
    A_p, B_p, C_p, D_p = create_plant_model(mirror.f_res, mirror.damping_ratio, dt)

    # Initialize states
    plant_state = np.zeros((A_p.shape[0], 1))
    notch_in = [0.0, 0.0, 0.0]
    notch_out = [0.0, 0.0, 0.0]
    integral = 0.0
    prev_error = 0.0

    mirror_pos = np.zeros(n_steps)
    residual_error = np.zeros(n_steps)
    measured_error = np.zeros(n_steps)
    qpd_dropouts = 0
    applied_commands = np.zeros(n_steps)
    saturated_commands = 0
    current_mirror_pos = 0.0
    measurement_history = [0.0] * config.measurement_delay_steps
    command_history = [0.0] * (config.command_delay_steps + 1)
    previous_command = 0.0
    qpd = None
    last_valid_measurement = 0.0
    if config.use_qpd:
        qpd_seed = config.seed if config.seed is not None else 0
        qpd = QPDModel(config.qpd, dt, np.random.default_rng(qpd_seed))

    # Simulation loop
    for k in range(n_steps):
        true_error = atm_disturbance[k] - current_mirror_pos
        measurement_history.append(true_error)
        delayed_error = measurement_history[-(config.measurement_delay_steps + 1)]
        if qpd is None:
            error = delayed_error
        else:
            error, signal_valid = qpd.measure(delayed_error)
            if signal_valid:
                last_valid_measurement = error
            else:
                qpd_dropouts += 1
                error = last_valid_measurement
        measured_error[k] = error

        # PID calculation
        integral += error * dt
        derivative = (error - prev_error) / dt
        pid_output = (mirror.kp * error) + (mirror.ki * integral) + (mirror.kd * derivative)
        prev_error = error

        # Notch filter
        notch_in = [pid_output, notch_in[0], notch_in[1]]
        filtered_cmd = (
            b_n[0]*notch_in[0] + b_n[1]*notch_in[1] + b_n[2]*notch_in[2]
            - a_n[1]*notch_out[0] - a_n[2]*notch_out[1]
        ) / a_n[0]
        notch_out = [filtered_cmd, notch_out[0], notch_out[1]]

        command = filtered_cmd
        limit = config.actuator.command_limit
        if limit is not None:
            command = float(np.clip(command, -limit, limit))
        rate_limit = config.actuator.rate_limit
        if rate_limit is not None:
            max_delta = rate_limit * dt
            command = float(np.clip(command, previous_command - max_delta, previous_command + max_delta))
        if command != filtered_cmd:
            saturated_commands += 1
            if config.actuator.anti_windup_gain > 0:
                integral += config.actuator.anti_windup_gain * (command - filtered_cmd) * dt
        previous_command = command
        command_history.append(command)
        applied_command = command_history.pop(0)
        applied_commands[k] = applied_command

        # Plant update
        next_state = np.dot(A_p, plant_state) + B_p * applied_command
        mirror_pos[k] = (np.dot(C_p, plant_state) + D_p * applied_command).item()
        plant_state = next_state
        current_mirror_pos = mirror_pos[k]

        # Calculate residual error
        residual_error[k] = atm_disturbance[k] - mirror_pos[k]

    # Calculate metrics
    metrics = {
        'rms_error': np.sqrt(np.mean(residual_error**2)),
        'max_error': np.max(np.abs(residual_error)),
        'std_error': np.std(residual_error),
        'mean_abs_error': np.mean(np.abs(residual_error)),
        'settling_time': calculate_settling_time(residual_error, dt),
        'rejection_ratio': np.std(atm_disturbance) / np.sqrt(np.mean(residual_error**2)),
        'measurement_error_rms': np.sqrt(np.mean((measured_error - residual_error)**2)),
        'qpd_dropout_fraction': qpd_dropouts / n_steps,
        'control_effort_rms': np.sqrt(np.mean(applied_commands**2)),
        'actuator_saturation_fraction': saturated_commands / n_steps,
    }

    return metrics


def calculate_settling_time(error: np.ndarray, dt: float, threshold: float = 0.05) -> float:
    """
    Calculate settling time based on error signal.

    Args:
        error: Error signal array
        dt: Sampling time
        threshold: Fraction of initial error for settling

    Returns:
        Settling time in seconds
    """
    final_error = np.mean(error[-100:])  # Average of last 100 samples
    threshold_value = threshold * np.max(np.abs(error))

    # Find when error stays within threshold
    settled = np.where(np.abs(error - final_error) < threshold_value)[0]

    if len(settled) > 0:
        # Find first index where it stays settled for at least 100 samples
        for i in settled:
            if i + 100 < len(error) and np.all(np.abs(error[i:i+100] - final_error) < threshold_value):
                return i * dt

    return len(error) * dt  # Never settled


def detect_outliers(data: np.ndarray, method: str = 'iqr') -> np.ndarray:
    """
    Detect outliers using IQR or Grubbs' test.

    Args:
        data: Input data array
        method: 'iqr' or 'grubbs'

    Returns:
        Boolean array (True = outlier)
    """
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (data < lower) | (data > upper)

    elif method == 'grubbs':
        # Grubbs' test for outliers
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        G = np.max(np.abs(data - mean)) / std
        # Critical value approximation for alpha=0.05
        t_crit = stats.t.ppf(0.025 / n, n - 2)
        G_crit = (n - 1) * np.sqrt(t_crit**2 / (n * (n - 2 + t_crit**2)))
        return np.abs(data - mean) / std > G_crit

    return np.zeros(len(data), dtype=bool)


def test_normality(data: np.ndarray) -> Tuple[bool, float]:
    """
    Test if data follows normal distribution using Shapiro-Wilk test.

    Args:
        data: Input data array

    Returns:
        (is_normal, p_value)
    """
    if len(data) < 3:
        return False, 0.0

    # Shapiro-Wilk test (for n < 5000)
    if len(data) <= 5000:
        stat, p_value = stats.shapiro(data)
    else:
        # Use D'Agostino's normality test for larger samples
        stat, p_value = stats.normaltest(data)

    return p_value > 0.05, p_value


def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        data1: First sample
        data2: Second sample

    Returns:
        Cohen's d value
    """
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (np.mean(data1) - np.mean(data2)) / pooled_std
    return float(d)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_statistical_tests(
    data_a: np.ndarray,
    data_b: np.ndarray,
    metric_name: str
) -> List[StatisticalTestResult]:
    """
    Perform comprehensive statistical tests comparing two samples.

    Args:
        data_a: Sample from mirror A
        data_b: Sample from mirror B
        metric_name: Name of the metric being compared

    Returns:
        List of test results
    """
    results = []

    # Remove outliers
    outliers_a = detect_outliers(data_a, method='iqr')
    outliers_b = detect_outliers(data_b, method='iqr')
    clean_a = data_a[~outliers_a]
    clean_b = data_b[~outliers_b]

    # Test normality
    is_normal_a, p_norm_a = test_normality(clean_a)
    is_normal_b, p_norm_b = test_normality(clean_b)
    both_normal = is_normal_a and is_normal_b

    # Test for equal variances (Levene's test)
    _, p_levene = stats.levene(clean_a, clean_b)
    equal_variances = p_levene > 0.05

    # Calculate effect size
    d = cohens_d(clean_a, clean_b)
    effect_interp = interpret_effect_size(d)

    if both_normal:
        if equal_variances:
            # Student's t-test (equal variances)
            stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=True)
            test_name = "Student's t-test"
        else:
            # Welch's t-test (unequal variances)
            stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=False)
            test_name = "Welch's t-test"

        results.append(StatisticalTestResult(
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p_val),
            significant=p_val < 0.05,
            effect_size=d,
            effect_interpretation=effect_interp
        ))
    else:
        # Mann-Whitney U test (non-parametric)
        stat, p_val = stats.mannwhitneyu(clean_a, clean_b, alternative='two-sided')
        results.append(StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=float(stat),
            p_value=float(p_val),
            significant=p_val < 0.05,
            effect_size=d,
            effect_interpretation=effect_interp
        ))

    # Paired t-test if samples are paired (same turbulence realizations)
    if len(clean_a) == len(clean_b):
        stat_paired, p_val_paired = stats.ttest_rel(data_a, data_b)
        results.append(StatisticalTestResult(
            test_name="Paired t-test",
            statistic=float(stat_paired),
            p_value=float(p_val_paired),
            significant=p_val_paired < 0.05,
            effect_size=d,
            effect_interpretation=effect_interp
        ))

    return results


def calculate_sample_size(
    expected_effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size for given effect size and power.

    Args:
        expected_effect_size: Expected Cohen's d
        alpha: Significance level
        power: Statistical power (1 - beta)

    Returns:
        Required sample size per group
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = 2 * (z_alpha + z_beta)**2 / expected_effect_size**2
    return int(np.ceil(n))


def compare_mirrors(
    mirror_a: MirrorConfig,
    mirror_b: MirrorConfig,
    sim_config: SimulationConfig,
    n_runs: int = 30,
    paired: bool = True
) -> MirrorComparisonResult:
    """
    Perform comprehensive comparison between two mirror models.

    Args:
        mirror_a: First mirror configuration
        mirror_b: Second mirror configuration
        sim_config: Simulation configuration
        n_runs: Number of simulation runs per mirror
        paired: Whether to use same turbulence realizations for both mirrors

    Returns:
        Complete comparison results
    """
    print(f"Comparing mirrors: {mirror_a.name} vs {mirror_b.name}")
    print(f"Running {n_runs} simulations per mirror...")

    # Storage for metrics
    metrics_names = ['rms_error', 'max_error', 'std_error', 'mean_abs_error', 'settling_time', 'rejection_ratio']
    metrics_a = {name: [] for name in metrics_names}
    metrics_b = {name: [] for name in metrics_names}

    # Run simulations
    for i in range(n_runs):
        if paired:
            # Use same turbulence for both mirrors (paired comparison)
            seed = sim_config.seed + i if sim_config.seed is not None else i
            atm = generate_von_karman_turbulence(SimulationConfig(
                dt=sim_config.dt,
                duration=sim_config.duration,
                f0=sim_config.f0,
                f_corner=sim_config.f_corner,
                scaling=sim_config.scaling,
                seed=seed
            ))
            result_a = run_single_simulation(mirror_a, sim_config, atm)
            result_b = run_single_simulation(mirror_b, sim_config, atm)
        else:
            # Independent turbulence realizations
            result_a = run_single_simulation(mirror_a, sim_config)
            result_b = run_single_simulation(mirror_b, sim_config)

        for name in metrics_names:
            metrics_a[name].append(result_a[name])
            metrics_b[name].append(result_b[name])

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs")

    # Create MetricResult objects
    metric_results_a = {
        name: MetricResult(name, np.array(values))
        for name, values in metrics_a.items()
    }
    metric_results_b = {
        name: MetricResult(name, np.array(values))
        for name, values in metrics_b.items()
    }

    # Perform statistical tests
    statistical_tests = {}
    for name in metrics_names:
        tests = perform_statistical_tests(
            np.array(metrics_a[name]),
            np.array(metrics_b[name]),
            name
        )
        for test in tests:
            statistical_tests[f"{name}_{test.test_name}"] = test

    # Generate recommendation
    recommendation = generate_recommendation(
        mirror_a, mirror_b,
        metric_results_a, metric_results_b,
        statistical_tests
    )

    return MirrorComparisonResult(
        mirror_a_name=mirror_a.name,
        mirror_b_name=mirror_b.name,
        metrics_a=metric_results_a,
        metrics_b=metric_results_b,
        statistical_tests=statistical_tests,
        recommendation=recommendation
    )


def generate_recommendation(
    mirror_a: MirrorConfig,
    mirror_b: MirrorConfig,
    metrics_a: Dict[str, MetricResult],
    metrics_b: Dict[str, MetricResult],
    tests: Dict[str, StatisticalTestResult]
) -> str:
    """
    Generate recommendation based on statistical comparison.

    Args:
        mirror_a: Mirror A configuration
        mirror_b: Mirror B configuration
        metrics_a: Metrics for mirror A
        metrics_b: Metrics for mirror B
        tests: Statistical test results

    Returns:
        Recommendation string
    """
    # Count wins for each mirror
    a_wins = 0
    b_wins = 0
    significant_differences = 0

    # Check RMS error (primary metric)
    rms_a = metrics_a['rms_error'].mean
    rms_b = metrics_b['rms_error'].mean

    for test_name, test in tests.items():
        if 'rms_error' in test_name and test.significant:
            significant_differences += 1
            if rms_a < rms_b:
                a_wins += 1
            else:
                b_wins += 1

    # Generate recommendation
    if a_wins > b_wins and significant_differences > 0:
        return f"Select {mirror_a.name}: Significantly better RMS error performance (p<0.05)"
    elif b_wins > a_wins and significant_differences > 0:
        return f"Select {mirror_b.name}: Significantly better RMS error performance (p<0.05)"
    elif significant_differences == 0:
        return "No significant difference detected: Consider cost and other factors"
    else:
        return "Marginal difference: Requires engineering judgment"


def print_comparison_report(result: MirrorComparisonResult):
    """Print formatted comparison report."""
    print("\n" + "="*80)
    print(f"MIRROR COMPARISON REPORT: {result.mirror_a_name} vs {result.mirror_b_name}")
    print("="*80)

    # Print metrics table
    print("\n## Performance Metrics Summary\n")
    print(f"{'Metric':<20} {'Mirror A (mean±SD)':<25} {'Mirror B (mean±SD)':<25} {'Winner'}")
    print("-" * 95)

    for metric_name in result.metrics_a.keys():
        ma = result.metrics_a[metric_name]
        mb = result.metrics_b[metric_name]

        # Lower is better for most metrics
        winner = result.mirror_a_name if ma.mean < mb.mean else result.mirror_b_name
        if metric_name == 'rejection_ratio':
            winner = result.mirror_a_name if ma.mean > mb.mean else result.mirror_b_name

        print(f"{metric_name:<20} {ma.mean:.4f}±{ma.std:.4f}          {mb.mean:.4f}±{mb.std:.4f}          {winner}")

    # Print statistical tests
    print("\n## Statistical Tests\n")
    print(f"{'Test':<40} {'Statistic':<12} {'p-value':<12} {'Significant':<12} {'Effect Size'}")
    print("-" * 95)

    for test_name, test in result.statistical_tests.items():
        sig = "Yes" if test.significant else "No"
        effect = f"{test.effect_size:.3f} ({test.effect_interpretation})" if test.effect_size else "N/A"
        print(f"{test_name:<40} {test.statistic:<12.4f} {test.p_value:<12.6f} {sig:<12} {effect}")

    # Print recommendation
    print("\n## Recommendation\n")
    print(result.recommendation)
    print("\n" + "="*80)


# Convenience functions for common comparisons
def compare_q_factors(
    Q1: float,
    Q2: float,
    name1: str = "Mirror A",
    name2: str = "Mirror B",
    n_runs: int = 30
) -> MirrorComparisonResult:
    """
    Compare two mirrors with different Q factors.

    Args:
        Q1: Quality factor for mirror 1
        Q2: Quality factor for mirror 2
        name1: Name for mirror 1
        name2: Name for mirror 2
        n_runs: Number of simulation runs

    Returns:
        Comparison results
    """
    mirror_a = MirrorConfig(name=name1, Q=Q1)
    mirror_b = MirrorConfig(name=name2, Q=Q2)
    sim_config = SimulationConfig(seed=42)

    return compare_mirrors(mirror_a, mirror_b, sim_config, n_runs=n_runs)


def compare_turbulence_conditions(
    mirror: MirrorConfig,
    f_corner_values: List[float],
    n_runs: int = 30
) -> Dict[str, MirrorComparisonResult]:
    """
    Compare mirror performance under different turbulence conditions.

    Args:
        mirror: Mirror configuration
        f_corner_values: List of corner frequencies to test
        n_runs: Number of runs per condition

    Returns:
        Dictionary of comparison results
    """
    results = {}
    base_config = SimulationConfig(seed=42, f_corner=f_corner_values[0])

    for i, f_corner in enumerate(f_corner_values[1:], 1):
        test_config = SimulationConfig(seed=42, f_corner=f_corner)

        # Create virtual mirrors for each condition
        mirror_base = MirrorConfig(name=f"Turbulence_{f_corner_values[0]}Hz", Q=mirror.Q)
        mirror_test = MirrorConfig(name=f"Turbulence_{f_corner}Hz", Q=mirror.Q)

        result = compare_mirrors(mirror_base, mirror_test, base_config, n_runs=n_runs)
        results[f"{f_corner_values[0]}Hz_vs_{f_corner}Hz"] = result

    return results


if __name__ == "__main__":
    # Example usage
    print("Mirror Comparison Framework")
    print("=" * 50)

    # Example: Compare Q=10 vs Q=20 mirrors
    result = compare_q_factors(Q1=10.0, Q2=20.0, n_runs=10)
    print_comparison_report(result)
