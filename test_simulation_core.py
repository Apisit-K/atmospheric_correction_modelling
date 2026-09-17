import numpy as np

from mirror_comparison import (
    ActuatorConfig,
    MirrorConfig,
    QPDConfig,
    QPDModel,
    SimulationConfig,
    generate_von_karman_turbulence,
    run_single_simulation,
)


def test_qpd_centroid_is_signed_and_calibrated_near_axis():
    sensor = QPDModel(QPDConfig(spot_sigma_mrad=8.0), 1e-4, np.random.default_rng(1))

    negative, negative_valid = sensor.measure(-2.0)
    zero, zero_valid = sensor.measure(0.0)
    positive, positive_valid = sensor.measure(2.0)

    assert negative_valid and zero_valid and positive_valid
    assert negative < zero < positive
    assert np.isclose(positive, 2.0, atol=0.1)
    assert np.isclose(negative, -2.0, atol=0.1)


def test_qpd_reports_signal_dropout():
    sensor = QPDModel(QPDConfig(received_power=0.0), 1e-4, np.random.default_rng(1))

    measured, valid = sensor.measure(2.0)

    assert measured == 0.0
    assert not valid


def test_turbulence_generation_is_reproducible_without_global_rng_changes():
    config = SimulationConfig(duration=0.01, seed=42)
    np.random.seed(123)
    before = np.random.random()

    first = generate_von_karman_turbulence(config)
    second = generate_von_karman_turbulence(config)
    after = np.random.random()

    np.random.seed(123)
    assert np.isclose(before, np.random.random())
    assert np.isclose(after, np.random.random())
    assert np.array_equal(first, second)


def test_closed_loop_simulation_returns_finite_metrics():
    mirror = MirrorConfig(name="test", Q=20.0)
    config = SimulationConfig(duration=0.02, seed=42, use_qpd=True)
    disturbance = generate_von_karman_turbulence(config)

    metrics = run_single_simulation(mirror, config, disturbance)

    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["qpd_dropout_fraction"] <= 1.0
    assert metrics["rms_error"] >= 0.0


def test_delayed_rate_limited_actuator_reports_saturation():
    mirror = MirrorConfig(name="test", Q=20.0)
    config = SimulationConfig(
        duration=0.02,
        seed=42,
        use_qpd=True,
        measurement_delay_steps=2,
        command_delay_steps=2,
        actuator=ActuatorConfig(
            command_limit=0.02,
            rate_limit=10.0,
            anti_windup_gain=1.0,
        ),
    )
    disturbance = generate_von_karman_turbulence(config)

    metrics = run_single_simulation(mirror, config, disturbance)

    assert metrics["actuator_saturation_fraction"] > 0.0
    assert metrics["control_effort_rms"] <= 0.02 + 1e-12
    assert all(np.isfinite(value) for value in metrics.values())
