import json
from pathlib import Path

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# --- 0. LOAD THE VERIFIED TUNED GAINS ---
results_path = Path(__file__).with_name("optimal_pid_gains.json")
if results_path.exists():
    with results_path.open("r") as fh:
        tuned_gains = json.load(fh)
    target = tuned_gains["S040512"]
    kp = float(target["kp"])
    ki = float(target["ki"])
    kd = float(target["kd"])
else:
    # Fallback to the last known-good values if the JSON has not been generated yet.
    kp, ki, kd = 0.93, 9.60, 0.000384

# --- 1. SYSTEM INITIALISATION ---
dt = 0.0001  # 10 kHz sampling rate
t = np.arange(0, 0.5, dt)
n_steps = len(t)

# --- 2. CONFIGURING SYSTEM FREQUENCIES ---
f_res = 437.0                       # FSM Resonant Frequency (Hz) from datasheet
w_res = 2 * np.pi * f_res           # Convert to rad/s

# Mirror Quality Factors (from your actual mirror measurements)
# Q = 20 (X axis) and Q = 21 (Y axis) means:
# - Higher Q = lower damping = sharper resonance peak
# - Original code used Q=10 (ζ=0.05), your mirror is ~2x more resonant
# - This requires more aggressive notch filtering or lower control gains
Q_x = 20.0                          # Quality factor for X axis
Q_y = 21.0                          # Quality factor for Y axis

# Convert Q to damping ratio: ζ = 1/(2Q)
zeta_fsm_x = 1.0 / (2.0 * Q_x)      # Damping ratio for X axis: 0.025
zeta_fsm_y = 1.0 / (2.0 * Q_y)      # Damping ratio for Y axis: ~0.0238

# Use X axis parameters for this simulation (or average them)
zeta_fsm = zeta_fsm_x               # Using X axis damping (Q=20)
# Alternative: zeta_fsm = (zeta_fsm_x + zeta_fsm_y) / 2  # Average both axes

# --- 3. CREATE THE NOTCH FILTER ---
f_notch = f_res                     # Center the notch exactly on the resonance
w_notch = 2 * np.pi * f_notch
# Notch Q should match or exceed the mirror Q for effective suppression
# Using Q=20 to match your mirror's X axis (or use higher Q for sharper notch)
Q_notch = 20.0                      # Match mirror Q_x for optimal suppression
zeta_notch = 1.0                    # Overdamped numerator to flatten the depth

# Continuous Notch: H(s) = (s^2 + w_n^2) / (s^2 + (w_n/Q)*s + w_n^2)
num_notch = [1, 0, w_notch**2]
den_notch = [1, w_notch / Q_notch, w_notch**2]
notch_cont = signal.lti(num_notch, den_notch)
notch_disc = notch_cont.to_discrete(dt, method='bilinear')

# Filter coefficients for difference equation: y[k] = b0*x[k] + b1*x[k-1]... - a1*y[k-1]...
b_n, a_n = notch_disc.num, notch_disc.den
notch_in = [0.0, 0.0, 0.0]
notch_out = [0.0, 0.0, 0.0]

# --- 4. CREATE THE FSM PLANT MODEL ---
num_p = [w_res**2]
den_p = [1, 2 * zeta_fsm * w_res, w_res**2]
plant_cont = signal.lti(num_p, den_p)
plant_disc = plant_cont.to_discrete(dt, method='bilinear')
A_p, B_p, C_p, D_p = signal.tf2ss(plant_disc.num, plant_disc.den)
plant_state = np.zeros((A_p.shape[0], 1))

# --- 5. ATMOSPHERIC DISTURBANCE ---
# Using von Karman spectrum for realistic atmospheric turbulence
# This models the Kolmogorov turbulence spectrum with proper roll-off
np.random.seed(12)

# von Karman turbulence parameters
f0 = 10.0                           # Inner scale frequency (Hz)
f_corner = 100.0                    # Corner frequency for high-freq roll-off (Hz)
scaling = 8.0                       # Overall amplitude scaling (mrad)

# Generate frequency axis for FFT
freqs = np.fft.fftfreq(n_steps, dt)
freqs[0] = 1e-10                    # Avoid division by zero at DC

# von Karman power spectrum: P(f) = 1 / (f^2 + f0^2)^(11/6)
spectrum_magnitude = 1.0 / ((freqs**2 + f0**2)**(11.0/12.0) * (1 + (np.abs(freqs)/f_corner)**2)**0.5)

# Generate complex Gaussian noise in frequency domain
white_noise_complex = (np.random.normal(0, 1, n_steps) + 
                       1j * np.random.normal(0, 1, n_steps)) / np.sqrt(2)

# Apply spectrum and transform to time domain
atm_fft = white_noise_complex * spectrum_magnitude
atm_disturbance = np.fft.ifft(atm_fft).real

# Normalize and scale to desired amplitude
atm_disturbance = atm_disturbance / np.std(atm_disturbance) * scaling

# --- 6. SIMULATION LOOP ---
# PID gains loaded from the verified scenario-based tuning run.
integral = 0.0
prev_error = 0.0

mirror_pos = np.zeros(n_steps)
residual_error = np.zeros(n_steps)
current_mirror_pos = 0.0

for k in range(n_steps):
    error = atm_disturbance[k] - current_mirror_pos
    
    # PID calculation
    integral += error * dt
    derivative = (error - prev_error) / dt
    pid_output = (kp * error) + (ki * integral) + (kd * derivative)
    prev_error = error
    
    # Pass PID output through the Notch Filter difference equation
    notch_in = [pid_output, notch_in[0], notch_in[1]]
    filtered_cmd = (b_n[0]*notch_in[0] + b_n[1]*notch_in[1] + b_n[2]*notch_in[2] 
                    - a_n[1]*notch_out[0] - a_n[2]*notch_out[1]) / a_n[0]
    notch_out = [filtered_cmd, notch_out[0], notch_out[1]]
    
    # Drive the physical FSM plant with the FILTERED command
    next_state = np.dot(A_p, plant_state) + B_p * filtered_cmd
    mirror_pos[k] = (np.dot(C_p, plant_state) + D_p * filtered_cmd).item()
    plant_state = next_state
    current_mirror_pos = mirror_pos[k]
    
    # Calculate residual error AFTER mirror responds to the control command
    residual_error[k] = atm_disturbance[k] - mirror_pos[k]

# --- 7. PLOT ---
fig, ax = plt.subplots(figsize=(11, 5))
fig.subplots_adjust(right=0.78)

lines = {
    "Atmospheric Jitter": ax.plot(t, atm_disturbance, label="Atmospheric Jitter", color="black", linewidth=2.0)[0],
    "FSM Tracking": ax.plot(t, mirror_pos, label="FSM Tracking (With Notch Filter)", color="tab:blue", linestyle="--")[0],
    "Residual Tracking Error": ax.plot(t, residual_error, label="Residual Tracking Error", color="tab:red")[0],
}

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Position / Error (mrad)")
ax.set_title(f"Atmospheric Correction Loop: Mirror Q={Q_x:.0f}/{Q_y:.0f} (X/Y), f₀={f_res:.0f}Hz")
ax.legend()
ax.grid(True)

check_ax = fig.add_axes([0.80, 0.35, 0.18, 0.25])
check_buttons = CheckButtons(
    check_ax,
    labels=list(lines.keys()),
    actives=[line.get_visible() for line in lines.values()],
)

def toggle_line(label):
    line = lines[label]
    line.set_visible(not line.get_visible())
    fig.canvas.draw_idle()

check_buttons.on_clicked(toggle_line)

plt.show()
