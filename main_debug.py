import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# --- 1. SYSTEM INITIALISATION ---
dt = 0.0001  # 10 kHz sampling rate
t = np.arange(0, 0.5, dt)
n_steps = len(t)

# --- 2. CONFIGURING SYSTEM FREQUENCIES ---
f_res = 437.0                       # FSM Resonant Frequency (Hz) from datasheet
w_res = 2 * np.pi * f_res           # Convert to rad/s
zeta_fsm = 0.05                     # Very low damping (highly resonant mirror)

# --- 3. CREATE THE NOTCH FILTER ---
f_notch = f_res                     # Center the notch exactly on the resonance
w_notch = 2 * np.pi * f_notch
Q = 10.0                            # Sharpness of notch (High Q = narrow/deep)
zeta_notch = 1.0                    # Overdamped numerator to flatten the depth

# Continuous Notch: H(s) = (s^2 + w_n^2) / (s^2 + (w_n/Q)*s + w_n^2)
num_notch = [1, 0, w_notch**2]
den_notch = [1, w_notch / Q, w_notch**2]
notch_cont = signal.lti(num_notch, den_notch)
notch_disc = notch_cont.to_discrete(dt, method='bilinear')

# Filter coefficients for difference equation: y[k] = b0*x[k] + b1*x[k-1]... - a1*y[k-1]...
b_n, a_n = notch_disc.num, notch_disc.den
print(f"Notch filter coefficients:")
print(f"  b_n = {b_n}")
print(f"  a_n = {a_n}")

notch_in = [0.0, 0.0, 0.0]
notch_out = [0.0, 0.0, 0.0]

# --- 4. CREATE THE FSM PLANT MODEL ---
num_p = [w_res**2]
den_p = [1, 2 * zeta_fsm * w_res, w_res**2]
plant_cont = signal.lti(num_p, den_p)
plant_disc = plant_cont.to_discrete(dt, method='bilinear')
A_p, B_p, C_p, D_p = signal.tf2ss(plant_disc.num, plant_disc.den)

print(f"\nPlant state-space matrices:")
print(f"  A_p = {A_p}")
print(f"  B_p = {B_p}")
print(f"  C_p = {C_p}")
print(f"  D_p = {D_p}")

plant_state = np.zeros((A_p.shape[0], 1))

# --- 5. ATMOSPHERIC DISTURBANCE ---
np.random.seed(12)
white_noise = np.random.normal(0, 1.0, n_steps)
b_atm, a_atm = signal.butter(2, 1000.0 / (0.5 / dt), btype='low')  # 100Hz atmospheric jitter
atm_disturbance = signal.lfilter(b_atm, a_atm, white_noise) * 8.0

print(f"\nAtmospheric disturbance stats:")
print(f"  Mean: {np.mean(atm_disturbance):.4f}")
print(f"  Std: {np.std(atm_disturbance):.4f}")
print(f"  Max: {np.max(atm_disturbance):.4f}")
print(f"  Min: {np.min(atm_disturbance):.4f}")

# --- 6. SIMULATION LOOP ---
# Because we have a notch filter, we can double our PID gains safely!
kp, ki, kd = 0.5, 400.0, 0.004

print(f"\nPID gains:")
print(f"  kp = {kp}, ki = {ki}, kd = {kd}")
print(f"  dt = {dt}")
print(f"  Effective integral gain per step: ki * dt = {ki * dt}")
print(f"  Effective derivative gain per step: kd / dt = {kd / dt}")

integral = 0.0
prev_error = 0.0

mirror_pos = np.zeros(n_steps)
residual_error = np.zeros(n_steps)
current_mirror_pos = 0.0

# Track values for debugging
pid_outputs = []
filtered_cmds = []
errors = []
integrals = []
derivatives = []

for k in range(min(100, n_steps)):  # Only run first 100 steps to see the explosion
    error = atm_disturbance[k] - current_mirror_pos
    
    # PID calculation
    integral += error * dt
    derivative = (error - prev_error) / dt
    pid_output = (kp * error) + (ki * integral) + (kd * derivative)
    prev_error = error
    
    # Store for debugging
    errors.append(error)
    integrals.append(integral)
    derivatives.append(derivative)
    pid_outputs.append(pid_output)
    
    # Pass PID output through the Notch Filter difference equation
    notch_in = [pid_output, notch_in[0], notch_in[1]]
    filtered_cmd = (b_n[0]*notch_in[0] + b_n[1]*notch_in[1] + b_n[2]*notch_in[2] 
                    - a_n[1]*notch_out[0] - a_n[2]*notch_out[1]) / a_n[0]
    notch_out = [filtered_cmd, notch_out[0], notch_out[1]]
    filtered_cmds.append(filtered_cmd)
    
    # Drive the physical FSM plant with the FILTERED command
    next_state = np.dot(A_p, plant_state) + B_p * filtered_cmd
    current_mirror_pos = (np.dot(C_p, plant_state) + D_p * filtered_cmd).item()
    mirror_pos[k] = current_mirror_pos
    residual_error[k] = atm_disturbance[k] - current_mirror_pos
    plant_state = next_state
    
    # Print debug info every 10 steps
    if k % 10 == 0:
        print(f"\nStep {k}:")
        print(f"  error = {error:.6f}, integral = {integral:.6f}, derivative = {derivative:.2f}")
        print(f"  pid_output = {pid_output:.2f}, filtered_cmd = {filtered_cmd:.2f}")
        print(f"  mirror_pos = {current_mirror_pos:.6f}")
        
    # Check for NaN or Inf
    if not np.isfinite(pid_output) or not np.isfinite(filtered_cmd) or not np.isfinite(current_mirror_pos):
        print(f"\n*** NUMERICAL ISSUE at step {k} ***")
        print(f"  pid_output = {pid_output}")
        print(f"  filtered_cmd = {filtered_cmd}")
        print(f"  mirror_pos = {current_mirror_pos}")
        break

print(f"\n\nFirst 20 PID outputs: {pid_outputs[:20]}")
print(f"First 20 filtered commands: {filtered_cmds[:20]}")
