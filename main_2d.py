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

# Mirror Quality Factors (from your actual mirror measurements)
# Q = 20 (X axis) and Q = 21 (Y axis)
Q_x = 20.0                          # Quality factor for X axis
Q_y = 21.0                          # Quality factor for Y axis

# Convert Q to damping ratio: ζ = 1/(2Q)
zeta_fsm_x = 1.0 / (2.0 * Q_x)      # Damping ratio for X axis: 0.025
zeta_fsm_y = 1.0 / (2.0 * Q_y)      # Damping ratio for Y axis: ~0.0238

# --- 3. CREATE NOTCH FILTERS FOR EACH AXIS ---
# X axis notch (Q=20)
f_notch = f_res
w_notch = 2 * np.pi * f_notch
Q_notch_x = Q_x
num_notch_x = [1, 0, w_notch**2]
den_notch_x = [1, w_notch / Q_notch_x, w_notch**2]
notch_cont_x = signal.lti(num_notch_x, den_notch_x)
notch_disc_x = notch_cont_x.to_discrete(dt, method='bilinear')
b_n_x, a_n_x = notch_disc_x.num, notch_disc_x.den

# Y axis notch (Q=21)
Q_notch_y = Q_y
num_notch_y = [1, 0, w_notch**2]
den_notch_y = [1, w_notch / Q_notch_y, w_notch**2]
notch_cont_y = signal.lti(num_notch_y, den_notch_y)
notch_disc_y = notch_cont_y.to_discrete(dt, method='bilinear')
b_n_y, a_n_y = notch_disc_y.num, notch_disc_y.den

# Filter state variables
notch_in_x = [0.0, 0.0, 0.0]
notch_out_x = [0.0, 0.0, 0.0]
notch_in_y = [0.0, 0.0, 0.0]
notch_out_y = [0.0, 0.0, 0.0]

# --- 4. CREATE FSM PLANT MODELS FOR EACH AXIS ---
# X axis plant (Q=20, ζ=0.025)
num_p_x = [w_res**2]
den_p_x = [1, 2 * zeta_fsm_x * w_res, w_res**2]
plant_cont_x = signal.lti(num_p_x, den_p_x)
plant_disc_x = plant_cont_x.to_discrete(dt, method='bilinear')
A_p_x, B_p_x, C_p_x, D_p_x = signal.tf2ss(plant_disc_x.num, plant_disc_x.den)
plant_state_x = np.zeros((A_p_x.shape[0], 1))

# Y axis plant (Q=21, ζ≈0.0238)
num_p_y = [w_res**2]
den_p_y = [1, 2 * zeta_fsm_y * w_res, w_res**2]
plant_cont_y = signal.lti(num_p_y, den_p_y)
plant_disc_y = plant_cont_y.to_discrete(dt, method='bilinear')
A_p_y, B_p_y, C_p_y, D_p_y = signal.tf2ss(plant_disc_y.num, plant_disc_y.den)
plant_state_y = np.zeros((A_p_y.shape[0], 1))

# --- 5. ATMOSPHERIC DISTURBANCE (2D) ---
np.random.seed(12)
# X axis disturbance
white_noise_x = np.random.normal(0, 1.0, n_steps)
b_atm, a_atm = signal.butter(2, 40.0 / (0.5 / dt), btype='low')
atm_disturbance_x = signal.lfilter(b_atm, a_atm, white_noise_x) * 8.0

# Y axis disturbance (different random seed for independence)
np.random.seed(13)
white_noise_y = np.random.normal(0, 1.0, n_steps)
atm_disturbance_y = signal.lfilter(b_atm, a_atm, white_noise_y) * 8.0

# --- 6. SIMULATION LOOP (2D) ---
# PID gains for each axis - OPTIMIZED for Q=20-21 mirror
# Axis-specific tuning: Y axis (Q=21) slightly more conservative than X (Q=20)
kp_x, ki_x, kd_x = 1.2, 7.0, 0.00025    # X axis: Q=20 (slightly lower Q = can handle more gain)
kp_y, ki_y, kd_y = 1.0, 6.0, 0.00020    # Y axis: Q=21 (higher Q = more conservative)

integral_x, integral_y = 0.0, 0.0
prev_error_x, prev_error_y = 0.0, 0.0

mirror_pos_x = np.zeros(n_steps)
mirror_pos_y = np.zeros(n_steps)
residual_error_x = np.zeros(n_steps)
residual_error_y = np.zeros(n_steps)
current_mirror_pos_x = 0.0
current_mirror_pos_y = 0.0

for k in range(n_steps):
    # X axis control
    error_x = atm_disturbance_x[k] - current_mirror_pos_x
    integral_x += error_x * dt
    derivative_x = (error_x - prev_error_x) / dt
    pid_output_x = (kp_x * error_x) + (ki_x * integral_x) + (kd_x * derivative_x)
    prev_error_x = error_x
    
    # X axis notch filter
    notch_in_x = [pid_output_x, notch_in_x[0], notch_in_x[1]]
    filtered_cmd_x = (b_n_x[0]*notch_in_x[0] + b_n_x[1]*notch_in_x[1] + b_n_x[2]*notch_in_x[2] 
                      - a_n_x[1]*notch_out_x[0] - a_n_x[2]*notch_out_x[1]) / a_n_x[0]
    notch_out_x = [filtered_cmd_x, notch_out_x[0], notch_out_x[1]]
    
    # X axis plant update
    next_state_x = np.dot(A_p_x, plant_state_x) + B_p_x * filtered_cmd_x
    mirror_pos_x[k] = (np.dot(C_p_x, plant_state_x) + D_p_x * filtered_cmd_x).item()
    plant_state_x = next_state_x
    current_mirror_pos_x = mirror_pos_x[k]
    residual_error_x[k] = atm_disturbance_x[k] - mirror_pos_x[k]
    
    # Y axis control
    error_y = atm_disturbance_y[k] - current_mirror_pos_y
    integral_y += error_y * dt
    derivative_y = (error_y - prev_error_y) / dt
    pid_output_y = (kp_y * error_y) + (ki_y * integral_y) + (kd_y * derivative_y)
    prev_error_y = error_y
    
    # Y axis notch filter
    notch_in_y = [pid_output_y, notch_in_y[0], notch_in_y[1]]
    filtered_cmd_y = (b_n_y[0]*notch_in_y[0] + b_n_y[1]*notch_in_y[1] + b_n_y[2]*notch_in_y[2] 
                      - a_n_y[1]*notch_out_y[0] - a_n_y[2]*notch_out_y[1]) / a_n_y[0]
    notch_out_y = [filtered_cmd_y, notch_out_y[0], notch_out_y[1]]
    
    # Y axis plant update
    next_state_y = np.dot(A_p_y, plant_state_y) + B_p_y * filtered_cmd_y
    mirror_pos_y[k] = (np.dot(C_p_y, plant_state_y) + D_p_y * filtered_cmd_y).item()
    plant_state_y = next_state_y
    current_mirror_pos_y = mirror_pos_y[k]
    residual_error_y[k] = atm_disturbance_y[k] - mirror_pos_y[k]

# --- 7. PLOT (2D Layout) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(right=0.75, hspace=0.3, wspace=0.3)

# X axis time series
ax_x = axes[0, 0]
lines_x = {
    "X Atmospheric": ax_x.plot(t, atm_disturbance_x, label="Atmospheric", color="black", linewidth=1.5)[0],
    "X Tracking": ax_x.plot(t, mirror_pos_x, label="Tracking", color="tab:blue", linestyle="--", linewidth=1.5)[0],
    "X Residual": ax_x.plot(t, residual_error_x, label="Residual", color="tab:red", linewidth=1.5)[0],
}
ax_x.set_xlabel("Time (seconds)")
ax_x.set_ylabel("Position / Error (mrad)")
ax_x.set_title(f"X Axis: Q={Q_x:.0f}, ζ={zeta_fsm_x:.4f}")
ax_x.legend(loc='upper left', fontsize=8)
ax_x.grid(True)

# Y axis time series
ax_y = axes[0, 1]
lines_y = {
    "Y Atmospheric": ax_y.plot(t, atm_disturbance_y, label="Atmospheric", color="black", linewidth=1.5)[0],
    "Y Tracking": ax_y.plot(t, mirror_pos_y, label="Tracking", color="tab:green", linestyle="--", linewidth=1.5)[0],
    "Y Residual": ax_y.plot(t, residual_error_y, label="Residual", color="tab:orange", linewidth=1.5)[0],
}
ax_y.set_xlabel("Time (seconds)")
ax_y.set_ylabel("Position / Error (mrad)")
ax_y.set_title(f"Y Axis: Q={Q_y:.0f}, ζ={zeta_fsm_y:.4f}")
ax_y.legend(loc='upper left', fontsize=8)
ax_y.grid(True)

# 2D trajectory plot (X vs Y)
ax_2d = axes[1, 0]
ax_2d.plot(atm_disturbance_x, atm_disturbance_y, 'k-', alpha=0.5, label="Disturbance", linewidth=1)
ax_2d.plot(mirror_pos_x, mirror_pos_y, 'b-', label="Mirror Position", linewidth=1.5)
ax_2d.set_xlabel("X Position (mrad)")
ax_2d.set_ylabel("Y Position (mrad)")
ax_2d.set_title("2D Trajectory: Mirror vs Disturbance")
ax_2d.legend(loc='upper left', fontsize=8)
ax_2d.grid(True)
ax_2d.axis('equal')

# Residual error comparison (RMS)
ax_stats = axes[1, 1]
window_size = 500  # 50ms window
rms_x = np.array([np.sqrt(np.mean(residual_error_x[max(0, i-window_size):i+1]**2)) 
                  for i in range(n_steps)])
rms_y = np.array([np.sqrt(np.mean(residual_error_y[max(0, i-window_size):i+1]**2)) 
                  for i in range(n_steps)])
ax_stats.plot(t, rms_x, 'b-', label=f"X Axis RMS (final={rms_x[-1]:.3f})", linewidth=1.5)
ax_stats.plot(t, rms_y, 'g-', label=f"Y Axis RMS (final={rms_y[-1]:.3f})", linewidth=1.5)
ax_stats.set_xlabel("Time (seconds)")
ax_stats.set_ylabel("RMS Error (mrad)")
ax_stats.set_title("Tracking Performance (Running RMS)")
ax_stats.legend(loc='upper left', fontsize=8)
ax_stats.grid(True)

# Overall title
fig.suptitle(f"2D Atmospheric Correction: Mirror Q={Q_x:.0f}/{Q_y:.0f} (X/Y), f₀={f_res:.0f}Hz", 
             fontsize=12, fontweight='bold')

# Add check buttons for toggling traces (positioned on the right)
check_ax = fig.add_axes([0.77, 0.35, 0.18, 0.3])
all_lines = {**lines_x, **lines_y}
check_buttons = CheckButtons(
    check_ax,
    labels=list(all_lines.keys()),
    actives=[line.get_visible() for line in all_lines.values()],
)

def toggle_line(label):
    line = all_lines[label]
    line.set_visible(not line.get_visible())
    fig.canvas.draw_idle()

check_buttons.on_clicked(toggle_line)

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("2D ATMOSPHERIC CORRECTION SIMULATION SUMMARY")
print("="*60)
print(f"\nMirror Parameters:")
print(f"  Resonance Frequency: {f_res:.1f} Hz")
print(f"  X Axis: Q={Q_x:.1f}, damping={zeta_fsm_x:.4f}")
print(f"  Y Axis: Q={Q_y:.1f}, damping={zeta_fsm_y:.4f}")
print(f"\nTracking Performance (Final RMS Error):")
print(f"  X Axis: {rms_x[-1]:.4f} mrad")
print(f"  Y Axis: {rms_y[-1]:.4f} mrad")
print(f"  Combined: {np.sqrt(rms_x[-1]**2 + rms_y[-1]**2):.4f} mrad")
print(f"\nDisturbance Statistics:")
print(f"  X Axis: std={np.std(atm_disturbance_x):.2f} mrad")
print(f"  Y Axis: std={np.std(atm_disturbance_y):.2f} mrad")
print(f"\nRejection Ratio (Disturbance/RMS Error):")
print(f"  X Axis: {np.std(atm_disturbance_x)/rms_x[-1]:.1f}x")
print(f"  Y Axis: {np.std(atm_disturbance_y)/rms_y[-1]:.1f}x")
print("="*60)
