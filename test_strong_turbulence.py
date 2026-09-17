import numpy as np
import scipy.signal as signal

dt = 0.0001
n_steps = 5000
f_res = 437.0
w_res = 2 * np.pi * f_res
zeta_fsm = 1.0 / 40.0

# Notch
w_notch = 2 * np.pi * f_res
num_notch = [1, 0, w_notch**2]
den_notch = [1, w_notch / 20.0, w_notch**2]
notch_disc = signal.lti(num_notch, den_notch).to_discrete(dt, method='bilinear')
b_n, a_n = notch_disc.num, notch_disc.den

# Plant
num_p = [w_res**2]
den_p = [1, 2 * zeta_fsm * w_res, w_res**2]
plant_disc = signal.lti(num_p, den_p).to_discrete(dt, method='bilinear')
A_p, B_p, C_p, D_p = signal.tf2ss(plant_disc.num, plant_disc.den)

# Strong turbulence (f_corner=200)
np.random.seed(12)
freqs = np.fft.fftfreq(n_steps, dt)
freqs[0] = 1e-10
spectrum = 1.0 / ((freqs**2 + 100)**(11.0/12.0) * (1 + (np.abs(freqs)/200.0)**2)**0.5)
noise = (np.random.normal(0, 1, n_steps) + 1j*np.random.normal(0, 1, n_steps)) / np.sqrt(2)
atm = np.fft.ifft(noise * spectrum).real
atm = atm / np.std(atm) * 8.0

def simulate(kp, ki, kd, label):
    integral, prev_error = 0.0, 0.0
    mirror_pos = np.zeros(n_steps)
    plant_state = np.zeros((2, 1))
    notch_in, notch_out = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    for k in range(n_steps):
        error = atm[k] - mirror_pos[k]
        integral += error * dt
        derivative = (error - prev_error) / dt
        pid = (kp * error) + (ki * integral) + (kd * derivative)
        prev_error = error
        
        notch_in = [pid, notch_in[0], notch_in[1]]
        filtered = (b_n[0]*notch_in[0] + b_n[1]*notch_in[1] + b_n[2]*notch_in[2] 
                    - a_n[1]*notch_out[0] - a_n[2]*notch_out[1]) / a_n[0]
        notch_out = [filtered, notch_out[0], notch_out[1]]
        
        next_state = np.dot(A_p, plant_state) + B_p * filtered
        mirror_pos[k] = (np.dot(C_p, plant_state) + D_p * filtered).item()
        plant_state = next_state
    
    rms = np.sqrt(np.mean((atm - mirror_pos)**2))
    print(f'{label}: RMS = {rms:.4f} mrad')
    return rms

print('Strong Turbulence Test (f_corner = 200 Hz)')
print('='*50)
rms1 = simulate(1.2, 7.0, 0.00025, 'Current  (1.2, 7.0, 0.00025)')
rms2 = simulate(1.0, 6.0, 0.00020, 'Adjusted (1.0, 6.0, 0.00020)')
rms3 = simulate(1.0, 5.0, 0.00015, 'Conserv. (1.0, 5.0, 0.00015)')
print('='*50)
best = min(rms1, rms2, rms3)
if best == rms1: print('Recommendation: Keep CURRENT gains')
elif best == rms2: print('Recommendation: Use ADJUSTED gains')
else: print('Recommendation: Use CONSERVATIVE gains')
