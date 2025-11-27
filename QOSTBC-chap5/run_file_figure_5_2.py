# run_simulation_qostbc.py

import numpy as np
import os
import functions.stbc_functions as stbc # فرض می‌کنیم توابع جدید به این فایل اضافه خواهند شد

# --- Simulation Parameters ---
SNRdB = np.arange(6, 18.1, 2) # Range: 6, 8, 10, 12, 14, 16, 18
SNR = 10**(SNRdB / 10)
Nb = int(1e6)
varh_dB = 0
varh = 10**(varh_dB / 10)

# --- QOSTBC Specific Parameter (phi) ---
# Create the array for phi values
ph0 = np.array([
    0, 
    np.pi/4,  # 45 degrees
    np.pi/3,  # 60 degrees
    np.pi/2   # 90 degrees
]) 

# --- Setup for saving results ---
RESULTS_DIR = 'results_qostbc'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulation_results.npz')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Simulation: 4x1 BPSK with QOSTBC (5.2) ---
print("--- Sim: 4x1 BPSK with QOSTBC (5.2) ---")
N, M, K = 4, 1, 4 # 4 Tx, 1 Rx, 4 bits per block

# Initialize a 2D array to store results: (num_phi_values x num_snr_values)
Err1 = np.zeros((len(ph0), len(SNR)))

# Outer loop: Iterate over each value of phi
for p0, phi_val in enumerate(ph0):
    print(f"\n--- Running simulation for phi = {phi_val:.4f} ---")
    
    # Inner loop: Iterate over each SNR value
    for i0, snr_val in enumerate(SNR):
        Err = 0
        num_loops = int(Nb / K)
        
        # This is a new parameter needed by the ML detector
        signal_amp = np.sqrt(snr_val / N)
        
        # Monte Carlo loop
        for _ in range(num_loops):
            # 1. Generate 4 BPSK symbols
            b0 = 2 * np.random.randint(0, 2, size=(K, 1)) - 1
            
            # 2. QOSTBC Encoding (depends on phi)
            C = stbc.code_qostbc_5_2(b0, phi_val)
            T = C.shape[0]
            
            # 3. Generate Channel and Noise
            H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
            n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M))
            
            # 4. Calculate Received Signal
            r0 = signal_amp * (C @ H) + n0
            
            # 5. ML Detection (now requires phi and signal_amp)
            ds_MLD = stbc.ml_qostbc_bpsk_5_2(r0, H, phi_val, signal_amp)
            
            # 6. Count Errors
            Err += np.sum(ds_MLD != b0)
        
        # Store BER in the 2D results matrix
        Err1[p0, i0] = Err / (num_loops * K)
        print(f"  SNR (dB): {SNRdB[i0]:<4} | BER: {Err1[p0, i0]:.7f}")

# --- Save Results ---
np.savez(
    RESULTS_FILE,
    snr_db=SNRdB,
    phi_values=ph0,
    ber_results=Err1
)
print(f"\nQOSTBC simulation complete. Results saved to '{RESULTS_FILE}'.")