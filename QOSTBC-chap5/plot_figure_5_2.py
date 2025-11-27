import numpy as np
import matplotlib.pyplot as plt
import os

# --- Parameters ---
RESULTS_DIR = 'results_qostbc'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulation_results.npz')

# --- Check for results file ---
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Results file not found at '{RESULTS_FILE}'")
    print("Please run 'run_simulation_qostbc.py' first.")
else:
    # --- Load Data ---
    print(f"Loading results from '{RESULTS_FILE}'...")
    data = np.load(RESULTS_FILE)
    SNRdB = data['snr_db']
    phi_values = data['phi_values']
    Err1 = data['ber_results']

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    
    # Define styles for each curve to match the MATLAB plot
    styles = ['--rs', '-go', ':b*', '-.k^']
    
    # Loop through the results for each phi and plot its BER curve
    for p0, phi_val in enumerate(phi_values):
        label_text = f"phi = {phi_val:.2f}"
        plt.semilogy(SNRdB, Err1[p0, :], styles[p0 % len(styles)], label=label_text)

    # --- Formatting ---
    plt.grid(True, which="both", linestyle='--')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Performance of 4x1 QOSTBC with Varying \u03C6', fontsize=16) # Using unicode for phi
    plt.legend(fontsize=12)
    plt.ylim([1e-6, 1.0])
    plt.xlim([min(SNRdB), max(SNRdB)])
    plt.tight_layout()
    plt.show()