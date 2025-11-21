import numpy as np
import matplotlib.pyplot as plt
import os

# --- Parameters (Updated for M=2) ---
RESULTS_DIR = 'results_m2'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulation_results_m2.npz')

# --- Check for results file ---
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Results file not found at '{RESULTS_FILE}'")
    print("Please run 'run_simulation_m2.py' first to generate the results.")
else:
    # --- Load Data ---
    print(f"Loading results from '{RESULTS_FILE}'...")
    data = np.load(RESULTS_FILE)
    SNRdB = data['snr_db']
    Err1 = data['err1']
    Err2 = data['err2']
    Err3 = data['err3']
    Err4 = data['err4']
    Err5 = data['err5']
    Err6 = data['err6']

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    
    # Updated labels to reflect M=2
    plt.semilogy(SNRdB, Err1, 'rs--', label='1 Tx, 2 Rx (BPSK)')
    plt.semilogy(SNRdB, Err2, 'gd-', label='2 Tx, 2 Rx (Alamouti, BPSK)')
    plt.semilogy(SNRdB, Err3, 'bv:', label='3 Tx, 2 Rx (Rate-1, BPSK)')
    plt.semilogy(SNRdB, Err4, 'k*--', label='4 Tx, 2 Rx (Rate-1, BPSK)')
    plt.semilogy(SNRdB, Err5, 'c^-', label='3 Tx, 2 Rx (Rate-1, QPSK)')
    plt.semilogy(SNRdB, Err6, 'mo:', label='4 Tx, 2 Rx (Rate-1/2, QPSK)')

    # --- Formatting ---
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Performance of STBC Schemes with 2 Receive Antennas', fontsize=16)
    plt.legend(fontsize=11)
    plt.ylim([1e-7, 1.0]) # Adjusted ylim for potentially lower BER
    plt.xlim([min(SNRdB), max(SNRdB)])
    plt.tight_layout()
    plt.show()