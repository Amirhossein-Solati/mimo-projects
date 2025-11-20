import numpy as np
import matplotlib.pyplot as plt
import os

# --- Parameters ---
RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulation_results.npz')

# --- Check for results file ---
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Results file not found at '{RESULTS_FILE}'")
    print("Please run 'run_simulation.py' first to generate the results.")
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

    plt.semilogy(SNRdB, Err1, 'rs--', label='1 Tx, 1 Rx (BPSK)')
    plt.semilogy(SNRdB, Err2, 'gd-', label='2 Tx, 1 Rx (Alamouti, BPSK)')
    # Note: Corrected legends based on my analysis of the codes
    plt.semilogy(SNRdB, Err3, 'bv:', label='3 Tx, 1 Rx (Rate-1, BPSK)')
    plt.semilogy(SNRdB, Err4, 'k*--', label='4 Tx, 1 Rx (Rate-1, BPSK)')
    plt.semilogy(SNRdB, Err5, 'c^-', label='3 Tx, 1 Rx (Rate-1, QPSK)') # Note: code is Rate-1
    plt.semilogy(SNRdB, Err6, 'mo:', label='4 Tx, 1 Rx (Rate-1/2, QPSK)')

    # --- Formatting ---
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Performance of Various STBC Schemes in Rayleigh Fading', fontsize=16)
    plt.legend(fontsize=11)
    plt.ylim([1e-6, 1.0])
    plt.xlim([min(SNRdB), max(SNRdB)])
    plt.tight_layout()
    plt.show()