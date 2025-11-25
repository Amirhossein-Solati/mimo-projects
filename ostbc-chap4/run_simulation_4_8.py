import numpy as np
import os
import functions.stbc_functions as stbc


SNRdB = np.arange(-1, 16.1, 5)  # New SNR range: -1, 4, 9, 14, 19
SNR = 10**(SNRdB / 10)
Nb = int(2e6)  # Increased number of bits for higher accuracy
varh_dB = 0
varh = 10**(varh_dB / 10)
RESULTS_DIR = 'results_m2' # Using a new directory for the new results
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulation_results_m2.npz')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Simulation 1: 1 Tx, 2 Rx (SIMO) BPSK ---
print("--- Sim 1: 1x2 SIMO BPSK ---")
N, M = 1, 2 # M is now 2
Err1 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    for _ in range(Nb):
        b1 = 2 * np.random.randint(0, 2) - 1
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 1x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(1, M) + 1j * np.random.randn(1, M)) # n0 is now 1x2
        r0 = np.sqrt(snr_val / N) * b1 * H + n0 # r0 is now 1x2
        # The detector function still works! It's designed for any M.
        ds_MLD = stbc.ml_bpsk_siso(r0, H)
        if ds_MLD != b1:
            Err += 1
    Err1[i] = Err / Nb
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err1[i]:.6f}")

# --- Simulation 2: 2 Tx, 2 Rx (MIMO) Alamouti BPSK ---
print("\n--- Sim 2: 2x2 MIMO Alamouti BPSK ---")
N, M = 2, 2 # M is now 2
Err2 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    num_loops = int(Nb / N)
    for _ in range(num_loops):
        b1 = 2 * np.random.randint(0, 2, size=(N, 1)) - 1
        C = stbc.alamouti(b1)
        T = C.shape[0]
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 2x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M)) # n0 is now 2x2
        r = np.sqrt(snr_val / N) * (C @ H) + n0 # r is now 2x2
        ds_MLD = stbc.ml_alamouti_bpsk(r, H)
        Err += np.sum(ds_MLD != b1)
    Err2[i] = Err / (num_loops * N)
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err2[i]:.6f}")

# --- Simulation 3: 3 Tx, 2 Rx (MIMO) BPSK with Code (4.61) ---
print("\n--- Sim 3: 3x2 MIMO BPSK (Code 4.61) ---")
N, M, K = 3, 2, 4 # M is now 2
Err3 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    num_loops = int(Nb / K)
    for _ in range(num_loops):
        b1 = 2 * np.random.randint(0, 2, size=(K, 1)) - 1
        C = stbc.code_4_61(b1)
        T = C.shape[0]
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 3x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M)) # n0 is now 4x2
        r = np.sqrt(snr_val / N) * (C @ H) + n0 # r is now 4x2
        ds_MLD = stbc.ml_4_61_bpsk(r, H)
        Err += np.sum(ds_MLD != b1)
    Err3[i] = Err / (num_loops * K)
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err3[i]:.6f}")

# --- Simulation 4: 4 Tx, 2 Rx (MIMO) BPSK with Code (4.38) ---
print("\n--- Sim 4: 4x2 MIMO BPSK (Code 4.38) ---")
N, M, K = 4, 2, 4 # M is now 2
Err4 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    num_loops = int(Nb / K)
    for _ in range(num_loops):
        b1 = 2 * np.random.randint(0, 2, size=(K, 1)) - 1
        C = stbc.code_4_38(b1)
        T = C.shape[0]
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 4x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M)) # n0 is now 4x2
        r = np.sqrt(snr_val / N) * (C @ H) + n0 # r is now 4x2
        ds_MLD = stbc.ml_4_38_bpsk(r, H)
        Err += np.sum(ds_MLD != b1)
    Err4[i] = Err / (num_loops * K)
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err4[i]:.6f}")

# --- Simulation 5: 3 Tx, 2 Rx (MIMO) QPSK with Code G348 ---
print("\n--- Sim 5: 3x2 MIMO QPSK (Code G348) ---")
N, M, K_sym, bits_per_sym = 3, 2, 4, 2
bits_per_block = K_sym * bits_per_sym
Err5 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    num_loops = int(Nb / bits_per_block)
    for _ in range(num_loops):
        b0 = np.random.randint(0, 2, size=(bits_per_block, 1))
        b1 = stbc.myMod_QPSK(b0)
        C = stbc.code_G348(b1)
        T = C.shape[0]
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 3x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M))
        r = np.sqrt(snr_val / N) * (C @ H) + n0
        ds_MLD = stbc.ml_qpsk_G348(r, H)
        ds_MLD1 = stbc.demod_qpsk(ds_MLD)
        Err += np.sum(ds_MLD1 != b0)
    Err5[i] = Err / (num_loops * bits_per_block)
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err5[i]:.6f}")

# --- Simulation 6: 4 Tx, 2 Rx (MIMO) QPSK with Code G448 ---
print("\n--- Sim 6: 4x2 MIMO QPSK (Code G448) ---")
N, M, K_sym, bits_per_sym = 4, 2, 4, 2
bits_per_block = K_sym * bits_per_sym
Err6 = np.zeros(len(SNR))
for i, snr_val in enumerate(SNR):
    Err = 0
    num_loops = int(Nb / bits_per_block)
    for _ in range(num_loops):
        b0 = np.random.randint(0, 2, size=(bits_per_block, 1))
        b1 = stbc.myMod_QPSK(b0)
        C = stbc.code_G448(b1)
        T = C.shape[0]
        H = np.sqrt(varh / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) # H is now 4x2
        n0 = np.sqrt(1 / 2) * (np.random.randn(T, M) + 1j * np.random.randn(T, M))
        r = np.sqrt(snr_val / N) * (C @ H) + n0
        ds_MLD = stbc.ml_qpsk_G448(r, H)
        ds_MLD1 = stbc.demod_qpsk(ds_MLD)
        Err += np.sum(ds_MLD1 != b0)
    Err6[i] = Err / (num_loops * bits_per_block)
    print(f"SNR (dB): {SNRdB[i]}, BER: {Err6[i]:.6f}")

# --- Save Results ---
np.savez(
    RESULTS_FILE,
    snr_db=SNRdB,
    err1=Err1, err2=Err2, err3=Err3,
    err4=Err4, err5=Err5, err6=Err6
)
print(f"\nSimulation complete. Results saved to '{RESULTS_FILE}'.")