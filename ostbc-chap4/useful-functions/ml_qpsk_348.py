import numpy as np
from itertools import product
from generator_348 import code_G348

# --- Main Detector Function (Corrected) ---

def myMLD_QPSK_G348(r, H):
    """
    Maximum Likelihood Detector for the G348 STBC with QPSK modulation.

    This function performs an exhaustive search over all 4^4 = 256 possible
    combinations of four QPSK symbols.

    Parameters
    ----------
    r : np.ndarray
        The received signal matrix, shape (8, M), where M is the number of Rx antennas.
        (The code is transmitted over 8 time slots).
    H : np.ndarray
        The channel matrix, shape (3, M), where M is the number of Rx antennas.
        (The code uses 3 transmit antennas).

    Returns
    -------
    np.ndarray
        A column vector of shape (4, 1) containing the four detected QPSK symbols.
    """
    # 1. Define the QPSK constellation.
    sx = np.array([1, -1, -1j, 1j])

    # 2. Generate all 256 possible combinations of four QPSK symbols.
    possible_symbol_quads = product(sx, repeat=4)

    min_cost = np.inf
    detected_symbols = None

    # 3. Iterate through each of the 256 possible symbol combinations.
    for symbol_quad in possible_symbol_quads:
        current_symbols = np.array(symbol_quad)
        
        # Create the hypothetical 8x3 STBC matrix (G348).
        C = code_G348(current_symbols)
        
        # Calculate the cost (squared Frobenius norm of the error matrix).
        cost = np.linalg.norm(r - (C @ H), 'fro')**2
        
        if cost < min_cost:
            min_cost = cost
            detected_symbols = current_symbols

    # 4. Return the detected symbols as a column vector.
    return detected_symbols.reshape(4, 1)


# Test Section
if __name__ == '__main__':
    print("--- Testing myMLD_QPSK_G348 with correct dimensions ---")

    N_test, M_test = 3, 2

    H_sample = np.sqrt(0.5) * (np.random.randn(N_test, M_test) + 1j * np.random.randn(N_test, M_test))
    
    # Define hypothetically sent symbols
    sent_symbols = np.array([1j, -1, -1j, 1])
    
    # Create the corresponding STBC matrix G348
    C_sent = code_G348(sent_symbols)
    T_test = C_sent.shape[0]  # This will be 8
    
    # Create a noise-free received signal r = C * H
    r_noise_free = C_sent @ H_sample
    
    print(f"Shape of STBC matrix C: {C_sent.shape}") # Expected: (8, 3)
    print(f"Shape of Channel matrix H: {H_sample.shape}") # Expected: (3, 2)
    print(f"Shape of received signal r: {r_noise_free.shape}") # Expected: (8, 2)
    assert C_sent.shape == (8, 3)
    assert r_noise_free.shape == (8, M_test)
    
    # Add Gaussian noise with correct dimensions (T, M) -> (8, 2)
    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(T_test, M_test) + 1j * np.random.randn(T_test, M_test))
    r_with_noise = r_noise_free + noise
    
    # Run the detector
    print("\nRunning ML detection (256 iterations)...")
    detected_result = myMLD_QPSK_G348(r_with_noise, H_sample)
    
    print("\nSent Symbols:\n", sent_symbols.reshape(4, 1))
    print("Detected Symbols:\n", detected_result)
    
    # Verification with noise-free signal should always pass
    detected_noise_free = myMLD_QPSK_G348(r_noise_free, H_sample)
    assert np.allclose(sent_symbols.reshape(4, 1), detected_noise_free)
    print("\nNoise-free detection test PASSED.")