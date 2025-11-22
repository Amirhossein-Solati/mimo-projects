import numpy as np
from itertools import product
from code_438 import code_4_38 as c44


def ml_bpsk_c44(r, H):
    """
    Maximum Likelihood Detector for the STBC from c44 with BPSK modulation.

    This function performs an exhaustive search over all 2^4 = 16 possible
    combinations of four BPSK symbols to find the combination that was most
    likely transmitted.

    Parameters
    ----------
    r : np.ndarray
        The received signal matrix, shape (4, 1).
    H : np.ndarray
        The channel matrix, shape (4, 1).

    Returns
    -------
    np.ndarray
        A column vector of shape (4, 1) containing the four detected BPSK symbols.
    """
    # 1. Define the BPSK constellation.
    sx = np.array([1, -1])

    # 2. Generate all 16 possible combinations of four BPSK symbols.
    # `product(sx, repeat=4)` creates an iterator for all combinations.
    possible_symbol_quads = product(sx, repeat=4)

    # Variables to store the best match found so far.
    min_cost = np.inf
    detected_symbols = None

    # 3. Iterate through each of the 16 possible symbol combinations.
    for symbol_quad in possible_symbol_quads:
        # a. Convert the current combination (a tuple) to a NumPy array.
        current_symbols = np.array(symbol_quad)
        
        # b. Create the hypothetical STBC matrix for this combination using c44.
        C = c44(current_symbols)
        
        # c. Calculate the cost (squared Frobenius norm of the difference).
        # This is the standard and efficient way to compute `trace(k0'*k0)`.
        cost_matrix = r - (C @ H)
        cost = np.linalg.norm(cost_matrix, 'fro')**2
        
        # d. If this cost is the smallest yet, update the best match.
        if cost < min_cost:
            min_cost = cost
            detected_symbols = current_symbols

    # 4. Return the detected symbols shaped as a column vector (4, 1).
    return detected_symbols.reshape(4, 1)

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Define a sample channel H with 4 gains (4 Tx antennas)
    H_sample = np.array([[0.5 - 0.2j], [0.1 + 0.8j], [-0.3 - 0.4j], [0.9 + 0.1j]])
    
    # Define a set of symbols hypothetically sent
    sent_symbols = np.array([-1, 1, 1, -1])
    
    # Create the corresponding STBC matrix
    C_sent = c44(sent_symbols)
    
    # Create a noise-free received signal r
    r_noise_free = C_sent @ H_sample
    
    # Add some noise
    noise = 0.01 * (np.random.randn(4, 1) + 1j * np.random.randn(4, 1))
    r_with_noise = r_noise_free + noise
    
    # Run the detector
    detected_result = ml_bpsk_c44(r_with_noise, H_sample)
    
    print("Sent Symbols:\n", sent_symbols.reshape(4, 1))
    print("\nDetected Symbols:\n", detected_result)

    # Test with the noise-free signal to ensure the detector's logic is correct
    detected_noise_free_result = ml_bpsk_c44(r_noise_free, H_sample)
    print("\nDetected Symbols (Noise-Free):\n", detected_noise_free_result)
    assert np.array_equal(sent_symbols.reshape(4,1), detected_noise_free_result)
    print("Noise-free test passed!")