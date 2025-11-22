import numpy as np

from generator_448 import code_G448 as g448 

def code_G348(sym):
    """
    Constructs the STBC matrix G348 for 3 transmit antennas.

    This code is derived by taking the first three columns of the G448 code matrix.
    It's a rate-1/2 code that transmits 4 symbols over 8 time slots using 3 antennas.

    Parameters
    ----------
    sym : np.ndarray
        A numpy array containing the four complex symbols to be encoded.
        It is expected to have a size of 4.

    Returns
    -------
    np.ndarray
        The 8x3 complex-valued STBC matrix for G348.
    """
    # 1. Input validation is implicitly handled by code_G448.
    symbols = np.asarray(sym)

    # 2. Generate the full 8x4 G448 code matrix.
    g448_matrix = g448(symbols)

    # 3. Select the first three columns to form the G348 matrix.
    # Input is (8, 4), output will be (8, 3).
    g348_matrix = g448_matrix[:, 0:3]
    
    return g348_matrix

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Create four example complex symbols (e.g., from a QPSK modulator)
    qpsk_symbols = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    
    # Generate the G348 code matrix
    stbc_matrix = code_G348(qpsk_symbols)
    
    print("Input Symbols:\n", qpsk_symbols)
    print("\nGenerated G348 Matrix (8x3):\n", stbc_matrix)
    
    # Verify the shape of the output matrix
    print("\nShape of the output matrix:", stbc_matrix.shape)
    
    assert stbc_matrix.shape == (8, 3)
    
    print("Shape test passed!")