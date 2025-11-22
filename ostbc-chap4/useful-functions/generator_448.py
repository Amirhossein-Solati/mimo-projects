import numpy as np
from code_438 import code_4_38 as c44


def code_G448(sym):
    """
    Constructs the STBC matrix G448 for 4 transmit antennas.

    This code is a rate-1/2 code, constructed by vertically stacking the
    orthogonal matrix from `code_4_38` (let's call it G0) and its
    complex conjugate, `conj(G0)`. This results in an 8x4 matrix,
    transmitting 4 symbols over 8 time slots.

    Parameters
    ----------
    sym : np.ndarray
        A numpy array containing the four complex symbols to be encoded.

    Returns
    -------
    np.ndarray
        The 8x4 complex-valued STBC matrix for G448.
    """
    # 1. Input validation is handled inside `code_4_38`.
    # Let's ensure the input is a NumPy array.
    symbols = np.asarray(sym)

    # 2. Generate the base 4x4 orthogonal matrix.
    # Reusing code is a key principle of good software design.
    g0_matrix = c44(symbols)
    
    # 3. Calculate the element-wise complex conjugate of the base matrix.
    # `np.conj()` is the NumPy function for this.
    conj_g0_matrix = np.conj(g0_matrix)
    
    # 4. Vertically stack the two matrices.
    # `np.vstack()` takes a tuple of arrays and stacks them vertically.
    g448_matrix = np.vstack((g0_matrix, conj_g0_matrix))
    
    return g448_matrix

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Create four example complex QPSK symbols
    qpsk_symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2) # Normalized
    
    # Generate the G448 code matrix
    stbc_matrix = code_G448(qpsk_symbols)
    
    g0_matrix = code_G448(qpsk_symbols)

    print("Input Symbols:\n", qpsk_symbols)
    print("\nBased Matrix for generating 8x4:\n", np.round(g0_matrix,2))
    print("\nGenerated G448 Matrix (8x4):\n", np.round(stbc_matrix, 2))
    
    # Verify the shape of the output matrix
    print("\nShape of the output matrix:", stbc_matrix.shape)
    assert stbc_matrix.shape == (8, 4)
    print("Shape test passed!")