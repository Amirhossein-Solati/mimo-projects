import numpy as np

def code_4_38(sym):
    """
    Constructs the Rate-1 Orthogonal STBC for 4 transmit antennas, as per equation (4.38).

    This code maps four input symbols to a 4x4 orthogonal transmission matrix,
    which is sent from 4 antennas over 4 time slots.

    The structure of the code matrix is:
    [ x1   x2   x3   x4 ]
    [ -x2  x1  -x4   x3 ]
    [ -x3  x4   x1  -x2 ]
    [ -x4 -x3   x2   x1 ]

    Parameters
    ----------
    sym : np.ndarray or list/tuple
        A numpy array (or list/tuple) containing the four symbols (x1, x2, x3, x4)
        to be encoded. It is expected to have a size of 4.

    Returns
    -------
    np.ndarray
        The 4x4 complex-valued STBC matrix.

    Raises
    ------
    ValueError
        If the input 'sym' does not contain exactly four elements.
    """
    # 1. Input Validation: Ensure the input has exactly four symbols.
    if np.size(sym) != 4:
        raise ValueError(f"Input must contain exactly 4 symbols, but got {np.size(sym)}.")

    # 2. Extract symbols for clarity and to match the mathematical formula.
    # .flatten() ensures robustness against different input shapes (row vs. column).
    x1, x2, x3, x4 = sym.flatten()

    # 3. Construct the 4x4 STBC matrix directly using np.array.
    code_matrix = np.array([
        [ x1,  x2,  x3,  x4],
        [-x2,  x1, -x4,  x3],
        [-x3,  x4,  x1, -x2],
        [-x4, -x3,  x2,  x1]
    ])
    
    return code_matrix

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Create four example BPSK symbols
    symbols = np.array([1, -1, 1, 1])
    
    # Generate the code matrix
    stbc_matrix = code_4_38(symbols)
    
    print("Input Symbols:\n", symbols)
    print("\nGenerated STBC Matrix (4.38):\n", stbc_matrix)

    # --- Verification of Orthogonality ---
    
    # For real symbols, C^H is just the transpose C.T
    # C_T_C = stbc_matrix.T @ stbc_matrix
    # print("\nVerification (C.T @ C):\n", C_T_C)
    
    # For complex symbols
    complex_symbols = np.array([1+2j, -1+1j, 1-3j, -1-1j])
    complex_stbc_matrix = code_4_38(complex_symbols)
    hermitian_transpose = complex_stbc_matrix.conj().T
    
    C_H_C = hermitian_transpose @ complex_stbc_matrix
    print("\nVerification with complex symbols (C^H @ C):\n", np.round(C_H_C, 2))