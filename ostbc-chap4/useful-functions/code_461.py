import numpy as np

def code_4_61(sym):
    """
    Constructs the Rate-1 STBC for 3 transmit antennas, as per equation (4.61).

    This code maps four input symbols to a 4x3 transmission matrix, which
    is sent from 3 antennas over 4 time slots.

    The structure of the code matrix is:
    [ x1   x2   x3 ]
    [ -x2  x1  -x4 ]
    [ -x3  x4   x1 ]
    [ -x4 -x3   x2 ]

    Parameters
    ----------
    sym : np.ndarray or list/tuple
        A numpy array (or list/tuple) containing the four symbols (x1, x2, x3, x4)
        to be encoded. It is expected to have a size of 4.

    Returns
    -------
    np.ndarray
        The 4x3 complex-valued STBC matrix.

    Raises
    ------
    ValueError
        If the input 'sym' does not contain exactly four elements.
    """
    # 1. Input Validation: Ensure the input has exactly four symbols.
    if np.size(sym) != 4:
        raise ValueError(f"Input must contain exactly 4 symbols, but got {np.size(sym)}.")

    # 2. Extract symbols for clarity.
    # .flatten() makes the code robust to input shape (e.g., (4,) or (4,1)).
    s = sym.flatten()
    x1, x2, x3, x4 = s[0], s[1], s[2], s[3]

    # 3. Construct the STBC matrix.
    # We create the 4x3 NumPy array directly.
    # Note that no complex conjugates are used in this specific code.
    code_matrix = np.array([
        [ x1,  x2,  x3],
        [-x2,  x1, -x4],
        [-x3,  x4,  x1],
        [-x4, -x3,  x2]
    ])
    
    return code_matrix

# --- Example of how to use the function ---
if __name__ == '__main__':
    # Create four example symbols (could be from any constellation like BPSK, QPSK, etc.)
    # Using simple integers for clarity.
    symbols = np.array([1, -1, 1, 1])
    
    # Generate the code matrix
    stbc_matrix = code_4_61(symbols)
    
    print("Input Symbols:\n", symbols)
    print("\nGenerated STBC Matrix (4.61):\n", stbc_matrix)

    # Example with complex symbols
    complex_symbols = np.array([1+1j, -1-1j, 1-1j, -1+1j])
    complex_stbc_matrix = code_4_61(complex_symbols)
    print("\nExample with complex symbols:\n", complex_stbc_matrix)

    # Example of error handling
    try:
        invalid_symbols = np.array([1, 2, 3])
        code_4_61(invalid_symbols)
    except ValueError as e:
        print(f"\nCaught expected error: {e}")