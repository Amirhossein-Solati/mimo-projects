"""
A collection of helper functions for Space-Time Block Code (STBC) simulations.
This module includes functions for modulation, demodulation, STBC encoding,
and Maximum Likelihood (ML) detection.
"""

import numpy as np
from itertools import product

# --- Modulation and Demodulation Functions ---

def mod_qpsk(x):
    """Modulates a sequence of bits into QPSK symbols."""
    if np.size(x) % 2 != 0:
        raise ValueError("Input array must have an even number of bits.")
    
    bits = x.flatten()
    constellation_map = {(0, 0): 1, (0, 1): 1j, (1, 0): -1j, (1, 1): -1}
    bit_pairs = bits.reshape(-1, 2)
    
    num_symbols = len(bit_pairs)
    symbols = np.zeros(num_symbols, dtype=np.complex128)
    
    for i in range(num_symbols):
        pair = tuple(bit_pairs[i])
        symbols[i] = constellation_map[pair]
        
    return symbols

def demod_qpsk(x):
    """Demodulates a sequence of QPSK symbols back into a stream of bits."""
    symbol_to_bit_map = {1: [0, 0], 1j: [0, 1], -1: [1, 1], -1j: [1, 0]}
    output_bits = []
    for symbol in x.flatten():
        bit_pair = symbol_to_bit_map.get(np.round(symbol.real) + 1j * np.round(symbol.imag))
        if bit_pair is not None:
            output_bits.extend(bit_pair)
        else: # Fallback for noisy symbols
             # Find the closest constellation point
            distances = {np.abs(symbol - const_point): bits for const_point, bits in symbol_to_bit_map.items()}
            closest = min(distances.keys())
            output_bits.extend(distances[closest])

    return np.array(output_bits, dtype=int).reshape(-1, 1)

# --- STBC Encoding Functions ---

def alamouti(x):
    """Constructs the 2x2 Alamouti STBC matrix."""
    if np.size(x) != 2:
        raise ValueError("Input for Alamouti must contain exactly 2 symbols.")
    s1, s2 = x.flatten()
    return np.array([[s1, s2], [-np.conj(s2), np.conj(s1)]])

def code_4_61(sym):
    """Constructs the Rate-1 STBC for 3 Tx antennas."""
    if np.size(sym) != 4:
        raise ValueError("Input for code_4_61 must contain exactly 4 symbols.")
    x1, x2, x3, x4 = sym.flatten()
    return np.array([[x1, x2, x3], [-x2, x1, -x4], [-x3, x4, x1], [-x4, -x3, x2]])

def code_4_38(sym):
    """Constructs the Rate-1 Orthogonal STBC for 4 Tx antennas."""
    if np.size(sym) != 4:
        raise ValueError("Input for code_4_38 must contain exactly 4 symbols.")
    x1, x2, x3, x4 = sym.flatten()
    return np.array([[x1, x2, x3, x4], [-x2, x1, -x4, x3], [-x3, x4, x1, -x2], [-x4, -x3, x2, x1]])

def code_G448(sym):
    """Constructs the Rate-1/2 G448 STBC for 4 Tx antennas."""
    g0_matrix = code_4_38(sym)
    return np.vstack((g0_matrix, np.conj(g0_matrix)))

def code_G348(sym):
    """Constructs the G348 STBC for 3 Tx antennas by puncturing G448."""
    g448_matrix = code_G448(sym)
    return g448_matrix[:, 0:3]

# --- ML Detection Functions ---

def ml_bpsk_siso(r, H):
    """
    ML Detector for a SIMO/SISO system with BPSK modulation.
    This implementation uses Maximal Ratio Combining (MRC) for the SIMO case.

    Parameters
    ----------
    r : np.ndarray
        The received signal vector/matrix, shape (N, M). For SIMO, (1, M).
    H : np.ndarray
        The channel vector/matrix, shape (N, M). For SIMO, (1, M).

    Returns
    -------
    int
        The detected BPSK symbol, either 1 or -1.
    """
    # 1. Perform Maximal Ratio Combining (MRC)
    # This involves multiplying the received signal element-wise with the
    # complex conjugate of the channel and then summing the results.
    # This is equivalent to the dot product of r and the conjugate of H.
    # r is (1, M), H is (1, M). We need the Hermitian product.
    # H.conj().T gives the Hermitian transpose, shape (M, 1).
    # r @ H.conj().T performs the matrix multiplication (1, M) @ (M, 1) -> (1, 1) scalar
    
    # A simpler way for vectors is using np.vdot for complex dot product
    # np.vdot(H, r) calculates sum(H_i^* * r_i) which is exactly what we need.
    combined_signal = np.vdot(H, r)
    
    # 2. Make the decision based on the real part of the combined signal.
    # The result of vdot is a scalar complex number.
    decision_metric = np.real(combined_signal)
    
    # 3. Return the detected symbol.
    return 1 if decision_metric > 0 else -1

def ml_alamouti_bpsk(r, H):
    """ML Detector for Alamouti STBC with BPSK modulation."""
    sx = np.array([1, -1])
    possible_pairs = product(sx, repeat=2)
    min_cost = np.inf
    detected_symbols = None
    for pair in possible_pairs:
        C = alamouti(np.array(pair))
        cost = np.linalg.norm(r - (C @ H), 'fro')**2
        if cost < min_cost:
            min_cost = cost
            detected_symbols = np.array(pair)
    return detected_symbols.reshape(2, 1)

def _generic_mld(r, H, code_func, constellation, num_symbols):
    """A generic ML detector to reduce code duplication."""
    possible_combinations = product(constellation, repeat=num_symbols)
    min_cost = np.inf
    detected_symbols = None
    for combo in possible_combinations:
        current_symbols = np.array(combo)
        C = code_func(current_symbols)
        cost = np.linalg.norm(r - (C @ H), 'fro')**2
        if cost < min_cost:
            min_cost = cost
            detected_symbols = current_symbols
    return detected_symbols.reshape(num_symbols, 1)

def ml_4_61_bpsk(r, H):
    """ML Detector for code_4_61 with BPSK."""
    return _generic_mld(r, H, code_4_61, np.array([1, -1]), 4)

def ml_4_38_bpsk(r, H):
    """ML Detector for code_4_38 with BPSK."""
    return _generic_mld(r, H, code_4_38, np.array([1, -1]), 4)

def ml_qpsk_G348(r, H):
    """ML Detector for G348 with QPSK."""
    return _generic_mld(r, H, code_G348, np.array([1, -1, 1j, -1j]), 4)

def ml_qpsk_G448(r, H):
    """ML Detector for G448 with QPSK."""
    return _generic_mld(r, H, code_G448, np.array([1, -1, 1j, -1j]), 4)

def ml_qostbc_bpsk_5_2(r0, H, phi, signal_amp):
    """
    Maximum Likelihood Detector for the QOSTBC from equation (5.2) with BPSK.

    This function performs an exhaustive search over all 16 possible combinations
    of four BPSK symbols to find the most likely transmitted sequence.

    Parameters
    ----------
    r0 : np.ndarray
        The received signal vector/matrix.
    H : np.ndarray
        The channel matrix.
    phi : float
        The rotation angle used at the transmitter.
    signal_amp : float
        The signal amplitude scaling factor (sqrt(SNR/N)).

    Returns
    -------
    np.ndarray
        A column vector of shape (4, 1) containing the four detected BPSK symbols.
    """
    # 1. Define the BPSK constellation.
    sx = np.array([1, -1])

    # 2. Generate an iterator for all 16 possible combinations of four BPSK symbols.
    possible_symbol_quads = product(sx, repeat=4)

    # Variables to track the best match.
    min_cost = np.inf
    detected_symbols = None

    # 3. Iterate through each of the 16 possible symbol combinations.
    for symbol_quad in possible_symbol_quads:
        # a. Convert the current combination (a tuple) to a NumPy array.
        current_symbols = np.array(symbol_quad).reshape(4, 1)
        
        # b. Create the hypothetical STBC matrix for this combination.
        # Note: This is different from previous detectors. The signal amplitude
        # is part of the hypothetical signal calculation before the channel.
        C = code_qostbc_5_2(current_symbols, phi)
        
        # c. Construct the full hypothetical noise-free received signal.
        hypothetical_r = signal_amp * (C @ H)
        
        # d. Calculate the cost (squared Frobenius norm of the error).
        cost = np.linalg.norm(r0 - hypothetical_r, 'fro')**2
        
        # e. If this cost is the smallest yet, update our best guess.
        if cost < min_cost:
            min_cost = cost
            detected_symbols = current_symbols

    # 4. Return the detected symbols (already a column vector).
    return detected_symbols

def code_qostbc_5_2(sym, phi):
    """
    Generates the Quasi-Orthogonal STBC matrix from equation (5.2).

    This code is constructed hierarchically using two Alamouti blocks.
    It takes 4 BPSK symbols and applies a phase rotation 'phi' to the
    second pair of symbols to create the quasi-orthogonal structure.

    Parameters
    ----------
    sym : np.ndarray
        A column vector of shape (4, 1) containing four BPSK symbols.
    phi : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The 4x4 complex-valued QOSTBC transmission matrix.
    """
    # 1. Input Validation
    if np.size(sym) != 4:
        raise ValueError("Input for QOSTBC must contain exactly 4 symbols.")
    
    # Flatten the input array for easy access
    s = sym.flatten()

    # 2. Apply phase rotation to the second pair of symbols.
    # The rotation factor is e^(j*phi). In Python, this is np.exp(1j * phi).
    rotation = np.exp(1j * phi)
    x1 = s[0]
    x2 = s[1]
    x3 = s[2] * rotation
    x4 = s[3] * rotation

    # 3. Create the two inner Alamouti blocks.
    g1 = alamouti(np.array([x1, x2]))
    g2 = alamouti(np.array([x3, x4]))
    
    # 4. Construct the final 4x4 matrix using the Alamouti-like structure.
    # np.block is a very convenient function for building matrices from blocks.
    # It's more readable than manually concatenating rows and columns.
    top_row = np.hstack((g1, g2))
    bottom_row = np.hstack((-np.conj(g2), np.conj(g1)))
    
    z = np.vstack((top_row, bottom_row))
    
    # Alternative and more concise way using np.block:
    # z = np.block([
    #     [g1,           g2],
    #     [-np.conj(g2), np.conj(g1)]
    # ])
    
    return z