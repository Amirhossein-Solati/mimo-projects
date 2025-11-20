"""
A collection of helper functions for Space-Time Block Code (STBC) simulations.
This module includes functions for modulation, demodulation, STBC encoding,
and Maximum Likelihood (ML) detection.
"""

import numpy as np
from itertools import product

# --- Modulation and Demodulation Functions ---

def myMod_QPSK(x):
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

def myDemod_QPSK(x):
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

def myAlamouti(x):
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

def myMLD_BPSK_SISO(r, H):
    """ML Detector for SISO BPSK."""
    sx = np.array([1, -1])
    # Simplified detector: project received signal onto the channel vector
    decision_metric = np.real(r * np.conj(H))
    return 1 if decision_metric > 0 else -1

def myMLD_Alamouti_BPSK(r, H):
    """ML Detector for Alamouti STBC with BPSK modulation."""
    sx = np.array([1, -1])
    possible_pairs = product(sx, repeat=2)
    min_cost = np.inf
    detected_symbols = None
    for pair in possible_pairs:
        C = myAlamouti(np.array(pair))
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

def myMLD_4_61_BPSK(r, H):
    """ML Detector for code_4_61 with BPSK."""
    return _generic_mld(r, H, code_4_61, np.array([1, -1]), 4)

def myMLD_4_38_BPSK(r, H):
    """ML Detector for code_4_38 with BPSK."""
    return _generic_mld(r, H, code_4_38, np.array([1, -1]), 4)

def myMLD_QPSK_G348(r, H):
    """ML Detector for G348 with QPSK."""
    return _generic_mld(r, H, code_G348, np.array([1, -1, 1j, -1j]), 4)

def myMLD_QPSK_G448(r, H):
    """ML Detector for G448 with QPSK."""
    return _generic_mld(r, H, code_G448, np.array([1, -1, 1j, -1j]), 4)