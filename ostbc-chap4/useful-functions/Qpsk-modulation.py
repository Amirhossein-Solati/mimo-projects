import numpy as np

def qpsk_modulation(bit_series):
    """
    This function get input array and take pairs out of it and maps to QPSK
    constellation points

    Parameters
    ----------
    bit_series : np.ndarray
        1D array of 0 or 1 (That bits)
        this must be (N,1) or (1,N) array

    Returns
    -------
    np.ndarray
        The output is a 1D array of complex QPSK symbols
        The lenght must be half of the bit_series

    Raises
    ------
    ValueError
        If the bit_series has an odd numbers of bits
    """

    # Ensure of an even number of bits as input
    num_bit = np.size(bit_series)
    if num_bit % 2 != 0:
        raise ValueError('Input must have an even number of bits.')
    
    #Ensure that bits are 1D array
    bit = bit_series.flatten()

    # QPSK constellation
    constellation_map = {
        (0,0) : 1,
        (0,1) : 1j,
        (1,0) : -1j,
        (1,1) : -1
    }

    # Take pair bits from bit stream like (0,0)
    bit_pairs = bit.reshape(-1,2)

    # Define number of pairs that be generated from bit stream
    num_symbols = num_bit // 2
    symbols = np.zeros(num_symbols,dtype=np.complex128)

    # Now we have to map each symbols to contellation of QPSK
    # at each iteration, mapped value append to symbol
    for i in range(num_symbols):
        pair = tuple(bit_pairs[i])
        symbols[i] = constellation_map[pair]

    return symbols

# Test Case
if __name__ == '__main__':
    # Input
    bit_series = np.array([0, 1, 0, 1, 0, 1, 1, 1])

    # Modulated output
    modulated_symbols = qpsk_modulation(bit_series)

    print(f"Bit Strem is : {bit_series}")
    print(f"QPSK modulation : {modulated_symbols}")

    try:
        invalid_bit = np.array([1,0,1])
        qpsk_modulation(invalid_bit)
    except ValueError as e:
        print(f"error value: {e}")