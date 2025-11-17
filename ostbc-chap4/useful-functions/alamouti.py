import numpy as np

def alamouti(x):
    """
    Alamouti Space-Time Block Code matrix

    This functions take two input symbols and map them to alamouti based 
    coding(matrix)
    Alamouti : Transmission from 2 antennas over two time slot

    Parameters
    ----------
    x : np.ndarray
        A (2,1) matrix of symbols that must be encoded in alamouti format

    Returns
    -------
    np.ndarray
        T 2X2 complex value matrix (Alamouti Code)

    Raises
    ------
    ValueError
        When the input doesn't contain exact 2 elements
    """

    if np.size(x) != 2:
        raise ValueError(f'The input must be exactly two symbols, but got {np.size(x)}')
    
    # Assign exact first and second input symbols to s1 and s2
    s1 = x.flatten()[0]
    s2 = x.flatten()[1]

    alamouti_matrix = np.array([
        [s1,            s2],
        [-np.conj(s2),  np.conj(s1)]
    ])

    return alamouti_matrix

if __name__ == "__main__":
    # Input Symbol
    input_symbol_complex = np.array([1+3j, -4+4j])
    input_symbol_real    = np.array([1, 3])
    not_correct_dims     = np.array([1,3,4])

    # Generate Alamouti Matrix
    alamouti_complex = alamouti(input_symbol_complex)
    alamouti_real    = alamouti(input_symbol_real)

    print(f"Complex Alamouti coded: {alamouti_complex}")
    print(f"real Alamouti coded: {alamouti_real}")

    try:
        alamouti(not_correct_dims)
    except ValueError as e:
        print(f"Error Raised because of dimention: {e}")

    



