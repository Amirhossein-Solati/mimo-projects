import numpy as np

def ml_bpsk(r,H):
    """
    This Function is for a maximum likelihood detector in SISO system
    with BPSK modulation.

    If we have the recieved signal that means `r` then with channel gain `H`
    we can detect which signal sent by trasmitter by calculating the minimum euclidean
    distance

    Parameters:
    -----------
    r : np.ndarray
        The recieved signal which is complex and 1X1 like 0.5+j0.2
    H : np.ndarray
        The Channel gain which is 1X1 and is complex

    Returns
    -------
    int
        The detected -1 or 1 from bpsk symbol
    """

    # Constellation for bpsk
    sx = np.array([1,-1])

    # Define cost function must be minimized and that is Euclidean distance
    cost_function = np.abs(r - sx.reshape(-1,1)*H)**2

    # Find minimum value's index
    min_index = np.argmin(cost_function)

    detected_symbol = sx[min_index]

    return detected_symbol

# Test Case

if __name__ == "__main__":
    
    actual_symbol = -1
    H = np.array([0.4 + 0.1j])
    noise = (np.random.randn() + 1j*np.random.randn()) * 0.1
    r = noise + actual_symbol*H
    detected_symbol = ml_bpsk(r,H)

    print(f"The detected BPSK symbol is : {detected_symbol}")
