__author__ = 'cory'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def analyze_sample(sample_data):
    """
    Perform an FFT on some sample data.
    :param sample_data:
    :return:
    """
    A = np.fft.fft(sample_data) ** 2 #/25.5
    mag = np.abs(np.fft.fftshift(A))
    return mag

def use_db_scale(powerband_data, midpoint):
    """
    Make the output of an FFT use a logarithmic, dB scale
    :param powerband_data:
    :return:
    """
    return 20 * np.log10(powerband_data[midpoint:])
