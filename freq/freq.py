from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def get_audio_data(filename):
    file = al.Sndfile(filename)
    data = file.read_frames(file.nframes)
    data_hash =  {
        "frames": data,
        "samplerate": file.samplerate,
        "channels": file.channels
    }
    file.close()
    return data_hash

def get_slice(data, frames, offset=0):
    return {
        "frames": data['frames'][offset:offset+frames],
        "samplerate": data['samplerate'],
        "channels": data['channels']
    }
def get_channel(data, channel=0):

    if data['channels'] == 2:
        return data['frames'][:,channel]
    else:
        return data['frames']

def get_powerband(frames, samplerate):
    ps = np.abs(np.fft.fft(frames))**2

    time_step = 1. / samplerate
    freqs = np.fft.fftfreq(frames.size, time_step)
    idx = np.argsort(freqs)

    mid = len(idx)/2
    
    return {
        'freq': freqs[idx][mid:],
        'power': np.log10(ps[idx][mid:])
    }

def load_and_plot(filename):
    data = get_audio_data(filename)
    frames = get_channel(data)
    pb = get_powerband(frames, data['samplerate'])
    plt.clf()
    plt.plot(pb['freq'], pb['power'])
    plt.show()