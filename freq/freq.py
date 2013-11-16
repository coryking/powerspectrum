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
        "nframes": file.nframes,
        "samplerate": file.samplerate,
        "channels": file.channels
    }
    file.close()
    return data_hash

def get_frame_slice(data, frames, offset=0):
    return data['frames'][offset:offset+frames]
def get_slice(data, frames, offset=0):
    return {
        "frames": get_frame_slice(data, frames,offset),
        "samplerate": data['samplerate'],
        "channels": data['channels']
    }
def get_channel(data, channel=0):

    if data['channels'] == 2:
        return data['frames'][:,channel]
    else:
        return data['frames']

def get_freqs(frame_count, samplerate):
    return np.fft.fftfreq(frame_count, 1. / samplerate)


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

def make_heatmap(filename, slices=1000):
    data = get_audio_data(filename)
    frames_per_slice = int(data['nframes'] / slices)
    print(frames_per_slice)
    x = np.arange(0, slices - 1)
    freqs = get_freqs(frames_per_slice, data['samplerate'])
    idx = np.argsort(freqs)
    mid = len(idx)/2
    half_freqs = freqs[idx][mid:]

    z = np.zeros([len(x), len(half_freqs),2])
    for slice in x:
        slice_data = get_frame_slice(data, frames_per_slice, frames_per_slice * slice)
        ps = np.abs(np.fft.fft(slice_data))**2
        ps_adj = np.log10(ps[idx][mid:])
        for freq in half_freqs:
            print(ps_adj[freq])
            z[slice][freq] = ps_adj[freq]
    plt.clf()
    plt.pcolormesh(x, half_freqs, z)
    plt.show()

def load_and_plot(filename):
    data = get_audio_data(filename)
    frames = get_channel(data)
    pb = get_powerband(frames, data['samplerate'])
    plt.clf()
    plt.plot(pb['freq'], pb['power'])
    plt.show()