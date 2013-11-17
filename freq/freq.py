from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
import scikits.audiolab as al

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

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
def get_seconds(nframes, samplerate):
    return nframes / samplerate
def get_frames_per_sample(nframes, samplerate, samples_per_second = 10.):
    """Number of samples for each slice"""
    return int(samplerate / samples_per_second)
def get_total_samples(nframes, samplerate, samples_per_second = 10.):
    return int(samples_per_second * (nframes / samplerate))

def get_freqs(frame_count, samplerate):
    return np.fft.fftfreq(frame_count, 1. / samplerate)

def get_powerband(frames, samplerate):
    ps = np.abs(np.fft.fft(frames))**2

    time_step = 1. / samplerate
    freqs = np.fft.fftfreq(frames.size, time_step)
    idx = np.argsort(freqs)

    mid = len(idx)/2
    print("freq: {0}, ps: {1}".format(len(idx), len(ps)))
    return {
        'freq': freqs[idx][mid:],
        'power': np.log10(ps[idx][mid:])
    }
def reshape_array(data, frequencies):
    bw = len(data) / frequencies
    resized_data = data.copy()
    resized_data.resize([frequencies, bw])
    return np.average(resized_data, axis=1)

def make_poly3d(filename, samples_per_second=10., frequencies=30):
    data = get_audio_data(filename)
    single_channel = get_channel(data)
    slices = get_total_samples(data['nframes'], data['samplerate'], samples_per_second)
    frames_per_slice = get_frames_per_sample(data['nframes'], data['samplerate'], samples_per_second)
    print(frames_per_slice)

    freqs = get_freqs(frames_per_slice, data['samplerate'])
    idx = np.argsort(freqs)
    mid = len(idx)/2
    half_freqs = freqs[idx][mid:]
    reduced_freqs = reshape_array(half_freqs, frequencies)
    total_seconds = get_seconds(data['nframes'], data['samplerate'])
    print("Total seconds: %s" % total_seconds)
    ys = reshape_array(half_freqs, frequencies)
    xs = np.fromfunction(lambda i: total_seconds * i/slices, [slices] )#np.arange(0, slices)
    zs = np.zeros([len(xs), len(ys)])
    xxs, yys = np.meshgrid(xs,ys)
    for x in range(0, slices):
        offset = frames_per_slice * x
        slice_data = single_channel[offset:offset + frames_per_slice]
        A = np.fft.fft(slice_data) /25.5
        mag = np.abs(np.fft.fftshift(A))
        response = 20 * np.log10(mag[mid:])
        reduced_ps = reshape_array(response, frequencies)
        zs[x] = reduced_ps
    plt.clf()

    print('yys: {0}, xxs: {1}, zs: {2}'.format(yys.shape, xxs.shape, zs.shape))
    plt.pcolormesh(xxs,yys,zs.transpose())
    plt.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    plt.title('Spectrum for {0}'.format(os.path.basename(filename)))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()

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
            print("x: {0}, y: {1}, z: {2}".format(slice, freq, ps_adj[freq]))
            z[slice][freq] = ps_adj[freq]
    plt.clf()
    plt.pcolormesh(x, half_freqs, z)
    plt.show()

def load_and_plot(filename):
    data = get_audio_data(filename)
    frames = get_channel(data)
    pb = get_powerband(frames, data['samplerate'])
    plt.clf()
    print("x: {0}, y: {1}".format(pb['freq'], pb['power']))
    plt.plot(pb['freq'], pb['power'])
    plt.show()