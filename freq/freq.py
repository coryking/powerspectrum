from __future__ import division

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

def make_poly3d(filename, slices=100, frequencies=30):
    data = get_audio_data(filename)
    single_channel = get_channel(data)
    frames_per_slice = int(data['nframes'] / slices)
    print(frames_per_slice)

    freqs = get_freqs(frames_per_slice, data['samplerate'])
    idx = np.argsort(freqs)
    mid = len(idx)/2
    half_freqs = freqs[idx][mid:]
    reduced_freqs = reshape_array(half_freqs, frequencies)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

    verts = []
    zs = np.arange(0, slices - 1)
    max_z = 0
    min_z = 0
    for z in zs:
        offset = frames_per_slice * z
        slice_data = single_channel[offset:offset + frames_per_slice]
        A = np.fft.fft(slice_data) /25.5
        mag = np.abs(np.fft.fftshift(A))
        response = 20 * np.log10(mag[mid:])
        
        max_z = max(response.max(), max_z)
        min_z = min(response.min(), min_z)
        reduced_ps = reshape_array(response, frequencies)
        reduced_ps[0], reduced_ps[-1] = 0,0
        verts.append(list(zip(reduced_freqs, reduced_ps)))
    #print(verts[1])
    print(len(verts))
    poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                               cc('y')])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('Frequency')
    ax.set_xlim3d(reduced_freqs.min(), reduced_freqs.max())
    ax.set_ylabel('Slices')
    ax.set_ylim3d(-1, slices)
    ax.set_zlabel('Power')
    ax.set_zlim3d(min_z, max_z)
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