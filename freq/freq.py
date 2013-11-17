from __future__ import division

import os

import argparse

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

def make_heatmap(filename, samples_per_second=10., frequencies=30):
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

def cli():
    parser = argparse.ArgumentParser(description='Do some badass spectral analysis.',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--samples-sec', default=10., type=float, help="Number of samples to take per second.  Higher = more resolution.  Default: %(default)s")
    parser.add_argument('-f', '--frequencies', default=30, type=int, help="Number of frequency bands to measure.  More is better.  Default: %(default)s")
    parser.add_argument('file', help='File to load')
    options = parser.parse_args()
    
    make_heatmap(options.file, options.samples_sec, options.frequencies)
    
if __name__ == '__main__':
    cli()
