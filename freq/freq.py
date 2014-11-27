from __future__ import division

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scikits.audiolab as al


import audio

def get_frame_slice(data, frames, offset=0):
    return data['frames'][offset:offset+frames]

def get_slice(data, frames, offset=0):
    return {
        "frames": get_frame_slice(data, frames,offset),
        "samplerate": data['samplerate'],
        "channels": data['channels']
    }


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

def make_heatmap(filename, slices_per_second=10., colormap=None):
    with audio.AudioData(filename, slices_per_second) as file:
        freqs = get_freqs(file.frames_per_slice, file.samplerate)

        idx = np.argsort(freqs)
        mid = len(idx)/2
        ys = freqs[idx][mid:]
        file.printinfo()
        xs = np.fromfunction(lambda i: file.duration * i/file.nslices, [file.nslices])
        zs = np.zeros([len(xs), len(ys)])
        xxs, yys = np.meshgrid(xs,ys)
        for x in range(0, file.nslices):
            print("Doing slice %s" % x)
            slice_data = file.read_next_slice() #single_channel[offset:offset + frames_per_slice]
            A = np.fft.fft(slice_data) ** 2 #/25.5
            mag = np.abs(np.fft.fftshift(A))
            response = 20 * np.log10(mag[mid:])
            reduced_ps = response
            zs[x] = reduced_ps

        plt.clf()

        print('yys: {0}, xxs: {1}, zs: {2}'.format(yys.shape, xxs.shape, zs.shape))
        nice_cmap = plt.get_cmap(colormap)
        plt.pcolormesh(xxs,yys,zs.transpose(), cmap=nice_cmap)
        plt.axis([xs.min(), xs.max(), ys.min(), ys.max()])
        plt.title('Spectrum for {0}'.format(os.path.basename(filename)))
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
        plt.show()

def _old_read_heatmap(filename, samples_per_second=10., colormap=None):
    data = audio.get_audio_data(filename)
    single_channel = audio.get_data_from_channel(data)
    slices = audio.get_total_samples(data['nframes'], data['samplerate'], samples_per_second)
    frames_per_slice = audio.get_frames_per_sample(data['nframes'], data['samplerate'], samples_per_second)
    print(frames_per_slice)

    freqs = get_freqs(frames_per_slice, data['samplerate'])
    idx = np.argsort(freqs)
    mid = len(idx)/2
    ys = freqs[idx][mid:]
    total_seconds = data['duration']
    print("Total seconds: %s" % total_seconds)
    xs = np.fromfunction(lambda i: total_seconds * i/slices, [slices] )
    zs = np.zeros([len(xs), len(ys)])
    xxs, yys = np.meshgrid(xs,ys)
    for x in range(0, slices):
        offset = frames_per_slice * x
        slice_data = single_channel[offset:offset + frames_per_slice]
        A = np.fft.fft(slice_data) ** 2 #/25.5
        mag = np.abs(np.fft.fftshift(A))
        response = 20 * np.log10(mag[mid:])
        reduced_ps = response
        zs[x] = reduced_ps
    plt.clf()

    print('yys: {0}, xxs: {1}, zs: {2}'.format(yys.shape, xxs.shape, zs.shape))
    nice_cmap = plt.get_cmap(colormap)
    plt.pcolormesh(xxs,yys,zs.transpose(), cmap=nice_cmap)
    plt.axis([xs.min(), xs.max(), ys.min(), ys.max()])
    plt.title('Spectrum for {0}'.format(os.path.basename(filename)))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()

def cli():
    parser = argparse.ArgumentParser(description='Do some badass spectral analysis.',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--samples-sec', default=15., type=float, help="Number of samples to take per second.  Higher = more resolution.  Default: %(default)s")
    parser.add_argument('-c', '--colormap', choices=[m for m in cm.datad.keys() if not m.endswith("_r")], default='gist_heat', help='Pick your color map')
    parser.add_argument('file', help='File to load')
    options = parser.parse_args()
    
    make_heatmap(options.file, options.samples_sec, colormap=options.colormap)
    
if __name__ == '__main__':
    cli()
