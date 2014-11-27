from __future__ import division

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scikits.audiolab as al


import audio
import analysis

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

def make_heatmap(filename, slices_per_second=10., colormap=None):
    with audio.AudioFile(filename, slices_per_second) as file:
        freqs = get_freqs(file.frames_per_slice, file.samplerate)

        idx = np.argsort(freqs)
        mid = len(idx)/2
        ys = freqs[idx][mid:]
        file.printinfo()
        xs = np.fromfunction(lambda i: file.duration * i/file.nslices, [file.nslices])
        zs = np.zeros([len(xs), len(ys)])
        xxs, yys = np.meshgrid(xs,ys)
        for x in range(0, file.nslices):
            slice_data = file.read_next_slice()
            spectral_data = analysis.analyze_sample(slice_data)

            zs[x] = analysis.use_db_scale(spectral_data, mid)

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
