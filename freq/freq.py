from __future__ import division

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scikits.audiolab as al


import audio

def make_heatmap(file, colormap=None, shading=None):
    xs = np.fromfunction(lambda i: file.duration * i/file.nslices, [file.nslices])
    zs = np.zeros([len(xs), len(file.frequencies)])

    for x in range(0, file.nslices):
        zs[x] = file.get_next_sample().get_fft()

    xxs, yys = np.meshgrid(xs,file.frequencies)

    plt.clf()

    print('yys: {0}, xxs: {1}, zs: {2}'.format(yys.shape, xxs.shape, zs.shape))
    nice_cmap = plt.get_cmap(colormap)
    plt.pcolormesh(xxs,yys,zs.transpose(), cmap=nice_cmap, shading=shading)
    plt.axis([xs.min(), xs.max(), file.frequencies.min(), file.frequencies.max()])
    plt.title('Spectrum for {0}'.format(os.path.basename(file.filename)))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()

def cli():
    parser = argparse.ArgumentParser(description='Do some badass spectral analysis.',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--samples-sec', default=15., type=float, help="Number of samples to take per second.  Higher = more resolution.  Default: %(default)s")
    parser.add_argument('-c', '--colormap', choices=[m for m in cm.datad.keys() if not m.endswith("_r")], default='gist_heat', help='Pick your color map')
    parser.add_argument('--shading', choices=['flat', 'gouraud'], default='flat', help="'flat' indicates a solid color for each quad. When 'gouraud', each quad will be Gouraud shaded. When gouraud shading, edgecolors is ignored.")
    parser.add_argument('file', help='File to load')
    options = parser.parse_args()

    with audio.AudioFile(options.file, slices_per_second=options.samples_sec) as file:
        file.printinfo()
        make_heatmap(file, colormap=options.colormap, shading=options.shading)
    
if __name__ == '__main__':
    cli()
