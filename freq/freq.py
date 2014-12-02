from __future__ import division

import os
import argparse

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

import scikits.audiolab as al


import audio

def _get_plot(frequencies, sample):
    return plt.plot(frequencies, sample,'b-')

def animate_frequency(file, args):
    samples = file.analyze_audio(log_scale=args.log_scale)
    frames = []
    plt.clf()
    fig = plt.figure()
    for x in range(0, len(samples)-1):
        frame = _get_plot(file.frequencies, samples[x])
        frames.append(frame)

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    plt.show()

def plot_frequency(file, args):
    samples = file.analyze_audio(log_scale=args.log_scale)
    sample = samples[args.sample_number]

    plt.clf()
    plt.plot(file.frequencies, sample,'-')
    plt.title('Spectrum for {0}'.format(os.path.basename(file.filename)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.show()

def heatmap(file, args):

    xxs, yys = np.meshgrid(file.sample_times,file.frequencies)
    zs = file.analyze_audio(log_scale=args.log_scale)
    plt.clf()

    print('yys: {0}, xxs: {1}, zs: {2}'.format(yys.shape, xxs.shape, zs.shape))
    nice_cmap = plt.get_cmap(args.colormap)
    plt.pcolormesh(xxs,yys,zs.transpose(), cmap=nice_cmap, shading=args.shading)
    plt.axis([file.sample_times.min(), file.sample_times.max(), file.frequencies.min(), file.frequencies.max()])
    plt.title('Spectrum for {0}'.format(os.path.basename(file.filename)))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()



def cli():
    parser = argparse.ArgumentParser(description='Do some badass spectral analysis.',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--samples-sec', default=15., type=float, help="Number of samples to take per second.  Higher = more resolution.  Default: %(default)s")
    parser.add_argument('-f', '--file', required=True, help='File to load')
    parser.add_argument('--log-scale', action='store_true', help='Plot data using a log scale')
    subparser = parser.add_subparsers(title='Plot Type', help='What kind of graph to draw')
    parser_heatmap = subparser.add_parser('heatmap', help='Draw a heatmap of the file')
    parser_heatmap.add_argument('-c', '--colormap', choices=[m for m in cm.datad.keys() if not m.endswith("_r")], default='gist_heat', help='Pick your color map')
    parser_heatmap.add_argument('--shading', choices=['flat', 'gouraud'], default='flat', help="'flat' indicates a solid color for each quad. When 'gouraud', each quad will be Gouraud shaded. When gouraud shading, edgecolors is ignored.")
    parser_heatmap.set_defaults(func=heatmap)

    parser_plot_freq = subparser.add_parser('plot_frequency', help="Draw a frequency/magnitude graph")
    parser_plot_freq.add_argument('--sample-number', type=int, required=True, help="Which sample to draw the graph for")
    parser_plot_freq.set_defaults(func=plot_frequency)

    parser_animate_frequency = subparser.add_parser('animate_frequency', help="Animate a frequency/magnitude graph")
    parser_animate_frequency.set_defaults(func=animate_frequency)
    options = parser.parse_args()

    with audio.AudioFile(options.file, slices_per_second=options.samples_sec) as file:
        file.printinfo()
        options.func(file, options)

if __name__ == '__main__':
    cli()
