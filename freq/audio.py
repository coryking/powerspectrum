__author__ = 'cory'

import scikits.audiolab as al

import numpy as np

import analysis

class AudioFile:
    def __init__(self, filename, slices_per_second=10.):
        self.slices_per_second = slices_per_second
        self._file = al.Sndfile(filename)
        self.nframes = self._file.nframes
        self.samplerate = self._file.samplerate
        self.channels = self._file.channels
        self.duration = self.nframes / self.samplerate
        self.frames_per_slice = int(self.samplerate / self.slices_per_second)
        self.nslices = int(self.slices_per_second * (self.nframes / self.samplerate))
        self.frequencies = self._get_sample_frequencies()
    def _get_sample_frequencies(self):
        frequencies = analysis.get_sample_frequencies(self.frames_per_slice, self.samplerate)
        idx = np.argsort(frequencies)
        mid = len(idx)/2
        return frequencies[idx][mid:]

    def get_next_sample(self):
        data = self.read_frames(self.frames_per_slice)
        return AudioSample(data, self.frames_per_slice, self.channels, self.samplerate, self.frequencies)

    def read_frames(self, frames, channel=0):
        return self._file.read_frames(frames)

    def printinfo(self):
        print("Frequencies: %s" % len(self.frequencies))
        print("Channels: %s" % self.channels)
        print("Total seconds: %s" % self.duration)
        print("Slices: %s" % self.nslices)
        print("Frames: %s" % self.nframes)
        print("Frames / Slice: %s" % self.frames_per_slice)
        print("Slices / Second: %s" % self.slices_per_second)
        print("Frames / Second: %s" % self.samplerate)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

class AudioSample:
    def __init__(self, data, nframes, channels, sample_rate, frequencies):
        self._data = data
        self._nframes = nframes
        self._channels = channels
        self._sample_rate = sample_rate
        self._frequencies = frequencies

    def get_sample_frequencies(self):
        """
        Return the Discrete Fourier Transform sample frequencies.
        :return:
        """
        return self._frequencies

    def get_fft(self, channel=0, log_scale=True):
        data = self._get_chanel_data(channel)
        fft_data = analysis.analyze_sample(data)
        if log_scale:
            return analysis.use_db_scale(fft_data, len(self._frequencies)-1)
        else:
            return fft_data

    def _get_chanel_data(self, channel=0):
        if self._channels > 1:
            return self._data[:,channel]
        else:
            return self._data
