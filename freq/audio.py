__author__ = 'cory'

import scikits.audiolab as al

class AudioData:
    def __init__(self, filename, slices_per_second=10.):
        self.slices_per_second = slices_per_second
        self._file = al.Sndfile(filename)
        self.nframes = self._file.nframes
        self.samplerate = self._file.samplerate
        self.channels = self._file.channels
        self.duration = self.nframes / self.samplerate
        self.frames_per_slice = int(self.samplerate / self.slices_per_second)
        self.nslices = int(self.slices_per_second * (self.nframes / self.samplerate))

    def read_next_slice(self, channel=0):
        return self.read_frames(self.frames_per_slice, channel=channel)

    def read_frames(self, frames, channel=0):
        frame_data = self._file.read_frames(frames)
        if self.channels == 2:
            return frame_data[:channel]
        else:
            return frame_data

    def printinfo(self):
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

def get_audio_data(filename):
    file = al.Sndfile(filename)
    data = file.read_frames(file.nframes)
    data_hash =  {
        "frames": data,
        "nframes": file.nframes,
        "samplerate": file.samplerate,
        "channels": file.channels,
        "duration": file.nframes / file.samplerate,
    }
    file.close()
    return data_hash

def get_total_samples(nframes, samplerate, samples_per_second = 10.):
    return int(samples_per_second * (nframes / samplerate))

def get_frames_per_sample(nframes, samplerate, samples_per_second = 10.):
    """Number of samples for each slice"""
    return int(samplerate / samples_per_second)

def get_data_from_channel(data, channel=0):

    if data['channels'] == 2:
        return data['frames'][:,channel]
    else:
        return data['frames']