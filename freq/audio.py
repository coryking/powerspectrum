__author__ = 'cory'

import scikits.audiolab as al

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