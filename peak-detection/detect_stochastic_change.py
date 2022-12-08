import numpy as np
from scipy import signal

def detect_stochastic_changes(audio_data):
    """ 
    This code uses the scipy library to compute the spectrogram of the audio data and detect changes in the 
    energy of the spectrogram. 

    It returns the detected changes as the indices of local minima and maxima in the energy sequence
    """
    # Compute the spectrogram of the audio data
    freqs, times, spec = signal.spectrogram(audio_data)

    # Compute the energy of the spectrogram
    energy = np.sum(spec**2, axis=0)

    # Detect changes in the energy by looking for local minima and maxima
    minima = signal.argrelmin(energy)[0]
    maxima = signal.argrelmax(energy)[0]

    # Return the detected changes
    return minima, maxima