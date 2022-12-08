import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from scipy.ndimage import (generate_binary_structure,
                                      binary_erosion,
                                      iterate_structure)

def detect_peaks_from_spectrogram(spectrogram):
    """ 
    This code uses the scipy library to detect local maxima in the spectrogram, and converts the detected peaks to 
    strings of frequency and time values.
    It then combines these strings into a single text representation of the spectrogram.
    """
    # Compute the energy of the spectrogram
    energy = np.sum(spectrogram**2, axis=0)

    # Find local maxima of the energy
    neighborhood = generate_binary_structure(len(energy), connectivity=1)
    local_max = maximum_filter(energy, footprint=neighborhood) == energy
    background = (energy == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Only keep peaks that are higher than the background
    peaks = local_max ^ eroded_background

    # Find the indices of the peaks
    peak_indices = np.nonzero(peaks)[0]

    # Convert the peak indices to frequency and time values
    peak_freqs, peak_times = np.meshgrid(range(spectrogram.shape[0]),
                                         range(spectrogram.shape[1]))
    peak_freqs = peak_freqs[peaks]
    peak_times = peak_times[peaks]

    return peak_freqs, peak_times