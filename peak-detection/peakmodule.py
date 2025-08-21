import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion

"""
Audio to Text Conversion Module

This module provides functions to convert audio signals to text representations
by detecting peaks in spectrograms and encoding them as frequency-time pairs.
"""

def detect_energy_changes(audio_data, sample_rate=None):
    """
    Detect energy changes in audio by finding local minima and maxima in spectrogram energy.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int, optional): Sample rate of audio. If None, uses default spectrogram parameters
    
    Returns:
        tuple: (minima_indices, maxima_indices) - indices of energy minima and maxima
    """
    # Compute spectrogram with optional sample rate
    if sample_rate:
        freqs, times, spec = signal.spectrogram(audio_data, fs=sample_rate)
    else:
        freqs, times, spec = signal.spectrogram(audio_data)
    
    # Compute energy across frequency bins
    energy = np.sum(spec**2, axis=0)
    
    # Find local extrema
    minima = signal.argrelmin(energy)[0]
    maxima = signal.argrelmax(energy)[0]
    
    return minima, maxima
    
def detect_peaks_from_spectrogram(spectrogram, min_distance=1):
    """
    Detect peaks in a spectrogram using morphological operations.
    
    Args:
        spectrogram (np.ndarray): 2D spectrogram array (freq x time)
        min_distance (int): Minimum distance between peaks
    
    Returns:
        tuple: (peak_frequencies, peak_times, peak_values) - arrays of peak locations and amplitudes
    """
    # Find peaks in the 2D spectrogram
    neighborhood = generate_binary_structure(2, connectivity=1)
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
    
    # Remove background noise
    background = (spectrogram == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    peaks = local_max & ~eroded_background
    
    # Get peak coordinates and values
    peak_coords = np.where(peaks)
    peak_freqs = peak_coords[0]
    peak_times = peak_coords[1]
    peak_values = spectrogram[peaks]
    
    return peak_freqs, peak_times, peak_values

def detect_peaks_from_audio(audio_data, sample_rate=22050, nperseg=256, noverlap=None):
    """
    Detect peaks in audio by computing spectrogram and finding local maxima.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        nperseg (int): Length of each segment for spectrogram
        noverlap (int): Number of points to overlap between segments
    
    Returns:
        tuple: (frequencies, times, peak_freqs, peak_times, peak_values)
               - frequencies: frequency bins
               - times: time bins  
               - peak_freqs: frequency indices of peaks
               - peak_times: time indices of peaks
               - peak_values: amplitude values at peaks
    """
    # Compute spectrogram
    freqs, times, spec = signal.spectrogram(
        audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
    )
    
    # Detect peaks in spectrogram
    peak_freqs, peak_times, peak_values = detect_peaks_from_spectrogram(spec)
    
    return freqs, times, peak_freqs, peak_times, peak_values

def audio_to_text(audio_data, sample_rate=22050, format_type='pairs', separator='|'):
    """
    Convert audio to text representation using peak detection.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        format_type (str): Output format - 'pairs', 'separate', or 'compact'
        separator (str): Character to separate values
    
    Returns:
        str: Text representation of audio peaks
    """
    # Detect peaks
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate
    )
    
    # Convert indices to actual frequency and time values
    actual_freqs = freqs[peak_freqs]
    actual_times = times[peak_times]
    
    # Format output based on type
    if format_type == 'pairs':
        # Format as (freq,time) pairs
        pairs = [f"({freq:.1f},{time:.3f})" for freq, time in zip(actual_freqs, actual_times)]
        return separator.join(pairs)
    elif format_type == 'separate':
        # Separate frequency and time lists
        freq_str = separator.join([f"{f:.1f}" for f in actual_freqs])
        time_str = separator.join([f"{t:.3f}" for t in actual_times])
        return f"FREQ:{freq_str} TIME:{time_str}"
    else:  # compact
        # Simple space-separated values
        values = []
        for freq, time in zip(actual_freqs, actual_times):
            values.extend([f"{freq:.1f}", f"{time:.3f}"])
        return ' '.join(values)

def spectrogram_to_text(spectrogram, freq_bins=None, time_bins=None, format_type='pairs', separator='|'):
    """
    Convert spectrogram to text representation using peak detection.
    
    Args:
        spectrogram (np.ndarray): 2D spectrogram array
        freq_bins (np.ndarray, optional): Frequency bin values
        time_bins (np.ndarray, optional): Time bin values
        format_type (str): Output format - 'pairs', 'separate', or 'compact'
        separator (str): Character to separate values
    
    Returns:
        str: Text representation of spectrogram peaks
    """
    # Detect peaks
    peak_freqs, peak_times, peak_values = detect_peaks_from_spectrogram(spectrogram)
    
    # Use actual frequency/time values if provided, otherwise use indices
    if freq_bins is not None:
        actual_freqs = freq_bins[peak_freqs]
    else:
        actual_freqs = peak_freqs
        
    if time_bins is not None:
        actual_times = time_bins[peak_times]
    else:
        actual_times = peak_times
    
    # Format output
    if format_type == 'pairs':
        pairs = [f"({freq:.1f},{time:.3f})" for freq, time in zip(actual_freqs, actual_times)]
        return separator.join(pairs)
    elif format_type == 'separate':
        freq_str = separator.join([f"{f:.1f}" for f in actual_freqs])
        time_str = separator.join([f"{t:.3f}" for t in actual_times])
        return f"FREQ:{freq_str} TIME:{time_str}"
    else:  # compact
        values = []
        for freq, time in zip(actual_freqs, actual_times):
            values.extend([f"{freq:.1f}", f"{time:.3f}"])
        return ' '.join(values)
