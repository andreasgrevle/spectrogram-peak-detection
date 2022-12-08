import detect_peaks_from_spectrogram

def spectrogram_to_text(spectrogram):
    # Detect peaks in the spectrogram
    peak_freqs, peak_times = detect_peaks_from_spectrogram(spectrogram)

    # Convert the peak frequencies and times to strings
    peak_freqs_str = [str(freq) for freq in peak_freqs]
    peak_times_str = [str(time) for time in peak_times]

    # Join the strings into a single text representation of the spectrogram
    text = ' '.join(peak_freqs_str + peak_times_str)

    return text