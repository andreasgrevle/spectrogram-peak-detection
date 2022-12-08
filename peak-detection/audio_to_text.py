import detect_peaks_from_audio

def audio_to_text(audio_data):
    # Detect peaks in the audio data
    peak_freqs, peak_times = detect_peaks_from_audio(audio_data)

    # Convert the peak frequencies and times to strings
    peak_freqs_str = [str(freq) for freq in peak_freqs]
    peak_times_str = [str(time) for time in peak_times]

    # Join the strings into a single text representation of the audio data
    text = ' '.join(peak_freqs_str + peak_times_str)

    return text