#!/usr/bin/env python3
"""
Audio to Text Conversion - Main Example Script

This script demonstrates how to use the peak detection module to convert
audio signals to text representations.
"""

import numpy as np
from scipy.io import wavfile
import sys
import os

# Add the peak-detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'peak-detection'))
from peakmodule import audio_to_text, detect_peaks_from_audio, detect_energy_changes


def load_audio_file(filepath):
    """Load audio file and return sample rate and data."""
    try:
        sample_rate, audio_data = wavfile.read(filepath)
        # Convert to float and normalize if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Handle stereo by taking first channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
            
        return sample_rate, audio_data
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None


def generate_test_signal(duration=2.0, sample_rate=22050):
    """Generate a test audio signal with multiple frequency components."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a signal with multiple frequency components
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +      # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t) +     # E6 note
        0.1 * np.random.randn(len(t))             # Some noise
    )
    
    # Add some amplitude modulation
    envelope = np.exp(-t / 2.0)  # Decay envelope
    signal *= envelope
    
    return signal


def main():
    """Main demonstration function."""
    print("Audio to Text Conversion Demo")
    print("=" * 40)
    
    # Check if audio file is provided
    audio_file = "audiosample.wav"
    if os.path.exists(audio_file):
        print(f"Loading audio file: {audio_file}")
        sample_rate, audio_data = load_audio_file(audio_file)
        
        if audio_data is None:
            print("Failed to load audio file, using generated test signal")
            audio_data = generate_test_signal()
            sample_rate = 22050
    else:
        print("No audio file found, generating test signal")
        audio_data = generate_test_signal()
        sample_rate = 22050
    
    print(f"Audio length: {len(audio_data)} samples")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    print()
    
    # Demonstrate different text formats
    print("Converting audio to text in different formats:")
    print("-" * 50)
    
    # Format 1: Frequency-time pairs
    text_pairs = audio_to_text(audio_data, sample_rate, format_type='pairs', separator=' | ')
    print("Format 'pairs':")
    print(text_pairs[:200] + "..." if len(text_pairs) > 200 else text_pairs)
    print()
    
    # Format 2: Separate frequency and time lists
    text_separate = audio_to_text(audio_data, sample_rate, format_type='separate', separator=',')
    print("Format 'separate':")
    print(text_separate[:200] + "..." if len(text_separate) > 200 else text_separate)
    print()
    
    # Format 3: Compact format
    text_compact = audio_to_text(audio_data, sample_rate, format_type='compact')
    print("Format 'compact':")
    print(text_compact[:200] + "..." if len(text_compact) > 200 else text_compact)
    print()
    
    # Demonstrate peak detection details
    print("Peak Detection Analysis:")
    print("-" * 30)
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(audio_data, sample_rate)
    
    print(f"Total peaks detected: {len(peak_freqs)}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    print(f"Time range: {times[0]:.3f} - {times[-1]:.3f} seconds")
    
    if len(peak_freqs) > 0:
        actual_freqs = freqs[peak_freqs]
        actual_times = times[peak_times]
        print(f"Peak frequency range: {actual_freqs.min():.1f} - {actual_freqs.max():.1f} Hz")
        print(f"Peak time range: {actual_times.min():.3f} - {actual_times.max():.3f} seconds")
        print(f"Peak amplitude range: {peak_values.min():.6f} - {peak_values.max():.6f}")
    
    print()
    
    # Demonstrate energy change detection
    print("Energy Change Detection:")
    print("-" * 25)
    minima, maxima = detect_energy_changes(audio_data, sample_rate)
    print(f"Energy minima detected: {len(minima)}")
    print(f"Energy maxima detected: {len(maxima)}")
    
    if len(maxima) > 0:
        print(f"First few energy maxima at time indices: {maxima[:5]}")
    if len(minima) > 0:
        print(f"First few energy minima at time indices: {minima[:5]}")


if __name__ == "__main__":
    main()