#!/usr/bin/env python3
"""
Visualization Demo Script

This script demonstrates all the visualization capabilities for the 
spectrogram peak detection project.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os

# Add the peak-detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'peak-detection'))
from peakmodule import audio_to_text, detect_peaks_from_audio
from visualization import (
    plot_spectrogram_with_peaks,
    plot_peak_scatter,
    plot_energy_analysis,
    plot_frequency_distribution,
    plot_peak_timeline,
    create_comprehensive_analysis,
    plot_comparison
)


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


def generate_test_signals():
    """Generate multiple test signals for demonstration."""
    sample_rate = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    signals = {}
    
    # Signal 1: Musical chord (multiple harmonics)
    fundamental = 220  # A3
    chord_signal = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +      # Fundamental
        0.3 * np.sin(2 * np.pi * fundamental * 1.25 * t) +  # Minor third
        0.2 * np.sin(2 * np.pi * fundamental * 1.5 * t) +   # Perfect fifth
        0.1 * np.sin(2 * np.pi * fundamental * 2 * t)       # Octave
    )
    # Add envelope
    envelope = np.exp(-t / 2.0) * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))
    signals['Musical Chord'] = chord_signal * envelope
    
    # Signal 2: Chirp signal (frequency sweep)
    f0, f1 = 200, 2000
    chirp_signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    signals['Frequency Sweep'] = chirp_signal * np.exp(-t / 4.0)
    
    # Signal 3: Noisy signal with peaks
    noise_base = 0.1 * np.random.randn(len(t))
    # Add some clear peaks
    peak_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    peak_freqs = [440, 880, 660, 1100, 330]
    for pt, pf in zip(peak_times, peak_freqs):
        peak_mask = np.abs(t - pt) < 0.1
        noise_base[peak_mask] += 0.5 * np.sin(2 * np.pi * pf * t[peak_mask])
    signals['Noisy with Peaks'] = noise_base
    
    return signals, sample_rate


def demo_individual_plots():
    """Demonstrate individual visualization functions."""
    print("\n" + "="*60)
    print("INDIVIDUAL PLOT DEMONSTRATIONS")
    print("="*60)
    
    # Generate test signal
    signals, sample_rate = generate_test_signals()
    audio_data = signals['Musical Chord']
    
    print(f"\nUsing test signal: Musical Chord")
    print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    
    # 1. Spectrogram with peaks
    print("\n1. Creating spectrogram with detected peaks overlay...")
    fig1, ax1 = plot_spectrogram_with_peaks(
        audio_data, sample_rate, 
        title="Musical Chord - Spectrogram with Peaks"
    )
    
    # 2. Peak scatter plot
    print("2. Creating peak scatter plot colored by amplitude...")
    fig2, ax2 = plot_peak_scatter(
        audio_data, sample_rate, 
        color_by='amplitude'
    )
    
    # 3. Energy analysis
    print("3. Creating energy analysis with extrema detection...")
    fig3, axes3 = plot_energy_analysis(audio_data, sample_rate)
    
    # 4. Frequency distribution
    print("4. Creating frequency distribution histogram...")
    fig4, ax4 = plot_frequency_distribution(audio_data, sample_rate)
    
    # 5. Peak timeline with frequency bands
    print("5. Creating peak timeline with frequency band grouping...")
    freq_bands = [0, 200, 500, 1000, 2000, 4000, sample_rate//2]
    fig5, ax5 = plot_peak_timeline(
        audio_data, sample_rate, 
        freq_bands=freq_bands
    )
    
    # Show peak detection statistics
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate
    )
    print(f"\nDetected {len(peak_freqs)} peaks in the musical chord signal")
    if len(peak_freqs) > 0:
        actual_freqs = freqs[peak_freqs]
        print(f"Frequency range: {actual_freqs.min():.1f} - {actual_freqs.max():.1f} Hz")
        print(f"Most prominent frequencies: {sorted(actual_freqs)[:5]}")
    
    return [fig1, fig2, fig3, fig4, fig5]


def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis function."""
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Try to load real audio file first
    audio_file = "audiosample.wav"
    if os.path.exists(audio_file):
        print(f"\nLoading real audio file: {audio_file}")
        sample_rate, audio_data = load_audio_file(audio_file)
        if audio_data is None:
            print("Failed to load audio file, using generated signal")
            signals, sample_rate = generate_test_signals()
            audio_data = signals['Frequency Sweep']
    else:
        print("\nNo audio file found, using generated frequency sweep signal")
        signals, sample_rate = generate_test_signals()
        audio_data = signals['Frequency Sweep']
    
    print(f"Audio length: {len(audio_data)} samples")
    print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    
    # Create comprehensive analysis
    figures = create_comprehensive_analysis(
        audio_data, sample_rate, 
        save_path="plots",
        show_plots=False  # Don't show immediately, we'll show at the end
    )
    
    print(f"\nGenerated {len(figures)} visualization plots")
    print("Plots saved to 'plots' directory")
    
    return figures


def demo_signal_comparison():
    """Demonstrate comparison between different signal types."""
    print("\n" + "="*60)
    print("SIGNAL COMPARISON DEMONSTRATION")
    print("="*60)
    
    signals, sample_rate = generate_test_signals()
    
    # Compare musical chord vs noisy signal
    signal1 = signals['Musical Chord']
    signal2 = signals['Noisy with Peaks']
    
    print("\nComparing 'Musical Chord' vs 'Noisy with Peaks' signals...")
    
    fig, axes = plot_comparison(
        signal1, signal2, sample_rate,
        labels=('Musical Chord', 'Noisy with Peaks')
    )
    
    # Print comparison statistics
    results1 = detect_peaks_from_audio(signal1, sample_rate)
    results2 = detect_peaks_from_audio(signal2, sample_rate)
    
    print(f"Musical Chord: {len(results1[2])} peaks detected")
    print(f"Noisy Signal: {len(results2[2])} peaks detected")
    
    return fig


def demo_text_output_with_visualization():
    """Demonstrate text output alongside visualizations."""
    print("\n" + "="*60)
    print("TEXT OUTPUT WITH VISUALIZATION")
    print("="*60)
    
    signals, sample_rate = generate_test_signals()
    audio_data = signals['Musical Chord']
    
    # Generate text representations
    print("\nGenerating text representations of the musical chord...")
    
    text_pairs = audio_to_text(audio_data, sample_rate, format_type='pairs')
    text_separate = audio_to_text(audio_data, sample_rate, format_type='separate')
    text_compact = audio_to_text(audio_data, sample_rate, format_type='compact')
    
    print(f"\nPairs format (first 100 chars): {text_pairs[:100]}...")
    print(f"Separate format (first 100 chars): {text_separate[:100]}...")
    print(f"Compact format (first 100 chars): {text_compact[:100]}...")
    
    # Create visualization to accompany the text
    fig, ax = plot_spectrogram_with_peaks(
        audio_data, sample_rate,
        title="Musical Chord - Text Representation Source"
    )
    
    return fig


def main():
    """Main demonstration function."""
    print("SPECTROGRAM PEAK DETECTION - VISUALIZATION DEMO")
    print("="*60)
    print("This demo showcases various visualization capabilities for")
    print("analyzing spectrogram peak detection results.")
    
    # Set matplotlib to non-interactive mode initially
    plt.ioff()
    
    all_figures = []
    
    try:
        # 1. Individual plot demonstrations
        figs1 = demo_individual_plots()
        all_figures.extend(figs1)
        
        # 2. Comprehensive analysis
        figs2 = demo_comprehensive_analysis()
        all_figures.extend(figs2.values())
        
        # 3. Signal comparison
        fig3 = demo_signal_comparison()
        all_figures.append(fig3)
        
        # 4. Text output with visualization
        fig4 = demo_text_output_with_visualization()
        all_figures.append(fig4)
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print(f"Generated {len(all_figures)} visualization plots total")
        print("\nVisualization capabilities demonstrated:")
        print("• Spectrogram with peak overlays")
        print("• Peak scatter plots with various color schemes")
        print("• Energy analysis and extrema detection")
        print("• Frequency distribution histograms")
        print("• Peak timelines with frequency band grouping")
        print("• Comprehensive multi-plot analysis")
        print("• Signal comparison visualizations")
        print("• Integration with text output formats")
        
        # Plots are saved to the plots/ folder for later viewing
        print(f"\nAll {len(all_figures)} plots have been saved to the 'plots/' folder.")
        print("You can view them at any time without running the demo again.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
