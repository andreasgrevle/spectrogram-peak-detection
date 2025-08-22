#!/usr/bin/env python3
"""
Audio to Text Conversion with Optional Visualization

This script demonstrates audio-to-text conversion and optionally creates
comprehensive visualizations of the peak detection analysis.
"""

import numpy as np
from scipy.io import wavfile
import sys
import os
import argparse

# Add the peak-detection module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'peak-detection'))
from peakmodule import audio_to_text, detect_peaks_from_audio, detect_energy_changes

# Try to import visualization functions (optional)
try:
    import matplotlib.pyplot as plt
    from visualization import (
        plot_spectrogram_with_peaks,
        plot_peak_scatter,
        plot_energy_analysis,
        plot_frequency_distribution,
        plot_peak_timeline,
        create_comprehensive_analysis,
        plot_comparison
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Visualization modules not available. Install matplotlib for visualization features.")


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
    
    # Signal 2: Simple test signal (original)
    simple_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +      # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t) +     # E6 note
        0.1 * np.random.randn(len(t))             # Some noise
    )
    simple_signal *= np.exp(-t / 2.0)  # Decay envelope
    signals['Simple Test'] = simple_signal
    
    # Signal 3: Frequency sweep
    f0, f1 = 200, 2000
    chirp_signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    signals['Frequency Sweep'] = chirp_signal * np.exp(-t / 4.0)
    
    return signals, sample_rate


def get_audio_data():
    """Get audio data from file or generate test signal."""
    # Try to load real audio file first
    audio_files = ["audiosamples/voice.wav", "audiosamples/audiosample.wav", "audiosamples/sine.wav"]
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            print(f"Loading audio file: {audio_file}")
            sample_rate, audio_data = load_audio_file(audio_file)
            if audio_data is not None:
                return audio_data, sample_rate, f"Real audio: {audio_file}"
    
    # No audio file found, use generated signal
    print("No audio files found, using generated test signal")
    signals, sample_rate = generate_test_signals()
    return signals['Simple Test'], sample_rate, "Generated: Simple Test"


def text_analysis(audio_data, sample_rate):
    """Perform text analysis of audio data."""
    print("\n" + "="*60)
    print("TEXT CONVERSION ANALYSIS")
    print("="*60)
    
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


def create_visualizations(audio_data, sample_rate, vis_type='comprehensive'):
    """Create visualizations of the audio analysis."""
    if not VISUALIZATION_AVAILABLE:
        print("\nVisualization not available. Install matplotlib to enable visualizations.")
        return []
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plt.ioff()  # Set matplotlib to non-interactive mode
    all_figures = []
    
    try:
        if vis_type == 'comprehensive':
            # Create comprehensive analysis
            print("Creating comprehensive analysis plots...")
            figures = create_comprehensive_analysis(
                audio_data, sample_rate, 
                save_path="plots",
                show_plots=False
            )
            all_figures.extend(figures.values())
            print(f"Generated {len(figures)} comprehensive plots")
            
        elif vis_type == 'individual':
            # Create individual plots
            print("Creating individual visualization plots...")
            
            # 1. Spectrogram with peaks
            print("1. Spectrogram with detected peaks...")
            fig1, ax1 = plot_spectrogram_with_peaks(
                audio_data, sample_rate, 
                title="Spectrogram with Peaks"
            )
            all_figures.append(fig1)
            
            # 2. Peak scatter plot
            print("2. Peak scatter plot...")
            fig2, ax2 = plot_peak_scatter(
                audio_data, sample_rate, 
                color_by='amplitude'
            )
            all_figures.append(fig2)
            
            # 3. Energy analysis
            print("3. Energy analysis...")
            fig3, axes3 = plot_energy_analysis(audio_data, sample_rate)
            all_figures.append(fig3)
            
            # 4. Frequency distribution
            print("4. Frequency distribution...")
            fig4, ax4 = plot_frequency_distribution(audio_data, sample_rate)
            all_figures.append(fig4)
            
            # 5. Peak timeline
            print("5. Peak timeline...")
            freq_bands = [0, 200, 500, 1000, 2000, 4000, sample_rate//2]
            fig5, ax5 = plot_peak_timeline(
                audio_data, sample_rate, 
                freq_bands=freq_bands
            )
            all_figures.append(fig5)
            
        elif vis_type == 'comparison':
            # Create signal comparison
            print("Creating signal comparison...")
            signals, sr = generate_test_signals()
            signal1 = signals['Musical Chord']
            signal2 = signals['Simple Test']
            
            fig, axes = plot_comparison(
                signal1, signal2, sr,
                labels=('Musical Chord', 'Simple Test')
            )
            all_figures.append(fig)
            
        print(f"\nTotal plots created: {len(all_figures)}")
        print("All plots saved to 'plots/' directory")
        
        # Clean up figures to save memory
        plt.close('all')
        
        return all_figures
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return []


def main():
    """Main program with optional visualization."""
    parser = argparse.ArgumentParser(
        description='Audio to Text Conversion with Optional Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                    # Text analysis only
  python main.py --visualize        # Text + comprehensive visualizations
  python main.py --vis individual   # Text + individual plots
  python main.py --vis comparison   # Text + signal comparison"""
    )
    
    parser.add_argument('--visualize', '--vis', 
                       choices=['comprehensive', 'individual', 'comparison'],
                       default=None,
                       help='Create visualizations (requires matplotlib)')
    
    args = parser.parse_args()
    
    print("Audio to Text Conversion" + (" with Visualization" if args.visualize else ""))
    print("=" * 60)
    
    # Get audio data
    audio_data, sample_rate, source_info = get_audio_data()
    print(f"Using: {source_info}")
    
    # Always perform text analysis
    text_analysis(audio_data, sample_rate)
    
    # Optionally create visualizations
    if args.visualize:
        if args.visualize == 'comprehensive':
            create_visualizations(audio_data, sample_rate, 'comprehensive')
        elif args.visualize == 'individual':
            create_visualizations(audio_data, sample_rate, 'individual')
        elif args.visualize == 'comparison':
            create_visualizations(audio_data, sample_rate, 'comparison')
    else:
        print("\n" + "="*60)
        print("TEXT ANALYSIS COMPLETE")
        print("="*60)
        print("To create visualizations, run with --visualize option:")
        print("  python main.py --visualize comprehensive")
        print("  python main.py --visualize individual")
        print("  python main.py --visualize comparison")


if __name__ == "__main__":
    main()