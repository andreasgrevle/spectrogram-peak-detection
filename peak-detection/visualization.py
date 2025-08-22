#!/usr/bin/env python3
"""
Visualization Module for Spectrogram Peak Detection

This module provides various visualization functions for displaying
spectrogram data, detected peaks, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from peakmodule import detect_peaks_from_audio, detect_energy_changes


def plot_spectrogram_with_peaks(audio_data, sample_rate=22050, nperseg=256, 
                               figsize=(12, 8), cmap='viridis', peak_color='red',
                               peak_size=20, title=None):
    """
    Plot spectrogram with detected peaks overlaid.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        nperseg (int): Length of each segment for spectrogram
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap for spectrogram
        peak_color (str): Color for peak markers
        peak_size (int): Size of peak markers
        title (str): Plot title
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Detect peaks
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate, nperseg
    )
    
    # Compute spectrogram for visualization
    f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrogram
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                       shading='gouraud', cmap=cmap)
    
    # Overlay peaks
    if len(peak_freqs) > 0:
        actual_freqs = freqs[peak_freqs]
        actual_times = times[peak_times]
        ax.scatter(actual_times, actual_freqs, c=peak_color, s=peak_size, 
                  alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title or f'Spectrogram with {len(peak_freqs)} Detected Peaks')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    plt.tight_layout()
    return fig, ax


def plot_peak_scatter(audio_data, sample_rate=22050, figsize=(10, 6),
                     color_by='amplitude', cmap='plasma', alpha=0.7):
    """
    Create a scatter plot of detected peaks in frequency-time space.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        figsize (tuple): Figure size
        color_by (str): Color peaks by 'amplitude', 'frequency', or 'time'
        cmap (str): Colormap for scatter plot
        alpha (float): Transparency of points
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Detect peaks
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate
    )
    
    if len(peak_freqs) == 0:
        print("No peaks detected for scatter plot")
        return None, None
    
    # Convert to actual values
    actual_freqs = freqs[peak_freqs]
    actual_times = times[peak_times]
    
    # Determine color values
    if color_by == 'amplitude':
        colors_vals = peak_values
        color_label = 'Amplitude'
    elif color_by == 'frequency':
        colors_vals = actual_freqs
        color_label = 'Frequency (Hz)'
    else:  # time
        colors_vals = actual_times
        color_label = 'Time (s)'
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(actual_times, actual_freqs, c=colors_vals, 
                        cmap=cmap, alpha=alpha, s=30)
    
    # Formatting
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Peak Distribution ({len(peak_freqs)} peaks)')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label)
    
    plt.tight_layout()
    return fig, ax


def plot_energy_analysis(audio_data, sample_rate=22050, figsize=(12, 8)):
    """
    Plot energy analysis showing energy changes over time.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axis objects
    """
    # Compute spectrogram and energy
    freqs, times, spec = signal.spectrogram(audio_data, fs=sample_rate)
    energy = np.sum(spec**2, axis=0)
    
    # Detect energy changes
    minima, maxima = detect_energy_changes(audio_data, sample_rate)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Waveform with energy markers
    time_audio = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
    ax1.plot(time_audio, audio_data, 'b-', alpha=0.7, label='Audio Signal')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Energy over time with extrema
    ax2.plot(times, energy, 'g-', linewidth=2, label='Energy')
    
    # Mark energy extrema
    if len(maxima) > 0:
        ax2.scatter(times[maxima], energy[maxima], c='red', s=50, 
                   label=f'Maxima ({len(maxima)})', zorder=5)
    if len(minima) > 0:
        ax2.scatter(times[minima], energy[minima], c='blue', s=50, 
                   label=f'Minima ({len(minima)})', zorder=5)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Analysis with Extrema Detection')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_frequency_distribution(audio_data, sample_rate=22050, bins=50, 
                               figsize=(10, 6)):
    """
    Plot histogram of detected peak frequencies.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        bins (int): Number of histogram bins
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Detect peaks
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate
    )
    
    if len(peak_freqs) == 0:
        print("No peaks detected for frequency distribution")
        return None, None
    
    actual_freqs = freqs[peak_freqs]
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    n, bins_edges, patches = ax.hist(actual_freqs, bins=bins, alpha=0.7, 
                                    color='skyblue', edgecolor='black')
    
    # Color bars by frequency
    for i, (patch, freq) in enumerate(zip(patches, bins_edges[:-1])):
        patch.set_facecolor(plt.cm.viridis(freq / actual_freqs.max()))
    
    # Formatting
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Number of Peaks')
    ax.set_title(f'Peak Frequency Distribution ({len(peak_freqs)} total peaks)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_freq = np.mean(actual_freqs)
    std_freq = np.std(actual_freqs)
    ax.axvline(mean_freq, color='red', linestyle='--', 
              label=f'Mean: {mean_freq:.1f} Hz')
    ax.axvline(mean_freq + std_freq, color='orange', linestyle=':', alpha=0.7,
              label=f'±1σ: {std_freq:.1f} Hz')
    ax.axvline(mean_freq - std_freq, color='orange', linestyle=':', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def plot_peak_timeline(audio_data, sample_rate=22050, figsize=(12, 6),
                      freq_bands=None):
    """
    Plot peaks over time, optionally grouped by frequency bands.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        figsize (tuple): Figure size
        freq_bands (list): List of frequency band edges for grouping
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Detect peaks
    freqs, times, peak_freqs, peak_times, peak_values = detect_peaks_from_audio(
        audio_data, sample_rate
    )
    
    if len(peak_freqs) == 0:
        print("No peaks detected for timeline plot")
        return None, None
    
    actual_freqs = freqs[peak_freqs]
    actual_times = times[peak_times]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if freq_bands is None:
        # Simple scatter plot
        scatter = ax.scatter(actual_times, actual_freqs, c=peak_values, 
                           cmap='plasma', alpha=0.7, s=20)
        plt.colorbar(scatter, ax=ax, label='Amplitude')
    else:
        # Group by frequency bands
        colors = plt.cm.Set3(np.linspace(0, 1, len(freq_bands)-1))
        
        for i in range(len(freq_bands)-1):
            band_mask = (actual_freqs >= freq_bands[i]) & (actual_freqs < freq_bands[i+1])
            if np.any(band_mask):
                ax.scatter(actual_times[band_mask], actual_freqs[band_mask], 
                          c=[colors[i]], alpha=0.7, s=20,
                          label=f'{freq_bands[i]:.0f}-{freq_bands[i+1]:.0f} Hz')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Formatting
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Peak Timeline ({len(peak_freqs)} peaks)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def create_comprehensive_analysis(audio_data, sample_rate=22050, 
                                 save_path=None, show_plots=False):
    """
    Create a comprehensive analysis with multiple visualization plots.
    
    Args:
        audio_data (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of audio
        save_path (str): Optional path to save plots
        show_plots (bool): Whether to display plots
    
    Returns:
        dict: Dictionary containing all figure objects
    """
    figures = {}
    
    print("Generating comprehensive visualization analysis...")
    
    # 1. Spectrogram with peaks
    print("Creating spectrogram with peaks...")
    fig1, ax1 = plot_spectrogram_with_peaks(audio_data, sample_rate)
    figures['spectrogram'] = fig1
    
    # 2. Peak scatter plot
    print("Creating peak scatter plot...")
    fig2, ax2 = plot_peak_scatter(audio_data, sample_rate, color_by='amplitude')
    if fig2 is not None:
        figures['scatter'] = fig2
    
    # 3. Energy analysis
    print("Creating energy analysis...")
    fig3, axes3 = plot_energy_analysis(audio_data, sample_rate)
    figures['energy'] = fig3
    
    # 4. Frequency distribution
    print("Creating frequency distribution...")
    fig4, ax4 = plot_frequency_distribution(audio_data, sample_rate)
    if fig4 is not None:
        figures['frequency_dist'] = fig4
    
    # 5. Peak timeline
    print("Creating peak timeline...")
    # Define some common frequency bands
    freq_bands = [0, 250, 500, 1000, 2000, 4000, 8000, sample_rate//2]
    fig5, ax5 = plot_peak_timeline(audio_data, sample_rate, freq_bands=freq_bands)
    if fig5 is not None:
        figures['timeline'] = fig5
    
    # Save plots if requested
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(os.path.join(save_path, f'{name}.png'), 
                       dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    
    return figures


def plot_comparison(audio_data1, audio_data2, sample_rate=22050, 
                   labels=('Audio 1', 'Audio 2'), figsize=(15, 10)):
    """
    Compare peak detection results between two audio signals.
    
    Args:
        audio_data1, audio_data2 (np.ndarray): Input audio signals
        sample_rate (int): Sample rate of audio
        labels (tuple): Labels for the two signals
        figsize (tuple): Figure size
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axis objects
    """
    # Detect peaks for both signals
    results1 = detect_peaks_from_audio(audio_data1, sample_rate)
    results2 = detect_peaks_from_audio(audio_data2, sample_rate)
    
    freqs1, times1, peak_freqs1, peak_times1, peak_values1 = results1
    freqs2, times2, peak_freqs2, peak_times2, peak_values2 = results2
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Spectrograms side by side
    for i, (audio, label) in enumerate(zip([audio_data1, audio_data2], labels)):
        f, t, Sxx = signal.spectrogram(audio, fs=sample_rate)
        im = axes[0, i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                                  shading='gouraud', cmap='viridis')
        axes[0, i].set_title(f'{label} - Spectrogram')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=axes[0, i])
    
    # Plot 2: Peak comparison scatter
    if len(peak_freqs1) > 0:
        axes[1, 0].scatter(times1[peak_times1], freqs1[peak_freqs1], 
                          alpha=0.7, label=f'{labels[0]} ({len(peak_freqs1)} peaks)')
    if len(peak_freqs2) > 0:
        axes[1, 0].scatter(times2[peak_times2], freqs2[peak_freqs2], 
                          alpha=0.7, label=f'{labels[1]} ({len(peak_freqs2)} peaks)')
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title('Peak Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 3: Frequency distribution comparison
    if len(peak_freqs1) > 0 and len(peak_freqs2) > 0:
        axes[1, 1].hist(freqs1[peak_freqs1], bins=30, alpha=0.7, 
                       label=labels[0], density=True)
        axes[1, 1].hist(freqs2[peak_freqs2], bins=30, alpha=0.7, 
                       label=labels[1], density=True)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Frequency Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
