# Audio to Text Conversion

Convert audio signals to text representations by detecting peaks in spectrograms and encoding them as frequency-time pairs.

## Features

- **Peak Detection**: Detect local maxima in spectrograms using morphological operations
- **Multiple Output Formats**: Choose from pairs, separate lists, or compact formats
- **Energy Analysis**: Detect energy changes in audio signals
- **Flexible Parameters**: Configurable sample rates, window sizes, and formatting options

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from peak_detection.peakmodule import audio_to_text, detect_peaks_from_audio
import numpy as np

# Generate or load audio data
audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1-second 440Hz tone

# Convert to text (frequency-time pairs)
text = audio_to_text(audio_data, sample_rate=22050, format_type='pairs')
print(text)
```

### Running the Demo

```bash
python main.py
```

This will demonstrate different output formats and analysis features using either the provided `audiosample.wav` file or a generated test signal.

## Output Formats

- **pairs**: `(freq1,time1) | (freq2,time2) | ...`
- **separate**: `FREQ:f1,f2,f3 TIME:t1,t2,t3`
- **compact**: `f1 t1 f2 t2 f3 t3 ...`

## API Reference

### `audio_to_text(audio_data, sample_rate=22050, format_type='pairs', separator='|')`
Convert audio to text representation.

### `detect_peaks_from_audio(audio_data, sample_rate=22050, nperseg=256, noverlap=None)`
Detect peaks in audio and return detailed analysis.

### `detect_energy_changes(audio_data, sample_rate=None)`
Detect energy minima and maxima in the audio signal.
