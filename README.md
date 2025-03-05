# Multimodal Emotion Recognition using MELD Dataset

This project implements a multimodal emotion recognition system using the MELD (Multimodal EmotionLines Dataset) dataset. The system combines text, video, and audio data to recognize emotions and sentiments in conversational scenarios.

## Overview

The system processes three types of data:
- **Text**: Dialogue utterances
- **Video**: Visual frames capturing facial expressions and gestures
- **Audio**: Voice features including tone and pitch

### Emotion Classes
- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

### Sentiment Classes
- Positive
- Negative
- Neutral


## Features

### Text Processing
- BERT tokenizer for text encoding
- Maximum sequence length: 128 tokens
- Automatic padding and truncation

### Video Processing
- Extracts 30 frames per video
- Resizes frames to 224x224
- Normalizes pixel values
- Handles missing frames with zero padding

### Audio Processing
- Extracts audio using FFmpeg
- Converts to 16kHz mono WAV
- Creates mel spectrograms
- Normalizes audio features

## Technical Details

### Model Architecture
The system uses:
- BERT for text processing
- Custom video frame extraction
- Mel spectrogram for audio features
- Multimodal fusion for final prediction

### Performance Considerations
- Batch processing for efficiency
- Memory-efficient data loading
- Robust error handling for missing data
- Custom collate function for None handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MELD dataset creators
- Transformers library by Hugging Face
- PyTorch team
