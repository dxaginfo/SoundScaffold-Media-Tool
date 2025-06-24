# SoundScaffold - Media Automation Tool

SoundScaffold is an audio scene management tool designed to organize, analyze, and optimize audio elements within media productions. It leverages Python, TensorFlow, Google Cloud technologies, and Gemini API to provide automated audio scene analysis, categorization, and enhancement capabilities.

## Features

- **Audio Scene Analysis**: Analyzes audio files to identify key elements, transitions, and quality issues
- **Sound Library Management**: Organizes and categorizes sound effects and audio elements
- **Audio Enhancement**: Applies AI-driven optimizations to improve audio quality
- **Scene Mapping**: Maps audio elements to visual scenes for better synchronization
- **Continuity Checking**: Ensures audio continuity across scene transitions
- **Gemini-Powered Analysis**: Utilizes Google's Gemini API for advanced content understanding and recommendations
- **Format Support**: Processes multiple audio formats including WAV, MP3, FLAC, ALAC, and AAC
- **Spatial Audio**: Analyzes and processes surround sound formats including 5.1, 7.1, and Atmos

## Core Components

### AudioAnalyzer

The `AudioAnalyzer` class provides functionality to analyze audio files and extract scene information, transitions, and quality metrics.

```python
from sound_scaffold import AudioAnalyzer

analyzer = AudioAnalyzer()
results = analyzer.analyze("scene_audio.wav")
print(results['scene_breakdown'])
```

### AudioEnhancer

The `AudioEnhancer` class provides functionality to enhance audio files with various processing algorithms optimized for media production.

```python
from sound_scaffold import AudioEnhancer

enhancer = AudioEnhancer()
enhanced_file = enhancer.enhance(
    "original.wav", 
    enhancement_type="dialogue_clarity",
    options={"level": 0.7}
)
print(f"Enhanced file saved to: {enhanced_file}")
```

### SoundLibraryManager

The `SoundLibraryManager` class provides functionality to manage, categorize, and search audio files in a sound library for media production.

```python
from sound_scaffold import SoundLibraryManager

manager = SoundLibraryManager()
results = manager.search_sounds({
    "category": "ambient",
    "mood": "tense",
    "duration_max": 30
})
for sound in results:
    print(f"{sound['name']}: {sound['description']}")
```

### GeminiIntegration

The `GeminiIntegration` class provides advanced audio content analysis and enhancement suggestions using Google's Gemini API.

```python
from sound_scaffold import GeminiIntegration, AudioAnalyzer

# First extract audio features
analyzer = AudioAnalyzer()
audio_features = analyzer._extract_features("scene_audio.wav")

# Use Gemini to analyze audio content
gemini = GeminiIntegration()
analysis = gemini.analyze_audio_content(audio_features)
print(analysis['gemini_analysis'])

# Get AI-powered enhancement suggestions
suggestions = gemini.suggest_audio_enhancements(
    audio_features, 
    analyzer._assess_quality(audio_features)
)
print(suggestions['enhancement_recommendations'])

# Categorize sounds with Gemini
categorization = gemini.categorize_sound("ambient rainfall with distant thunder")
print(categorization)
```

## API Endpoints

SoundScaffold provides a RESTful API for integration with other tools:

- `POST /analyze`: Submit audio file for analysis
- `GET /analysis/{analysis_id}`: Retrieve analysis results
- `POST /enhance`: Submit audio for enhancement
- `GET /library/search`: Search sound library
- `POST /library/add`: Add to sound library
- `GET /scenes/{project_id}`: Get audio scenes for project
- `POST /gemini/analyze`: Analyze audio using Gemini API
- `POST /gemini/enhance`: Get enhancement suggestions using Gemini API

## Installation

### Prerequisites

- Python 3.9+
- TensorFlow 2.10+
- Google Cloud SDK
- FFmpeg
- Google Gemini API access

### Installation Steps

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Google Cloud credentials
4. Configure Gemini API access:
   - Set the `GEMINI_API_KEY` environment variable, or
   - Use application default credentials
5. Run the application: `python api/app.py`

## Deployment

### Google Cloud Functions

SoundScaffold can be deployed as a Google Cloud Function for serverless operation.

```yaml
runtime: python39
entrypoint: app

env_variables:
  GOOGLE_CLOUD_PROJECT: "sound-scaffold-project"
  STORAGE_BUCKET: "sound-scaffold-assets"
  GEMINI_API_KEY: "your-gemini-api-key"
  
handlers:
- url: /.*
  script: auto
```

### Docker Container

Alternatively, deploy as a Docker container:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
```

## Integration with Other Media Automation Tools

SoundScaffold is designed to integrate seamlessly with other media automation tools:

- **SceneValidator**: For validating audio against scene requirements
- **TimelineAssembler**: For synchronizing audio with video timelines
- **VeoPromptExporter**: For audio-aware prompt generation for AI video tools
- **LoopOptimizer**: For optimizing repeating audio elements
- **FormatNormalizer**: For ensuring format consistency across media

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on GitHub or contact the Media Automation Tools team.