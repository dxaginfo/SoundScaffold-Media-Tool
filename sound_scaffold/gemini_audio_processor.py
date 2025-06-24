"""
Gemini AI integration for audio processing.

This module provides functionality to process audio data using the
Gemini API for advanced analysis and content understanding.
"""

import os
import json
import base64
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union
import numpy as np
import librosa
import soundfile as sf
import google.generativeai as genai
from .utils.file_handler import validate_audio_file, load_audio_file

logger = logging.getLogger(__name__)

# Default prompts for different audio analysis tasks
DEFAULT_PROMPTS = {
    'audio_description': (
        "Describe the audio content in detail, including ambience, sound effects, "
        "music elements, dialogue characteristics, emotional tone, and any "
        "distinctive features. Be specific about the sonic elements present."
    ),
    'sound_categorization': (
        "Categorize this sound into a primary category and subcategories. "
        "Primary categories may include: dialogue, ambience, sound effect, music, "
        "silence, noise. Provide subcategories like: dialogue (conversational, "
        "monologue, whispered), ambience (urban, nature, interior), etc. "
        "Provide your response as a JSON object with 'primary_category' and "
        "'subcategories' keys."
    ),
    'enhancement_suggestions': (
        "Based on the audio characteristics, suggest specific audio enhancement "
        "techniques that could improve its quality for media production. "
        "Consider aspects like clarity, background noise, balance, "
        "frequency issues, and dynamic range. Provide specific technical "
        "parameters when relevant (e.g., frequency ranges, dB adjustments)."
    ),
    'emotional_analysis': (
        "Analyze the emotional characteristics of this audio. Consider the "
        "emotional tone, intensity, mood, and emotional progression throughout "
        "the clip. If dialogue is present, consider the emotional content of the "
        "speech. Provide your response as a JSON object with 'primary_emotion', "
        "'intensity', 'mood', and 'progression' keys."
    ),
    'continuity_assessment': (
        "Assess this audio clip for potential continuity issues in a media "
        "production context. Identify any sudden changes, background "
        "inconsistencies, audio level mismatches, or other elements that "
        "might create continuity problems when edited into a sequence."
    )
}


class GeminiAudioProcessor:
    """
    Class for processing audio using Google's Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GeminiAudioProcessor.
        
        Args:
            api_key: Gemini API key. If None, will use GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. Using application default credentials.")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        
        # Get available models
        self.models = [m.name for m in genai.list_models() 
                     if 'generateContent' in m.supported_generation_methods]
        
        # Select the most capable model available
        self.model_name = next((m for m in self.models if 'vision' in m.lower()), self.models[0])
        logger.info(f"Using Gemini model: {self.model_name}")
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)
    
    def analyze_audio_content(
        self, 
        audio_path: str, 
        analysis_type: str = 'audio_description',
        custom_prompt: Optional[str] = None,
        sample_duration: Optional[float] = None,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze audio content using Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            analysis_type: Type of analysis to perform, from DEFAULT_PROMPTS keys
            custom_prompt: Custom prompt to use instead of default
            sample_duration: Duration in seconds to sample from the audio (None = full file)
            visualize: Whether to include visualizations in the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        validate_audio_file(audio_path)
        
        # Load audio file
        y, sr = load_audio_file(audio_path, sr=None)
        
        # If sample duration is specified, take a sample from the audio
        if sample_duration and sample_duration < librosa.get_duration(y=y, sr=sr):
            # Take from a random position if file is long enough
            if librosa.get_duration(y=y, sr=sr) > sample_duration * 2:
                start = np.random.randint(0, len(y) - int(sample_duration * sr))
                y = y[start:start + int(sample_duration * sr)]
            else:
                # Just take from the beginning
                y = y[:int(sample_duration * sr)]
        
        # Generate a mel spectrogram visualization
        visualization_path = None
        if visualize:
            visualization_path = self._create_audio_visualization(y, sr)
        
        # Prepare prompt
        prompt = custom_prompt or DEFAULT_PROMPTS.get(
            analysis_type, DEFAULT_PROMPTS['audio_description']
        )
        
        # Get audio features to include in the prompt
        audio_features = self._extract_audio_features_summary(y, sr)
        
        # Create a comprehensive prompt with audio features
        full_prompt = (
            f"TASK: {prompt}\n\n"
            f"AUDIO TECHNICAL DETAILS:\n{json.dumps(audio_features, indent=2)}\n\n"
            "Provide a detailed and technically accurate analysis based on the "
            "audio features and visualization."
        )
        
        # Prepare content parts for the Gemini API
        content_parts = [full_prompt]
        
        # Add visualization if available
        if visualization_path:
            with open(visualization_path, 'rb') as f:
                image_data = f.read()
            
            content_parts.append({
                "mime_type": "image/png",
                "data": base64.b64encode(image_data).decode('utf-8')
            })
            
            # Clean up the temporary file
            os.remove(visualization_path)
        
        # Get the response from Gemini
        response = self.model.generate_content(content_parts)
        
        # Extract JSON if present in the response
        result = {
            "analysis_type": analysis_type,
            "prompt": prompt,
            "response": response.text,
        }
        
        # Try to extract structured data if it looks like JSON
        if '{' in response.text and '}' in response.text:
            try:
                # Extract JSON from text (may be surrounded by markdown)
                json_text = response.text.split('```json')[1].split('```')[0] if '```json' in response.text else response.text
                json_text = json_text.strip()
                
                # Make sure it starts and ends with braces
                if not json_text.startswith('{'):
                    json_text = json_text[json_text.find('{'):]
                if not json_text.endswith('}'):
                    json_text = json_text[:json_text.rfind('}')+1]
                
                structured_data = json.loads(json_text)
                result["structured_data"] = structured_data
            except (json.JSONDecodeError, IndexError):
                logger.warning("Could not extract structured data from response")
        
        return result
    
    def categorize_sound(self, audio_path: str) -> Dict[str, Any]:
        """
        Categorize sound using Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing categorization results
        """
        return self.analyze_audio_content(
            audio_path, 
            analysis_type='sound_categorization'
        )
    
    def get_enhancement_suggestions(self, audio_path: str) -> Dict[str, Any]:
        """
        Get audio enhancement suggestions using Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing enhancement suggestions
        """
        return self.analyze_audio_content(
            audio_path, 
            analysis_type='enhancement_suggestions'
        )
    
    def analyze_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze emotional content of audio using Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing emotional analysis
        """
        return self.analyze_audio_content(
            audio_path, 
            analysis_type='emotional_analysis'
        )
    
    def assess_continuity(self, audio_path: str) -> Dict[str, Any]:
        """
        Assess audio continuity issues using Gemini AI.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing continuity assessment
        """
        return self.analyze_audio_content(
            audio_path, 
            analysis_type='continuity_assessment'
        )
    
    def _extract_audio_features_summary(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract a summary of audio features for Gemini prompt context.
        
        Args:
            y: Audio data
            sr: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Calculate spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Calculate tempo and beat information
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Speech detection heuristic
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std = np.std(mfccs, axis=1)
        speech_likelihood = "high" if mfcc_std[1] > 15 and mfcc_std[2] > 10 else "medium" if mfcc_std[1] > 10 else "low"
        
        # Return features summary
        return {
            "duration": float(duration),
            "sample_rate": int(sr),
            "channels": 1 if y.ndim == 1 else y.shape[0],
            "energy": {
                "mean": float(np.mean(rms)),
                "max": float(np.max(rms)),
                "dynamic_range_db": float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-8)))
            },
            "spectral": {
                "centroid_mean": float(np.mean(spectral_centroid)),
                "centroid_std": float(np.std(spectral_centroid)),
                "contrast_mean": float(np.mean(np.mean(spectral_contrast, axis=1))),
            },
            "rhythm": {
                "tempo_bpm": float(tempo),
                "beat_strength": float(np.mean(onset_env)),
                "beat_regularity": float(np.std(librosa.util.peak_pick(onset_env, 3, 3, 3, 3, 0.5, 10)) / np.mean(onset_env)) if len(onset_env) > 0 else 0
            },
            "speech_characteristics": {
                "speech_likelihood": speech_likelihood,
                "harmonic_ratio": float(np.mean(librosa.feature.spectral_flatness(y=y))),
            }
        }
    
    def _create_audio_visualization(self, y: np.ndarray, sr: int) -> str:
        """
        Create a visualization of the audio for Gemini analysis.
        
        Args:
            y: Audio data
            sr: Sample rate
            
        Returns:
            Path to the visualization image file
        """
        import matplotlib.pyplot as plt
        
        # Create a figure with multiple plots
        plt.figure(figsize=(10, 8))
        
        # Waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Mel spectrogram
        plt.subplot(3, 1, 2)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Chromagram
        plt.subplot(3, 1, 3)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title('Chromagram')
        
        plt.tight_layout()
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name)
        plt.close()
        
        return temp_file.name