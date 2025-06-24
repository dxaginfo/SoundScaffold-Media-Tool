"""
Gemini API Integration for SoundScaffold

This module provides functionality to leverage Google's Gemini API for
advanced audio content analysis, description, and enhancement suggestions.
"""

import os
import json
import logging
import google.generativeai as genai
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Gemini API
try:
    # Get API key from environment variable or use service account authentication
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Uses application default credentials
        logger.info("No API key found, using application default credentials")
except Exception as e:
    logger.error(f"Error initializing Gemini API: {e}")


class GeminiIntegration:
    """
    Provides integration with Google's Gemini API for audio content analysis.
    """
    
    DEFAULT_MODEL = "gemini-1.5-pro"
    
    def __init__(self, model_name=None):
        """
        Initialize the Gemini integration with optional model specification.
        
        Args:
            model_name (str, optional): Name of the Gemini model to use. Defaults to None.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading Gemini model: {e}")
            self.model = None
    
    def analyze_audio_content(self, audio_features, analysis_prompt=None):
        """
        Analyze audio content using Gemini API.
        
        Args:
            audio_features (dict): Extracted audio features and metadata.
            analysis_prompt (str, optional): Custom prompt for the analysis. Defaults to None.
            
        Returns:
            dict: Analysis results from Gemini API.
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return {"error": "Gemini model not initialized"}
        
        try:
            # Create a context-rich prompt based on audio features
            if not analysis_prompt:
                prompt = self._create_analysis_prompt(audio_features)
            else:
                prompt = analysis_prompt
            
            # Generate content with Gemini
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            structured_analysis = self._parse_analysis_response(response.text)
            
            return {
                "gemini_analysis": structured_analysis,
                "raw_response": response.text
            }
        
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return {"error": str(e)}
    
    def suggest_audio_enhancements(self, audio_features, quality_metrics):
        """
        Generate enhancement suggestions for audio content.
        
        Args:
            audio_features (dict): Extracted audio features.
            quality_metrics (dict): Audio quality metrics.
            
        Returns:
            dict: Enhancement suggestions.
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return {"error": "Gemini model not initialized"}
        
        try:
            # Create a prompt for enhancement suggestions
            prompt = self._create_enhancement_prompt(audio_features, quality_metrics)
            
            # Generate content with Gemini
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            enhancements = self._parse_enhancement_response(response.text)
            
            return {
                "enhancement_suggestions": enhancements,
                "raw_response": response.text
            }
        
        except Exception as e:
            logger.error(f"Error generating enhancement suggestions: {e}")
            return {"error": str(e)}
    
    def categorize_sound(self, sound_description, audio_features=None):
        """
        Categorize a sound using Gemini API.
        
        Args:
            sound_description (str): Description of the sound.
            audio_features (dict, optional): Extracted audio features if available.
            
        Returns:
            dict: Categorization results.
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return {"error": "Gemini model not initialized"}
        
        try:
            # Create a prompt for sound categorization
            prompt = f"""
            As an audio expert, categorize the following sound:
            
            SOUND DESCRIPTION: {sound_description}
            
            Provide a JSON response with the following structure:
            {{
                "primary_category": "main category name",
                "secondary_categories": ["category1", "category2"],
                "mood": ["mood1", "mood2"],
                "suitable_contexts": ["context1", "context2"],
                "tags": ["tag1", "tag2", "tag3"]
            }}
            
            Be specific and provide detailed categorization.
            """
            
            if audio_features:
                prompt += f"\n\nADDITIONAL AUDIO FEATURES: {json.dumps(audio_features, indent=2)}"
            
            # Generate content with Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON from response
            categories = self._extract_json_from_text(response.text)
            
            return categories
        
        except Exception as e:
            logger.error(f"Error categorizing sound: {e}")
            return {"error": str(e)}
    
    def _create_analysis_prompt(self, audio_features):
        """
        Create a prompt for audio content analysis.
        
        Args:
            audio_features (dict): Extracted audio features.
            
        Returns:
            str: Generated prompt.
        """
        # Extract relevant features for the prompt
        duration = audio_features.get('duration', 'unknown')
        tempo = audio_features.get('tempo', 'unknown')
        
        prompt = f"""
        As an expert in audio analysis for media production, analyze the following audio features:
        
        AUDIO FEATURES:
        - Duration: {duration} seconds
        - Tempo: {tempo} BPM
        
        Based on these features and your expertise, provide a detailed analysis of this audio in JSON format with the following structure:
        {{
            "content_type": "primary content type (dialogue, music, ambient, etc.)",
            "scene_description": "detailed description of what this audio likely represents",
            "mood_analysis": ["mood1", "mood2"],
            "dialogue_assessment": {{
                "clarity": 0-10 rating,
                "presence": 0-10 rating,
                "quality": "assessment of dialogue quality"
            }},
            "music_assessment": {{
                "presence": 0-10 rating,
                "style": "likely music style if present",
                "prominence": "assessment of how prominent music is"
            }},
            "ambient_assessment": {{
                "presence": 0-10 rating,
                "type": "type of ambient sounds if present",
                "environment": "likely environment represented"
            }},
            "technical_observations": [
                "observation1",
                "observation2"
            ]
        }}
        """
        
        return prompt
    
    def _create_enhancement_prompt(self, audio_features, quality_metrics):
        """
        Create a prompt for audio enhancement suggestions.
        
        Args:
            audio_features (dict): Extracted audio features.
            quality_metrics (dict): Audio quality metrics.
            
        Returns:
            str: Generated prompt.
        """
        # Extract quality metrics for the prompt
        snr = quality_metrics.get('signal_to_noise', 'unknown')
        clarity = quality_metrics.get('clarity', 'unknown')
        distortion = quality_metrics.get('distortion', 'unknown')
        dynamic_range = quality_metrics.get('dynamic_range', 'unknown')
        
        prompt = f"""
        As an audio engineer specializing in media production, suggest enhancements for audio with these characteristics:
        
        AUDIO QUALITY METRICS:
        - Signal-to-Noise Ratio: {snr} dB
        - Clarity: {clarity} (0-1 scale)
        - Distortion: {distortion} (0-1 scale, lower is better)
        - Dynamic Range: {dynamic_range} dB
        
        Provide detailed enhancement suggestions in JSON format with the following structure:
        {{
            "primary_issues": [
                "issue1",
                "issue2"
            ],
            "enhancement_recommendations": [
                {{
                    "type": "enhancement type",
                    "description": "detailed description",
                    "priority": "high/medium/low",
                    "expected_improvement": "description of expected result"
                }},
                {{
                    "type": "enhancement type",
                    "description": "detailed description",
                    "priority": "high/medium/low",
                    "expected_improvement": "description of expected result"
                }}
            ],
            "processing_chain": [
                "step1",
                "step2",
                "step3"
            ],
            "tool_recommendations": [
                "tool1",
                "tool2"
            ]
        }}
        """
        
        return prompt
    
    def _parse_analysis_response(self, response_text):
        """
        Parse and structure the Gemini API response for audio analysis.
        
        Args:
            response_text (str): Raw response text from Gemini API.
            
        Returns:
            dict: Structured analysis data.
        """
        try:
            # Extract JSON from the response text
            structured_data = self._extract_json_from_text(response_text)
            return structured_data
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            # Fallback to returning the raw text
            return {"raw_analysis": response_text}
    
    def _parse_enhancement_response(self, response_text):
        """
        Parse and structure the Gemini API response for enhancement suggestions.
        
        Args:
            response_text (str): Raw response text from Gemini API.
            
        Returns:
            dict: Structured enhancement data.
        """
        try:
            # Extract JSON from the response text
            structured_data = self._extract_json_from_text(response_text)
            return structured_data
        except Exception as e:
            logger.error(f"Error parsing enhancement response: {e}")
            # Fallback to returning the raw text
            return {"raw_suggestions": response_text}
    
    def _extract_json_from_text(self, text):
        """
        Extract a JSON object from text that might contain additional content.
        
        Args:
            text (str): Text potentially containing JSON.
            
        Returns:
            dict: Extracted JSON as a Python dictionary.
        """
        try:
            # Look for content between { and }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            # If no JSON object found, try to parse the entire text
            return json.loads(text)
        
        except json.JSONDecodeError:
            logger.warning("Failed to extract JSON from text")
            # Return a simplified structure with the raw text
            return {"parsed_text": text}