"""
AnalysisResult model for SoundScaffold

This module defines the AnalysisResult class for storing audio analysis results.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


class AnalysisResult:
    """
    Model class for storing audio analysis results.
    
    Attributes:
        id (str): Unique identifier for the analysis result.
        file_path (str): Path to the analyzed audio file.
        duration (float): Duration of the audio file in seconds.
        sample_rate (int): Sample rate of the audio file in Hz.
        scenes (List[Dict[str, Any]]): List of detected audio scenes.
        quality_metrics (Dict[str, Any]): Audio quality metrics.
        features (Dict[str, Any]): Extracted audio features.
        analysis_date (datetime): Timestamp of when the analysis was performed.
    """
    
    def __init__(
        self,
        file_path: str,
        duration: float,
        sample_rate: int,
        scenes: List[Dict[str, Any]],
        quality_metrics: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        analysis_date: Optional[datetime] = None
    ):
        """
        Initialize an AnalysisResult instance.
        
        Args:
            file_path: Path to the analyzed audio file.
            duration: Duration of the audio file in seconds.
            sample_rate: Sample rate of the audio file in Hz.
            scenes: List of detected audio scenes.
            quality_metrics: Audio quality metrics.
            features: Extracted audio features (optional).
            id: Unique identifier (generated if not provided).
            analysis_date: Timestamp (current time if not provided).
        """
        self.id = id or str(uuid.uuid4())
        self.file_path = file_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.scenes = scenes
        self.quality_metrics = quality_metrics
        self.features = features
        self.analysis_date = analysis_date or datetime.now()
    
    def to_dict(self, include_features: bool = False) -> Dict[str, Any]:
        """
        Convert the analysis result to a dictionary.
        
        Args:
            include_features: Whether to include the features in the result.
                Features can be large, so they are excluded by default.
        
        Returns:
            Dictionary representation of the analysis result.
        """
        result = {
            'id': self.id,
            'file_path': self.file_path,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'scenes': self.scenes,
            'quality_metrics': self.quality_metrics,
            'analysis_date': self.analysis_date.isoformat()
        }
        
        if include_features and self.features:
            # Filter out numpy arrays and other non-serializable objects
            serializable_features = {}
            for key, value in self.features.items():
                # Skip waveform and other large arrays
                if key not in ['waveform', 'harmonic', 'percussive', 
                              'mel_spectrogram', 'mfcc']:
                    try:
                        # Check if value is JSON serializable
                        json.dumps({key: value})
                        serializable_features[key] = value
                    except (TypeError, OverflowError):
                        # If not serializable, convert to string or skip
                        if hasattr(value, 'tolist'):
                            serializable_features[key] = value.tolist()
                        else:
                            serializable_features[key] = str(value)
            
            result['features'] = serializable_features
        
        return result
    
    def to_json(self, include_features: bool = False) -> str:
        """
        Convert the analysis result to a JSON string.
        
        Args:
            include_features: Whether to include the features in the result.
        
        Returns:
            JSON string representation of the analysis result.
        """
        return json.dumps(self.to_dict(include_features=include_features), 
                         default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """
        Create an AnalysisResult instance from a dictionary.
        
        Args:
            data: Dictionary containing analysis result data.
        
        Returns:
            AnalysisResult instance.
        """
        # Convert ISO date string back to datetime
        if 'analysis_date' in data and isinstance(data['analysis_date'], str):
            data['analysis_date'] = datetime.fromisoformat(data['analysis_date'])
        
        return cls(
            file_path=data.get('file_path', ''),
            duration=data.get('duration', 0.0),
            sample_rate=data.get('sample_rate', 44100),
            scenes=data.get('scenes', []),
            quality_metrics=data.get('quality_metrics', {}),
            features=data.get('features', None),
            id=data.get('id', None),
            analysis_date=data.get('analysis_date', None)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AnalysisResult':
        """
        Create an AnalysisResult instance from a JSON string.
        
        Args:
            json_str: JSON string containing analysis result data.
        
        Returns:
            AnalysisResult instance.
        """
        return cls.from_dict(json.loads(json_str))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis result.
        
        Returns:
            Dictionary containing a summary of the analysis result.
        """
        scene_types = {}
        for scene in self.scenes:
            scene_type = scene.get('type', 'unknown')
            if scene_type in scene_types:
                scene_types[scene_type] += 1
            else:
                scene_types[scene_type] = 1
        
        return {
            'file_path': self.file_path,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'num_scenes': len(self.scenes),
            'scene_types': scene_types,
            'overall_quality': self.quality_metrics.get('overall_quality', 'unknown')
        }
    
    def __str__(self) -> str:
        """String representation of the analysis result."""
        summary = self.get_summary()
        return (f"AnalysisResult(id={self.id}, "
                f"file='{self.file_path}', "
                f"duration={self.duration:.2f}s, "
                f"scenes={len(self.scenes)}, "
                f"quality={summary['overall_quality']})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the analysis result."""
        return (f"AnalysisResult(id='{self.id}', "
                f"file_path='{self.file_path}', "
                f"duration={self.duration}, "
                f"sample_rate={self.sample_rate}, "
                f"scenes={len(self.scenes)}, "
                f"analysis_date='{self.analysis_date}')")