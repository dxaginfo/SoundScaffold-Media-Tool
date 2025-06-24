"""
File handling utilities for SoundScaffold.

This module provides functions for handling audio files,
including validation, loading, and saving.
"""

import os
import logging
from typing import List, Optional, Tuple
import soundfile as sf
import librosa
import numpy as np
from google.cloud import storage

logger = logging.getLogger(__name__)

# Supported audio file formats
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that the file exists and is a supported audio format.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if file is valid, raises exception otherwise
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a supported audio format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    _, extension = os.path.splitext(file_path)
    if extension.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {extension}. "
            f"Supported formats are: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Try to read a small portion of the file to ensure it's valid
    try:
        y, sr = librosa.load(file_path, sr=None, duration=0.1)
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}")
    
    return True


def load_audio_file(
    file_path: str, 
    sr: Optional[int] = None, 
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate. If None, uses the file's sample rate
        mono: If True, converts audio to mono
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a supported audio format
    """
    validate_audio_file(file_path)
    
    logger.info(f"Loading audio file: {file_path} (sr={sr}, mono={mono})")
    y, sr_orig = librosa.load(file_path, sr=sr, mono=mono)
    
    logger.info(f"Loaded audio: shape={y.shape}, sr={sr_orig}")
    return y, sr_orig


def save_audio_file(
    y: np.ndarray, 
    sr: int, 
    file_path: str, 
    format: Optional[str] = None
) -> str:
    """
    Save audio data to a file.
    
    Args:
        y: Audio data
        sr: Sample rate
        file_path: Path to save the audio file
        format: Audio format (wav, mp3, etc.). If None, inferred from file_path
        
    Returns:
        Path to the saved file
        
    Raises:
        ValueError: If the format is not supported
    """
    if format is None:
        _, extension = os.path.splitext(file_path)
        format = extension[1:].lower()
    
    if format not in [fmt[1:] for fmt in SUPPORTED_FORMATS]:
        raise ValueError(
            f"Unsupported audio format: {format}. "
            f"Supported formats are: {', '.join(fmt[1:] for fmt in SUPPORTED_FORMATS)}"
        )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    logger.info(f"Saving audio file: {file_path} (format={format})")
    sf.write(file_path, y, sr, format=format)
    
    return file_path


def upload_to_cloud_storage(
    local_file_path: str, 
    bucket_name: str, 
    blob_name: Optional[str] = None
) -> str:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to the local file
        bucket_name: Name of the GCS bucket
        blob_name: Name to use for the blob. If None, uses the filename
        
    Returns:
        Public URL of the uploaded file
        
    Raises:
        FileNotFoundError: If the local file does not exist
    """
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")
    
    if blob_name is None:
        blob_name = os.path.basename(local_file_path)
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload the file
    logger.info(f"Uploading {local_file_path} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(local_file_path)
    
    # Make the blob publicly accessible
    blob.make_public()
    
    return blob.public_url


def download_from_cloud_storage(
    bucket_name: str, 
    blob_name: str, 
    local_file_path: str
) -> str:
    """
    Download a file from Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to download
        local_file_path: Path to save the file locally
        
    Returns:
        Path to the downloaded file
        
    Raises:
        FileNotFoundError: If the blob does not exist
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Check if blob exists
    if not blob.exists():
        raise FileNotFoundError(
            f"Blob not found: gs://{bucket_name}/{blob_name}"
        )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(local_file_path)), exist_ok=True)
    
    # Download the file
    logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {local_file_path}")
    blob.download_to_filename(local_file_path)
    
    return local_file_path


def list_audio_files(
    directory: str, 
    recursive: bool = True
) -> List[str]:
    """
    List all audio files in a directory.
    
    Args:
        directory: Directory to search for audio files
        recursive: If True, search recursively through subdirectories
        
    Returns:
        List of paths to audio files
        
    Raises:
        FileNotFoundError: If the directory does not exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    audio_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in SUPPORTED_FORMATS:
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            _, ext = os.path.splitext(file)
            if ext.lower() in SUPPORTED_FORMATS:
                audio_files.append(os.path.join(directory, file))
    
    return audio_files