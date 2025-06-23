import os
import numpy as np
import librosa
import soundfile as sf
from google.cloud import storage

class AudioEnhancer:
    """Audio enhancement engine for SoundScaffold.
    
    This class provides functionality to enhance audio files with various processing
    algorithms optimized for media production.
    """
    
    DEFAULT_ENHANCEMENT_CONFIG = {
        'sample_rate': 44100,
        'bit_depth': 24,
        'dialogue_clarity': {
            'eq_boost': [800, 2500],  # Hz
            'compression_ratio': 2.5,
            'noise_reduction': 0.3
        },
        'ambient_enhancement': {
            'eq_boost': [200, 6000],  # Hz
            'reverb_mix': 0.2
        },
        'music_enhancement': {
            'eq_boost': [60, 8000],  # Hz
            'stereo_width': 0.3
        },
        'output_formats': ['wav', 'mp3', 'flac']
    }
    
    def __init__(self, enhancement_config=None):
        """Initialize the AudioEnhancer with optional custom configuration.
        
        Args:
            enhancement_config (dict, optional): Enhancement configuration dictionary. Defaults to None.
        """
        self.config = enhancement_config or self.DEFAULT_ENHANCEMENT_CONFIG
        self.storage_client = storage.Client()
    
    def enhance(self, audio_file, enhancement_type, options=None):
        """Apply selected enhancement to audio file.
        
        Args:
            audio_file (str): Path to the audio file.
            enhancement_type (str): Type of enhancement to apply ('dialogue_clarity', 'ambient_enhancement', 'music_enhancement', etc.).
            options (dict, optional): Additional options for the enhancement. Defaults to None.
            
        Returns:
            str: Path to the enhanced audio file.
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.config['sample_rate'])
            
            # Apply selected enhancement
            if enhancement_type == 'dialogue_clarity':
                enhanced_audio = self._enhance_dialogue(y, sr, options)
            elif enhancement_type == 'ambient_enhancement':
                enhanced_audio = self._enhance_ambient(y, sr, options)
            elif enhancement_type == 'music_enhancement':
                enhanced_audio = self._enhance_music(y, sr, options)
            elif enhancement_type == 'custom':
                enhanced_audio = self._apply_custom_enhancement(y, sr, options)
            else:
                # Default to general enhancement
                enhanced_audio = self._apply_general_enhancement(y, sr, options)
            
            # Create output filename
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = f"{basename}_{enhancement_type}.wav"
            
            # Save enhanced audio
            sf.write(output_file, enhanced_audio, sr, 'PCM_24')
            
            # Optionally upload to Cloud Storage
            if options and options.get('upload_to_cloud', False):
                bucket_name = options.get('bucket_name', 'sound-scaffold-assets')
                self._upload_to_cloud_storage(output_file, bucket_name)
            
            return output_file
        
        except Exception as e:
            print(f"Error enhancing audio: {e}")
            return None
    
    def denoise(self, audio_file, level=0.5):
        """Remove noise from audio file.
        
        Args:
            audio_file (str): Path to the audio file.
            level (float, optional): Denoising level from 0.0 to 1.0. Defaults to 0.5.
            
        Returns:
            str: Path to the denoised audio file.
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.config['sample_rate'])
            
            # For this example, we'll implement a simple spectral gating noise reduction
            # In a real implementation, this would use more sophisticated algorithms
            
            # Compute spectrogram
            D = librosa.stft(y)
            
            # Compute magnitudes
            mag = np.abs(D)
            
            # Estimate noise floor
            noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
            
            # Apply spectral gating
            gate_threshold = noise_floor * (1.0 + level * 3.0)  # Scale threshold by level
            mask = (mag > gate_threshold).astype(float)
            
            # Apply soft mask to spectrogram
            masked_mag = mag * mask
            masked_D = masked_mag * np.exp(1j * np.angle(D))
            
            # Invert back to time domain
            denoised_audio = librosa.istft(masked_D, length=len(y))
            
            # Create output filename
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = f"{basename}_denoised.wav"
            
            # Save denoised audio
            sf.write(output_file, denoised_audio, sr, 'PCM_24')
            
            return output_file
        
        except Exception as e:
            print(f"Error denoising audio: {e}")
            return None
    
    def normalize(self, audio_file, target_level=-18):
        """Normalize audio levels to target loudness.
        
        Args:
            audio_file (str): Path to the audio file.
            target_level (int, optional): Target loudness level in dB LUFS. Defaults to -18.
            
        Returns:
            str: Path to the normalized audio file.
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=self.config['sample_rate'])
            
            # For this example, we'll implement simple peak normalization
            # In a real implementation, this would use LUFS normalization
            
            # Compute current peak level
            current_peak = np.max(np.abs(y))
            
            # Compute target peak level (rough approximation from LUFS)
            target_peak = 10 ** (target_level / 20)  # Convert dB to linear scale
            
            # Apply gain adjustment
            if current_peak > 0:
                gain = target_peak / current_peak
                normalized_audio = y * gain
            else:
                normalized_audio = y
            
            # Create output filename
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = f"{basename}_normalized.wav"
            
            # Save normalized audio
            sf.write(output_file, normalized_audio, sr, 'PCM_24')
            
            return output_file
        
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return None
    
    def _enhance_dialogue(self, audio, sr, options=None):
        """Enhance dialogue clarity in audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            options (dict, optional): Additional options. Defaults to None.
            
        Returns:
            numpy.ndarray: Enhanced audio signal.
        """
        # Get options with defaults
        opts = options or {}
        level = opts.get('level', 0.7)  # Enhancement level from 0.0 to 1.0
        
        # Apply EQ boost to dialogue frequencies
        boost_freqs = self.config['dialogue_clarity']['eq_boost']
        audio_eq = self._apply_eq_boost(audio, sr, boost_freqs, level)
        
        # Apply compression to improve clarity
        compression_ratio = self.config['dialogue_clarity']['compression_ratio'] * level
        audio_compressed = self._apply_compression(audio_eq, compression_ratio)
        
        # Apply subtle noise reduction
        noise_reduction = self.config['dialogue_clarity']['noise_reduction'] * level
        enhanced_audio = self._apply_noise_reduction(audio_compressed, noise_reduction)
        
        return enhanced_audio
    
    def _enhance_ambient(self, audio, sr, options=None):
        """Enhance ambient sounds in audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            options (dict, optional): Additional options. Defaults to None.
            
        Returns:
            numpy.ndarray: Enhanced audio signal.
        """
        # Get options with defaults
        opts = options or {}
        level = opts.get('level', 0.7)  # Enhancement level from 0.0 to 1.0
        
        # Apply EQ boost to ambient frequencies
        boost_freqs = self.config['ambient_enhancement']['eq_boost']
        audio_eq = self._apply_eq_boost(audio, sr, boost_freqs, level)
        
        # Apply subtle reverb
        reverb_mix = self.config['ambient_enhancement']['reverb_mix'] * level
        enhanced_audio = self._apply_reverb(audio_eq, reverb_mix, sr)
        
        return enhanced_audio
    
    def _enhance_music(self, audio, sr, options=None):
        """Enhance music in audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            options (dict, optional): Additional options. Defaults to None.
            
        Returns:
            numpy.ndarray: Enhanced audio signal.
        """
        # Get options with defaults
        opts = options or {}
        level = opts.get('level', 0.7)  # Enhancement level from 0.0 to 1.0
        
        # Apply EQ boost to music frequencies
        boost_freqs = self.config['music_enhancement']['eq_boost']
        audio_eq = self._apply_eq_boost(audio, sr, boost_freqs, level)
        
        # Enhance stereo width if audio is stereo
        if len(audio.shape) > 1 and audio.shape[1] == 2:
            stereo_width = self.config['music_enhancement']['stereo_width'] * level
            enhanced_audio = self._enhance_stereo_width(audio_eq, stereo_width)
        else:
            enhanced_audio = audio_eq
        
        return enhanced_audio
    
    def _apply_general_enhancement(self, audio, sr, options=None):
        """Apply general enhancement to audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            options (dict, optional): Additional options. Defaults to None.
            
        Returns:
            numpy.ndarray: Enhanced audio signal.
        """
        # Get options with defaults
        opts = options or {}
        level = opts.get('level', 0.5)  # Enhancement level from 0.0 to 1.0
        
        # Apply gentle EQ enhancement across frequency spectrum
        boost_freqs = [100, 1000, 5000]  # Low, mid, high
        audio_eq = self._apply_eq_boost(audio, sr, boost_freqs, level * 0.5)
        
        # Apply light compression
        compression_ratio = 1.5 * level
        audio_compressed = self._apply_compression(audio_eq, compression_ratio)
        
        # Apply very subtle noise reduction
        enhanced_audio = self._apply_noise_reduction(audio_compressed, level * 0.2)
        
        return enhanced_audio
    
    def _apply_custom_enhancement(self, audio, sr, options):
        """Apply custom enhancement based on provided options.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            options (dict): Custom enhancement options.
            
        Returns:
            numpy.ndarray: Enhanced audio signal.
        """
        if not options:
            return audio
        
        enhanced_audio = np.copy(audio)
        
        # Apply EQ if specified
        if 'eq_boost' in options:
            enhanced_audio = self._apply_eq_boost(
                enhanced_audio, sr, options['eq_boost'], options.get('eq_level', 0.7)
            )
        
        # Apply compression if specified
        if 'compression_ratio' in options:
            enhanced_audio = self._apply_compression(
                enhanced_audio, options['compression_ratio']
            )
        
        # Apply noise reduction if specified
        if 'noise_reduction' in options:
            enhanced_audio = self._apply_noise_reduction(
                enhanced_audio, options['noise_reduction']
            )
        
        # Apply reverb if specified
        if 'reverb_mix' in options:
            enhanced_audio = self._apply_reverb(
                enhanced_audio, options['reverb_mix'], sr
            )
        
        return enhanced_audio
    
    def _apply_eq_boost(self, audio, sr, frequencies, level=0.7):
        """Apply EQ boost at specified frequencies.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            sr (int): Sample rate.
            frequencies (list): Frequencies to boost in Hz.
            level (float, optional): Boost level from 0.0 to 1.0. Defaults to 0.7.
            
        Returns:
            numpy.ndarray: EQ-boosted audio signal.
        """
        # This is a simplified EQ implementation
        # In a real implementation, this would use proper filters
        
        # Convert to frequency domain
        D = librosa.stft(audio)
        
        # Get frequency bins
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0] * 2 - 2)
        
        # Create EQ curve (initialize with ones)
        eq_curve = np.ones(len(freq_bins))
        
        # Add boosts for each frequency
        for freq in frequencies:
            # Find closest frequency bin
            bin_idx = np.argmin(np.abs(freq_bins - freq))
            
            # Apply bell curve around the frequency
            curve_width = len(freq_bins) // 20  # Width of the curve
            for i in range(max(0, bin_idx - curve_width), min(len(freq_bins), bin_idx + curve_width + 1)):
                # Calculate distance from center (0 to 1)
                dist = abs(i - bin_idx) / curve_width
                # Apply bell curve
                if dist < 1:
                    boost = (1 - dist**2) * level * 3  # Max boost of 3dB * level
                    eq_curve[i] += boost
        
        # Apply EQ curve to spectrum
        eq_curve = eq_curve.reshape(-1, 1)  # Reshape for broadcasting
        D_eq = D * eq_curve[:D.shape[0]]
        
        # Convert back to time domain
        audio_eq = librosa.istft(D_eq, length=len(audio))
        
        return audio_eq
    
    def _apply_compression(self, audio, ratio):
        """Apply dynamic range compression to audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            ratio (float): Compression ratio (higher = more compression).
            
        Returns:
            numpy.ndarray: Compressed audio signal.
        """
        # Simple compressor implementation
        # In a real implementation, this would be more sophisticated
        
        # Set threshold at 70% of peak amplitude
        threshold = 0.7 * np.max(np.abs(audio))
        
        # Apply compression
        compressed = np.copy(audio)
        mask = np.abs(compressed) > threshold
        
        if np.any(mask):
            # Compute gain reduction
            gain_reduction = (np.abs(compressed[mask]) - threshold) * (1 - 1/ratio)
            
            # Apply gain reduction
            compressed[mask] = np.sign(compressed[mask]) * (np.abs(compressed[mask]) - gain_reduction)
            
            # Normalize to original peak level
            compressed = compressed * (np.max(np.abs(audio)) / np.max(np.abs(compressed)))
        
        return compressed
    
    def _apply_noise_reduction(self, audio, amount):
        """Apply noise reduction to audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            amount (float): Noise reduction amount from 0.0 to 1.0.
            
        Returns:
            numpy.ndarray: Noise-reduced audio signal.
        """
        # Simple spectral subtraction noise reduction
        # In a real implementation, this would be more sophisticated
        
        # Convert to frequency domain
        D = librosa.stft(audio)
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Estimate noise floor (assuming first 100ms is noise)
        n_frames_noise = int(0.1 * librosa.get_samplerate() / D.shape[1])
        noise_floor = np.mean(mag[:, :n_frames_noise], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        gain = (mag - (noise_floor * amount)) / mag
        gain = np.maximum(gain, 0.1)  # Limit attenuation
        
        # Apply gain to magnitude
        mag_reduced = mag * gain
        
        # Reconstruct signal
        D_reduced = mag_reduced * np.exp(1j * phase)
        audio_reduced = librosa.istft(D_reduced, length=len(audio))
        
        return audio_reduced
    
    def _apply_reverb(self, audio, mix, sr):
        """Apply reverb effect to audio.
        
        Args:
            audio (numpy.ndarray): Audio signal.
            mix (float): Reverb wet/dry mix from 0.0 to 1.0.
            sr (int): Sample rate.
            
        Returns:
            numpy.ndarray: Reverb-applied audio signal.
        """
        # Simple convolution reverb simulation
        # In a real implementation, this would use actual impulse responses
        
        # Create a simple impulse response
        decay = 0.6
        ir_duration = 1.5  # seconds
        ir_length = int(ir_duration * sr)
        ir = np.exp(-np.linspace(0, decay * 10, ir_length))
        ir = ir * np.random.randn(ir_length)  # Add some randomness
        ir = ir / np.sum(np.abs(ir))  # Normalize
        
        # Apply convolution reverb
        reverb = np.convolve(audio, ir, mode='full')[:len(audio)]
        
        # Mix with dry signal
        result = (1 - mix) * audio + mix * reverb
        
        # Normalize to original level
        result = result * (np.max(np.abs(audio)) / np.max(np.abs(result)))
        
        return result
    
    def _enhance_stereo_width(self, audio, width):
        """Enhance stereo width of audio.
        
        Args:
            audio (numpy.ndarray): Stereo audio signal (2 channels).
            width (float): Width enhancement amount from 0.0 to 1.0.
            
        Returns:
            numpy.ndarray: Width-enhanced audio signal.
        """
        # Only process if audio is stereo
        if len(audio.shape) < 2 or audio.shape[1] != 2:
            return audio
        
        # Split into mid and side channels
        mid = (audio[:, 0] + audio[:, 1]) / 2
        side = (audio[:, 0] - audio[:, 1]) / 2
        
        # Enhance side channel
        side_enhanced = side * (1 + width)
        
        # Recombine to stereo
        left = mid + side_enhanced
        right = mid - side_enhanced
        
        # Stack channels
        enhanced = np.stack([left, right], axis=1)
        
        # Normalize to original level
        max_original = np.max(np.abs(audio))
        max_enhanced = np.max(np.abs(enhanced))
        if max_enhanced > 0:
            enhanced = enhanced * (max_original / max_enhanced)
        
        return enhanced
    
    def _upload_to_cloud_storage(self, file_path, bucket_name):
        """Upload file to Google Cloud Storage.
        
        Args:
            file_path (str): Path to the file to upload.
            bucket_name (str): Name of the Cloud Storage bucket.
            
        Returns:
            str: Public URL of the uploaded file.
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob_name = f"enhanced_audio/{os.path.basename(file_path)}"
            blob = bucket.blob(blob_name)
            
            # Upload file
            blob.upload_from_filename(file_path)
            
            # Make public
            blob.make_public()
            
            return blob.public_url
        
        except Exception as e:
            print(f"Error uploading to Cloud Storage: {e}")
            return None
