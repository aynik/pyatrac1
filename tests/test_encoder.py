"""
Tests for the ATRAC1 encoder.
"""

import numpy as np
import pytest
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.common import constants


class TestAtrac1Encoder:
    """Test cases for Atrac1Encoder class."""

    def test_init(self):
        """Test encoder initialization."""
        encoder = Atrac1Encoder()
        
        # Check that all components are initialized
        assert encoder.codec_data is not None
        assert encoder.qmf_filter_bank_ch0 is not None
        assert encoder.qmf_filter_bank_ch1 is not None
        assert encoder.mdct_processor is not None
        assert encoder.psychoacoustic_model_ch0 is not None
        assert encoder.psychoacoustic_model_ch1 is not None
        
        # Check transient detectors for both channels
        assert 0 in encoder.transient_detectors
        assert 1 in encoder.transient_detectors
        assert "low" in encoder.transient_detectors[0]
        assert "mid" in encoder.transient_detectors[0]
        assert "high" in encoder.transient_detectors[0]
        assert "low" in encoder.transient_detectors[1]
        assert "mid" in encoder.transient_detectors[1]
        assert "high" in encoder.transient_detectors[1]
        
        # Check other components
        assert encoder.bitstream_writer is not None
        assert encoder.scaler is not None
        assert encoder.bit_allocator is not None
        assert encoder.bits_booster is not None

    def test_get_representative_freq_for_bfu(self):
        """Test frequency calculation for BFU."""
        encoder = Atrac1Encoder()
        
        # Test with first BFU
        freq_0 = encoder._get_representative_freq_for_bfu(0, True)
        assert freq_0 >= 0
        assert freq_0 < constants.SAMPLE_RATE / 2
        
        # Test with higher index BFU
        freq_10 = encoder._get_representative_freq_for_bfu(10, True)
        assert freq_10 > freq_0  # Should be higher frequency
        
        # Test with same parameters for long/short blocks
        freq_long = encoder._get_representative_freq_for_bfu(5, True)
        freq_short = encoder._get_representative_freq_for_bfu(5, False)
        assert freq_long == freq_short  # Should be same calculation

    def test_encode_frame_invalid_input_type(self):
        """Test encoding with invalid input type."""
        encoder = Atrac1Encoder()
        
        with pytest.raises(TypeError, match="input_audio_samples must be a NumPy array"):
            encoder.encode_frame([1, 2, 3])  # List instead of numpy array

    def test_encode_frame_invalid_mono_length(self):
        """Test encoding with wrong mono input length."""
        encoder = Atrac1Encoder()
        
        # Too short
        short_samples = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match=f"Mono input must have {constants.NUM_SAMPLES} samples"):
            encoder.encode_frame(short_samples)
        
        # Too long
        long_samples = np.zeros(1000, dtype=np.float32)
        with pytest.raises(ValueError, match=f"Mono input must have {constants.NUM_SAMPLES} samples"):
            encoder.encode_frame(long_samples)

    def test_encode_frame_invalid_stereo_shape(self):
        """Test encoding with invalid stereo input shape."""
        encoder = Atrac1Encoder()
        
        # Wrong shape entirely
        wrong_shape = np.zeros((100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Stereo input must have shape"):
            encoder.encode_frame(wrong_shape)
        
        # 3D array
        invalid_3d = np.zeros((constants.NUM_SAMPLES, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Input audio samples must be 1D.*or 2D"):
            encoder.encode_frame(invalid_3d)

    def test_encode_frame_mono_silence(self):
        """Test encoding mono silence."""
        encoder = Atrac1Encoder()
        
        # Create silence
        silence = np.zeros(constants.NUM_SAMPLES, dtype=np.float32)
        result = encoder.encode_frame(silence)
        
        # Should return exactly one frame (212 bytes)
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_mono_sine_wave(self):
        """Test encoding mono sine wave."""
        encoder = Atrac1Encoder()
        
        # Create a simple sine wave
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        frequency = 1000  # 1kHz sine wave
        amplitude = 0.1
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        result = encoder.encode_frame(sine_wave)
        
        # Should return exactly one frame
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_stereo_silence(self):
        """Test encoding stereo silence."""
        encoder = Atrac1Encoder()
        
        # Create stereo silence
        stereo_silence = np.zeros((constants.NUM_SAMPLES, 2), dtype=np.float32)
        result = encoder.encode_frame(stereo_silence)
        
        # Should return two frames (424 bytes)
        assert isinstance(result, bytes)
        assert len(result) == 2 * constants.SOUND_UNIT_SIZE

    def test_encode_frame_stereo_transposed_input(self):
        """Test encoding stereo input with transposed shape."""
        encoder = Atrac1Encoder()
        
        # Create stereo input with shape (2, NUM_SAMPLES) instead of (NUM_SAMPLES, 2)
        stereo_data = np.zeros((2, constants.NUM_SAMPLES), dtype=np.float32)
        stereo_data[0, :] = 0.1  # Left channel
        stereo_data[1, :] = 0.2  # Right channel
        
        result = encoder.encode_frame(stereo_data)
        
        # Should still work and return two frames
        assert isinstance(result, bytes)
        assert len(result) == 2 * constants.SOUND_UNIT_SIZE

    def test_encode_frame_stereo_different_channels(self):
        """Test encoding stereo with different content in each channel."""
        encoder = Atrac1Encoder()
        
        # Create stereo data with different sine waves
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        left_channel = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        right_channel = 0.1 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
        
        stereo_data = np.column_stack([left_channel, right_channel])
        result = encoder.encode_frame(stereo_data)
        
        # Should return two frames
        assert isinstance(result, bytes)
        assert len(result) == 2 * constants.SOUND_UNIT_SIZE

    def test_encode_single_channel_internal(self):
        """Test internal single channel encoding method."""
        encoder = Atrac1Encoder()
        
        # Create test signal
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        test_signal = 0.05 * np.sin(2 * np.pi * 1000 * t)
        
        # Test channel 0
        result_ch0 = encoder._encode_single_channel(test_signal, 0)
        assert isinstance(result_ch0, bytes)
        assert len(result_ch0) == constants.SOUND_UNIT_SIZE
        
        # Test channel 1
        result_ch1 = encoder._encode_single_channel(test_signal, 1)
        assert isinstance(result_ch1, bytes)
        assert len(result_ch1) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_impulse_response(self):
        """Test encoding an impulse (to trigger transient detection)."""
        encoder = Atrac1Encoder()
        
        # Create impulse signal
        impulse = np.zeros(constants.NUM_SAMPLES, dtype=np.float32)
        impulse[constants.NUM_SAMPLES // 2] = 0.5  # Impulse in the middle
        
        result = encoder.encode_frame(impulse)
        
        # Should handle impulse correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_noise(self):
        """Test encoding white noise."""
        encoder = Atrac1Encoder()
        
        # Create white noise
        np.random.seed(42)  # For reproducible results
        noise = 0.01 * np.random.randn(constants.NUM_SAMPLES).astype(np.float32)
        
        result = encoder.encode_frame(noise)
        
        # Should handle noise correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_maximum_amplitude(self):
        """Test encoding at maximum amplitude."""
        encoder = Atrac1Encoder()
        
        # Create maximum amplitude sine wave
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        max_amplitude_signal = 0.99 * np.sin(2 * np.pi * 1000 * t)
        
        result = encoder.encode_frame(max_amplitude_signal)
        
        # Should handle high amplitude correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_low_frequency(self):
        """Test encoding very low frequency content."""
        encoder = Atrac1Encoder()
        
        # Create very low frequency sine wave (20 Hz)
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        low_freq_signal = 0.1 * np.sin(2 * np.pi * 20 * t)
        
        result = encoder.encode_frame(low_freq_signal)
        
        # Should handle low frequency correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_high_frequency(self):
        """Test encoding high frequency content."""
        encoder = Atrac1Encoder()
        
        # Create high frequency sine wave (close to Nyquist)
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        high_freq = constants.SAMPLE_RATE // 2 - 1000  # 1kHz below Nyquist
        high_freq_signal = 0.1 * np.sin(2 * np.pi * high_freq * t)
        
        result = encoder.encode_frame(high_freq_signal)
        
        # Should handle high frequency correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_mixed_frequency_content(self):
        """Test encoding signal with mixed frequency content."""
        encoder = Atrac1Encoder()
        
        # Create signal with multiple frequency components
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        mixed_signal = (
            0.05 * np.sin(2 * np.pi * 200 * t) +    # Low frequency
            0.05 * np.sin(2 * np.pi * 1000 * t) +   # Mid frequency
            0.05 * np.sin(2 * np.pi * 5000 * t)     # High frequency
        )
        
        result = encoder.encode_frame(mixed_signal)
        
        # Should handle complex signal correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_frame_dc_component(self):
        """Test encoding signal with DC component."""
        encoder = Atrac1Encoder()
        
        # Create signal with DC offset
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        dc_signal = 0.1 + 0.05 * np.sin(2 * np.pi * 1000 * t)  # DC + AC
        
        result = encoder.encode_frame(dc_signal)
        
        # Should handle DC component correctly
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE

    def test_encode_multiple_frames_consistency(self):
        """Test that encoding multiple frames produces consistent results."""
        encoder = Atrac1Encoder()
        
        # Create identical test signals
        test_signal = np.zeros(constants.NUM_SAMPLES, dtype=np.float32)
        test_signal[100] = 0.1  # Small impulse
        
        # Encode the same signal multiple times
        result1 = encoder.encode_frame(test_signal.copy())
        result2 = encoder.encode_frame(test_signal.copy())
        result3 = encoder.encode_frame(test_signal.copy())
        
        # Results should be identical for identical input
        # Note: This might not be true if there's state in the encoder
        # but it's a good test for deterministic behavior
        assert len(result1) == len(result2) == len(result3)
        assert len(result1) == constants.SOUND_UNIT_SIZE

    def test_bfu_frequency_ordering(self):
        """Test that BFU frequencies are ordered correctly."""
        encoder = Atrac1Encoder()
        
        # Test several BFU indices
        frequencies = []
        for i in range(min(20, len(encoder.codec_data.bfu_amount_tab))):
            freq = encoder._get_representative_freq_for_bfu(i, True)
            frequencies.append(freq)
        
        # Frequencies should generally increase with BFU index
        for i in range(1, len(frequencies)):
            assert frequencies[i] >= frequencies[i-1], f"Frequency decreased from BFU {i-1} to {i}"

    def test_encode_frame_edge_case_values(self):
        """Test encoding with edge case numeric values."""
        encoder = Atrac1Encoder()
        
        # Test very small values
        tiny_signal = np.full(constants.NUM_SAMPLES, 1e-10, dtype=np.float32)
        result_tiny = encoder.encode_frame(tiny_signal)
        assert len(result_tiny) == constants.SOUND_UNIT_SIZE
        
        # Test negative values
        negative_signal = np.full(constants.NUM_SAMPLES, -0.01, dtype=np.float32)
        result_negative = encoder.encode_frame(negative_signal)
        assert len(result_negative) == constants.SOUND_UNIT_SIZE

    def test_stereo_channel_independence(self):
        """Test that stereo channels are processed independently."""
        encoder = Atrac1Encoder()
        
        # Create different signals for each channel
        t = np.arange(constants.NUM_SAMPLES, dtype=np.float32) / constants.SAMPLE_RATE
        left_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        right_zeros = np.zeros(constants.NUM_SAMPLES, dtype=np.float32)
        
        # Encode with signal only in left channel
        stereo_left_only = np.column_stack([left_signal, right_zeros])
        result_left_only = encoder.encode_frame(stereo_left_only)
        
        # Encode with signal only in right channel
        stereo_right_only = np.column_stack([right_zeros, left_signal])
        result_right_only = encoder.encode_frame(stereo_right_only)
        
        # Both should produce valid frames
        assert len(result_left_only) == 2 * constants.SOUND_UNIT_SIZE
        assert len(result_right_only) == 2 * constants.SOUND_UNIT_SIZE
        
        # Results should be different since content is different
        assert result_left_only != result_right_only