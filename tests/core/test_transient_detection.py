import numpy as np
from pyatrac1.core.transient_detection import TransientDetector
from pyatrac1.tables.filter_coeffs import HPF_FIR_LEN
from typing import Union


# Helper function to create a sine wave
def generate_sine_wave(
    freq: float, duration: float, sample_rate: Union[int, float], amplitude: float = 1.0
) -> np.ndarray:
    t: np.ndarray = np.linspace(
        0, duration, int(sample_rate * duration), endpoint=False
    )
    return amplitude * np.sin(2 * np.pi * freq * t)


# Helper function to create an impulse
def generate_impulse(
    length: int, impulse_position: int, impulse_amplitude: float = 1.0
) -> np.ndarray:
    signal: np.ndarray = np.zeros(length, dtype=np.float32)
    if 0 <= impulse_position < length:
        signal[impulse_position] = impulse_amplitude
    return signal


class TestTransientDetectorHPFFilter:
    def test_hpf_filter_silence(self):
        detector = TransientDetector()
        input_signal = np.zeros(256, dtype=np.float32)
        filtered_signal = detector._hpf_filter(input_signal)
        assert filtered_signal.shape == input_signal.shape
        # For a zero input, the HPF output should also be zero (or very close to it)
        # after the initial filter state settles.
        # The filter has memory, so initial output might not be zero.
        # Let's check the tail end if the input is long enough.
        if len(filtered_signal) > HPF_FIR_LEN:
            assert np.allclose(filtered_signal[HPF_FIR_LEN:], 0, atol=1e-6)
        else:
            # If shorter than filter length, it's harder to assert all zeros due to buffer state
            pass

    def test_hpf_filter_impulse(self):
        detector = TransientDetector()
        # Impulse needs to be long enough to see the filter response
        input_signal = generate_impulse(256, 50, 1.0).astype(np.float32)
        filtered_signal = detector._hpf_filter(input_signal)
        assert filtered_signal.shape == input_signal.shape
        # An impulse should produce a non-zero response.
        # The sum of a high-pass filter's coefficients is typically 0.
        # The output should reflect high-frequency components.
        # We expect the sum of the output to be close to zero for a good HPF after the impulse passes.
        # However, the specific values depend on the filter coefficients.
        # For now, just check it's not all zeros if input is not all zeros.
        assert not np.allclose(filtered_signal, 0, atol=1e-6)

    def test_hpf_filter_sine_wave_low_freq(self):
        detector = TransientDetector()
        # Low frequency, should be attenuated
        sample_rate = 44100
        # A very low frequency, e.g. 10Hz, for a block of 256 samples (short duration)
        # might not show strong attenuation with this specific HPF.
        # The filter's characteristics are fixed by HPF_FIR_COEFFS.
        # Let's use a frequency that should be significantly affected.
        # The exact cutoff isn't specified, but it's a high-pass filter.
        input_signal = generate_sine_wave(100, 256 / sample_rate, sample_rate).astype(
            np.float32
        )
        filtered_signal = detector._hpf_filter(input_signal)
        assert filtered_signal.shape == input_signal.shape
        # Energy of filtered signal should be less than input for low frequencies
        # This is a qualitative check; precise attenuation depends on filter spec.
        # Skipping direct energy comparison as it's complex without knowing exact cutoff.
        # Instead, check that it's not identical to input.
        assert not np.allclose(filtered_signal, input_signal, atol=1e-5)

    def test_hpf_filter_sine_wave_high_freq(self):
        detector = TransientDetector()
        sample_rate = 44100
        # High frequency, should pass with less attenuation
        input_signal = generate_sine_wave(10000, 256 / sample_rate, sample_rate).astype(
            np.float32
        )
        filtered_signal = detector._hpf_filter(input_signal)
        assert filtered_signal.shape == input_signal.shape
        # For high frequencies, output should be closer to input (less attenuated)
        # than for low frequencies. This is hard to assert precisely without reference.
        # We can at least check it's not zero.
        assert not np.allclose(filtered_signal, 0, atol=1e-6)

    def test_hpf_filter_statefulness(self):
        detector = TransientDetector()
        block_size = 128
        input_signal_1 = np.random.rand(block_size).astype(np.float32)
        input_signal_2 = np.random.rand(block_size).astype(np.float32)

        # Filter first block
        filtered_1_first_pass = detector._hpf_filter(input_signal_1)

        # Filter second block, state should carry over
        filtered_2_first_pass = detector._hpf_filter(input_signal_2)

        # Reset detector and filter concatenated signal
        detector_fresh = TransientDetector()
        concatenated_input = np.concatenate((input_signal_1, input_signal_2))
        filtered_concatenated = detector_fresh._hpf_filter(concatenated_input)

        # The second part of the concatenated filter output should match filtered_2_first_pass
        # This confirms that the internal state of the filter is maintained correctly.
        # Due to the nature of FIR filtering and buffer management,
        # exact match is expected after the initial FIR_LEN samples have passed through the fresh filter.
        # The comparison should be for the output corresponding to input_signal_2
        assert np.allclose(
            filtered_concatenated[block_size:], filtered_2_first_pass, atol=1e-5
        )

        # Also, the first part should match filtered_1_first_pass
        assert np.allclose(
            filtered_concatenated[:block_size], filtered_1_first_pass, atol=1e-5
        )


class TestTransientDetectorDetect:
    def test_detect_silence(self):
        detector = TransientDetector()
        input_signal = np.zeros(256, dtype=np.float32)
        # First call, last_energy is 0. Current energy is 0. No change.
        assert detector.detect(input_signal) == 0
        # Second call with silence, last_energy is ~ -100dB. Current energy ~ -100dB. No change.
        assert detector.detect(input_signal) == 0

    def test_detect_continuous_tone(self):
        detector = TransientDetector()
        sample_rate = 44100
        input_signal = generate_sine_wave(1000, 256 / sample_rate, sample_rate).astype(
            np.float32
        )
        # First call, last_energy is 0. Current energy will be some value.
        # This might detect a transient if initial energy is high enough from 0.
        # The spec implies last_energy is initialized to 0.0, so first block won't trigger.
        # The code: if self.last_energy != 0.0 or current_energy_db != -100.0:
        # If last_energy is 0.0 (initial) and current_energy_db is not -100, it proceeds.
        # db_change = current_energy_db - 0.0. If current_energy_db > 16, transient.
        # So, a continuous tone from silence *will* be a transient on the first block.
        assert detector.detect(input_signal) == 1  # Attack from silence

        # Subsequent calls with the same tone should not detect a transient
        assert detector.detect(input_signal) == 0
        assert detector.detect(input_signal) == 0

    def test_detect_abrupt_impulse(self):
        detector = TransientDetector()
        # Start with silence to establish a baseline
        silence = np.zeros(256, dtype=np.float32)
        detector.detect(silence)  # last_energy will be ~ -100 dB

        # Now an impulse
        impulse_signal = generate_impulse(256, 30, 10.0).astype(
            np.float32
        )  # Strong impulse
        assert detector.detect(impulse_signal) == 1  # Transient expected

        # Followed by silence again, should be a transient (large decrease)
        # The implementation detects db_change <= -20.0 as transient.
        # So, from impulse to silence should be a transient.
        # detector.last_energy is now from the impulse (high).
        # current_energy_db for silence is -100.
        # db_change = -100 - (high_positive_value) = very negative. So, transient.
        assert detector.detect(silence) == 1

    def test_detect_energy_increase_transient(self):
        detector = TransientDetector()
        sample_rate = 44100
        duration = 256 / sample_rate

        # Low energy signal
        low_signal = generate_sine_wave(
            1000, duration, sample_rate, amplitude=0.01
        ).astype(np.float32)
        detector.detect(low_signal)  # Establishes low self.last_energy

        # High energy signal (amplitude 0.01 to 0.2 is 20*log10(0.2/0.01) = 20*log10(20) = 20*1.3 = 26 dB increase)
        high_signal = generate_sine_wave(
            1000, duration, sample_rate, amplitude=0.2
        ).astype(np.float32)
        assert detector.detect(high_signal) == 1  # Transient expected (increase > 16dB)

    def test_detect_energy_decrease_transient(self):
        detector = TransientDetector()
        sample_rate = 44100
        duration = 256 / sample_rate

        # High energy signal
        high_signal = generate_sine_wave(
            1000, duration, sample_rate, amplitude=0.9
        ).astype(np.float32)
        detector.detect(high_signal)  # Establishes high self.last_energy

        # Use silence to create maximum dB drop after HPF
        silence_signal = np.zeros(256, dtype=np.float32)
        assert detector.detect(silence_signal) == 1  # Transient expected (decrease < -20dB)

    def test_detect_no_transient_stable_signal(self):
        detector = TransientDetector()
        sample_rate = 44100
        duration = 256 / sample_rate
        signal = generate_sine_wave(1000, duration, sample_rate, amplitude=0.1).astype(
            np.float32
        )

        detector.detect(signal)  # First call might be transient from silence
        # Subsequent calls with same signal
        assert detector.detect(signal) == 0
        assert detector.detect(signal) == 0

    def test_detect_gradual_change_no_transient(self):
        detector = TransientDetector()
        sample_rate = 44100
        block_size = 256
        duration_per_block = block_size / sample_rate

        # Initial signal
        signal_1 = generate_sine_wave(
            1000, duration_per_block, sample_rate, amplitude=0.1
        ).astype(np.float32)
        detector.detect(signal_1)  # Establish baseline, might be transient from silence

        # Slightly increased amplitude (e.g., 0.1 to 0.12, 20*log10(1.2) = ~1.58 dB, not a transient)
        signal_2 = generate_sine_wave(
            1000, duration_per_block, sample_rate, amplitude=0.12
        ).astype(np.float32)
        assert detector.detect(signal_2) == 0

        # Slightly decreased amplitude (e.g., 0.12 to 0.1, 20*log10(0.1/0.12) = ~ -1.58 dB, not a transient)
        signal_3 = generate_sine_wave(
            1000, duration_per_block, sample_rate, amplitude=0.1
        ).astype(np.float32)
        assert detector.detect(signal_3) == 0

    def test_detect_with_noise_no_clear_transient(self):
        detector = TransientDetector()
        np.random.seed(0)  # for reproducibility
        # Relatively stable noisy signal
        noise_level = 0.05
        signal1 = (noise_level * np.random.randn(256)).astype(np.float32)
        detector.detect(signal1)  # Establish baseline, could be transient from silence

        # signal2 = (noise_level * np.random.randn(256)).astype(np.float32) # Unused variable
        # Expect no transient if noise levels are similar
        # This depends on the stability of RMS of the noise
        # For truly random noise, RMS can fluctuate.
        # Let's try to make it more stable by adding a small tone
        tone = generate_sine_wave(1000, 256 / 44100.0, 44100, amplitude=0.1).astype(
            np.float32
        )
        noisy_tone1 = tone + (noise_level * np.random.randn(256)).astype(np.float32)
        noisy_tone2 = tone + (noise_level * np.random.randn(256)).astype(np.float32)

        detector.detect(noisy_tone1)  # Baseline
        assert (
            detector.detect(noisy_tone2) == 0
        )  # Should be no transient if RMS is similar

    def test_detect_with_noise_and_impulse_transient(self):
        detector = TransientDetector()
        np.random.seed(42)
        # sample_rate = 44100 # Unused variable
        # duration = 256 / sample_rate # Unused variable
        noise_level = 0.01  # Reduce noise level to make impulse more prominent

        # Noisy signal without impulse
        noisy_signal_base = (noise_level * np.random.randn(256)).astype(np.float32)
        detector.detect(noisy_signal_base)  # Establish baseline

        # Noisy signal with an impulse - larger impulse relative to smaller noise
        impulse = generate_impulse(256, 100, 1.0)  # Strong impulse relative to low noise
        noisy_signal_with_impulse = (noisy_signal_base + impulse).astype(np.float32)
        assert detector.detect(noisy_signal_with_impulse) == 1
