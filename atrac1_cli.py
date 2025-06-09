import argparse
import wave
import numpy as np
import os

from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.decoder import Atrac1Decoder
from pyatrac1.aea.aea_reader import AeaReader, AeaReaderError
from pyatrac1.aea.aea_writer import AeaWriter, AeaWriterError
from pyatrac1.common.constants import SOUND_UNIT_SIZE  # Corrected constant name
from pyatrac1.common.debug_logger import enable_debug_logging, disable_debug_logging


def main():
    parser = argparse.ArgumentParser(description="ATRAC1 Audio Codec CLI Tool")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input audio file (.wav for encode; .atrac1 or .aea for decode)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output audio file (.atrac1 or .aea for encode; .wav for decode)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title for .aea file metadata (max 15 chars, used in encode mode if output is .aea)",
    )
    parser.add_argument(
        "--raw-channels",
        type=int,
        choices=[1, 2],
        help="Number of channels for raw .atrac1 decoding (required if input is .atrac1 and not .aea)",
    )
    parser.add_argument(
        "--raw-samplerate",
        type=int,
        default=44100,
        help="Sample rate for decoding (used for .aea and raw .atrac1, default: 44100 Hz)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["encode", "decode"],
        required=True,
        help="Operation mode: 'encode' to compress WAV to ATRAC1, 'decode' to decompress ATRAC1 to WAV",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        help="Enable debug logging to specified file (e.g., --debug-log pytrac_debug.log)",
    )

    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug_log:
        enable_debug_logging(args.debug_log)
        print(f"Debug logging enabled to: {args.debug_log}")

    if not os.path.exists(args.input):
        return

    if args.mode == "encode":
        try:
            with wave.open(args.input, "rb") as wav_in:
                n_channels = wav_in.getnchannels()
                samp_width = wav_in.getsampwidth()
                frame_rate = wav_in.getframerate()
                n_frames = wav_in.getnframes()

                if n_channels not in [1, 2]:
                    print(
                        f"Error: Input WAV has {n_channels} channels. ATRAC1 supports 1 or 2 channels."
                    )
                    return

                audio_bytes = wav_in.readframes(n_frames)
                audio_float_interleaved = np.array([], dtype=np.float32)

                if samp_width == 2:  # 16-bit signed PCM
                    audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float_interleaved = (
                        audio_array_int16.astype(np.float32) / 32768.0
                    )
                elif samp_width == 1:  # 8-bit unsigned PCM
                    audio_array_uint8 = np.frombuffer(audio_bytes, dtype=np.uint8)
                    audio_float_interleaved = (
                        audio_array_uint8.astype(np.float32) - 128.0
                    ) / 128.0
                else:
                    print(
                        f"Error: Input WAV has sample width {samp_width} bytes. "
                        f"This tool currently requires 16-bit or 8-bit WAV files for encoding."
                    )
                    print("Please convert your audio to 16-bit PCM WAV format.")
                    return

                if audio_float_interleaved.size == 0 and n_frames > 0:
                    return  # Failed to convert audio data

                NUM_SAMPLES_PER_CHANNEL_FRAME = 512

                # De-interleave audio data
                channels_raw_data = []
                if n_channels == 1:
                    channels_raw_data.append(audio_float_interleaved)
                elif n_channels == 2:
                    if audio_float_interleaved.size > 0:
                        channels_raw_data.append(audio_float_interleaved[0::2])
                        channels_raw_data.append(audio_float_interleaved[1::2])
                    else:  # Handle empty stereo input
                        channels_raw_data.append(np.array([], dtype=np.float32))
                        channels_raw_data.append(np.array([], dtype=np.float32))

                # Pad each channel and determine number of encoder frames
                padded_channels_for_encoder = []
                num_encoder_frames = 0

                if (
                    not channels_raw_data
                    or channels_raw_data[0].size == 0
                    and n_frames > 0
                ):
                    # If n_frames is 0, channels_raw_data[0].size will be 0, this is fine.
                    # This condition catches if conversion failed for non-empty input.
                    if n_frames > 0:
                        print("Error: No audio data in channels after de-interleaving.")
                        return
                    # If n_frames is 0, we'll create one frame of silence
                    num_encoder_frames = (
                        1  # Encode one frame of silence for empty input
                    )
                    for _ in range(n_channels):
                        padded_channels_for_encoder.append(
                            np.zeros(NUM_SAMPLES_PER_CHANNEL_FRAME, dtype=np.float32)
                        )
                else:
                    # Calculate padding based on the first channel
                    first_channel_len = len(channels_raw_data[0])
                    padding_this_channel = 0
                    if first_channel_len % NUM_SAMPLES_PER_CHANNEL_FRAME != 0:
                        padding_this_channel = NUM_SAMPLES_PER_CHANNEL_FRAME - (
                            first_channel_len % NUM_SAMPLES_PER_CHANNEL_FRAME
                        )

                    num_encoder_frames = (
                        first_channel_len + padding_this_channel
                    ) // NUM_SAMPLES_PER_CHANNEL_FRAME

                    for ch_samples in channels_raw_data:
                        current_ch_len = len(ch_samples)
                        current_padding_needed = 0
                        if current_ch_len % NUM_SAMPLES_PER_CHANNEL_FRAME != 0:
                            current_padding_needed = NUM_SAMPLES_PER_CHANNEL_FRAME - (
                                current_ch_len % NUM_SAMPLES_PER_CHANNEL_FRAME
                            )

                        padded_ch = np.pad(
                            ch_samples, (0, current_padding_needed), "constant"
                        )
                        if (
                            len(padded_ch)
                            != num_encoder_frames * NUM_SAMPLES_PER_CHANNEL_FRAME
                        ):
                            print(
                                f"Error: Channel padding resulted in unexpected length. Expected {num_encoder_frames * NUM_SAMPLES_PER_CHANNEL_FRAME}, got {len(padded_ch)}"
                            )
                            return
                        padded_channels_for_encoder.append(padded_ch)

                encoder = Atrac1Encoder()

                print(
                    f"Input WAV: {n_channels} channels, {samp_width * 8}-bit, {frame_rate} Hz, {n_frames} frames ({n_frames / frame_rate:.2f}s). Processing {num_encoder_frames} ATRAC frames."
                )
                print(f"Outputting to: {args.output}")

                is_aea_output = args.output.lower().endswith(".aea")

                if is_aea_output:
                    with AeaWriter(
                        args.output, channel_count=n_channels, title=args.title
                    ) as aea_writer:
                        for frame_idx in range(num_encoder_frames):
                            # Extract samples for this frame
                            if n_channels == 1:
                                start = frame_idx * NUM_SAMPLES_PER_CHANNEL_FRAME
                                end = start + NUM_SAMPLES_PER_CHANNEL_FRAME
                                frame_samples = padded_channels_for_encoder[0][start:end]
                            else:  # n_channels == 2
                                start = frame_idx * NUM_SAMPLES_PER_CHANNEL_FRAME
                                end = start + NUM_SAMPLES_PER_CHANNEL_FRAME
                                left_samples = padded_channels_for_encoder[0][start:end]
                                right_samples = padded_channels_for_encoder[1][start:end]
                                frame_samples = np.column_stack((left_samples, right_samples))

                            print(
                                f"Processing source frame {frame_idx + 1}/{num_encoder_frames} (Encode to AEA)..."
                            )
                            
                            try:
                                compressed_data = encoder.encode_frame(frame_samples)
                            except Exception as e:
                                print(f"An unexpected error occurred during encoding: {e}")
                                compressed_data = None
                            
                            if compressed_data:
                                if n_channels == 2:
                                    if (
                                        len(compressed_data)
                                        == 2 * SOUND_UNIT_SIZE  # Corrected constant
                                    ):
                                        aea_writer.write_frame(
                                            compressed_data[
                                                :SOUND_UNIT_SIZE  # Corrected constant
                                            ]
                                        )
                                        aea_writer.write_frame(
                                            compressed_data[
                                                SOUND_UNIT_SIZE:  # Corrected constant
                                            ]
                                        )
                                    else:
                                        if n_channels == 2:
                                            aea_writer.write_frame(
                                                compressed_data[:SOUND_UNIT_SIZE]
                                            )
                                            aea_writer.write_frame(
                                                compressed_data[SOUND_UNIT_SIZE:]
                                            )
                                        else:  # Mono
                                            aea_writer.write_frame(compressed_data)
                                else:
                                    # Write dummy frames if encoder fails
                                    if n_channels == 2:
                                        aea_writer.write_frame(bytes(SOUND_UNIT_SIZE))
                                        aea_writer.write_frame(bytes(SOUND_UNIT_SIZE))
                                    else:
                                        aea_writer.write_frame(bytes(SOUND_UNIT_SIZE))
                else:  # Raw .atrac1 output
                    encoded_data_chunks = []
                    for frame_idx in range(num_encoder_frames):
                        samples_for_this_frame_to_encode = []
                        for ch_data in padded_channels_for_encoder:
                            start = frame_idx * NUM_SAMPLES_PER_CHANNEL_FRAME
                            end = start + NUM_SAMPLES_PER_CHANNEL_FRAME
                            samples_for_this_frame_to_encode.append(ch_data[start:end])

                        print(
                            f"Processing source frame {frame_idx + 1}/{num_encoder_frames} (Encode to raw ATRAC1)..."
                        )
                        compressed_frame = encoder.encode_frame(
                            samples_for_this_frame_to_encode
                        )
                        if compressed_frame:
                            encoded_data_chunks.append(compressed_frame)
                        else:
                            print(
                                f"Warning: Encoder returned no data for frame {frame_idx + 1}"
                            )
                            # For raw, we might append empty bytes or a placeholder of correct size if stereo
                            if n_channels == 2:
                                encoded_data_chunks.append(
                                    bytes(2 * SOUND_UNIT_SIZE)  # Corrected constant
                                )
                            else:
                                encoded_data_chunks.append(
                                    bytes(SOUND_UNIT_SIZE)  # Corrected constant
                                )

                    with open(args.output, "wb") as f_out:
                        for chunk in encoded_data_chunks:
                            f_out.write(chunk)

            pass

        except wave.Error as e:
            print(f"Error reading WAV file: {e}")
        except AeaWriterError as e:
            print(f"Error writing AEA file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during encoding: {e}")

    elif args.mode == "decode":
        print(f"Decoding '{args.input}' to '{args.output}'...")

        is_aea_input = args.input.lower().endswith(".aea")
        n_channels_decoded = 0
        frame_rate_decoded = (
            args.raw_samplerate
        )  # Used for all decoding. Defaults to 44100 Hz.

        temp_decoded_mono_frames = []  # List of 512-sample numpy arrays

        try:
            if is_aea_input:
                with AeaReader(args.input) as aea_reader:
                    metadata = aea_reader.get_metadata()
                    n_channels_decoded = metadata.channel_count
                    # AEA metadata does not store sample rate; it's taken from --raw-samplerate argument or its default.
                    print(
                        f"AEA Input: '{args.input}', Title: '{metadata.title}', Channels: {n_channels_decoded}, Sample Rate: {frame_rate_decoded} Hz (from --raw-samplerate or default), Total Frames (per ch): {metadata.total_frames}"
                    )

                    decoder = Atrac1Decoder()

                    sound_units_processed = 0
                    for compressed_sound_unit in aea_reader.frames():
                        decoded_samples = decoder.decode_frame(compressed_sound_unit)
                        if decoded_samples is not None and decoded_samples.size > 0:
                            temp_decoded_mono_frames.append(decoded_samples)
                        else:
                            temp_decoded_mono_frames.append(
                                np.zeros(512, dtype=np.float32)
                            )
                        sound_units_processed += 1

                    expected_sound_units = metadata.total_frames * n_channels_decoded
                    if sound_units_processed < expected_sound_units:
                        print(
                            f"Warning: AEA file provided {sound_units_processed} sound units, metadata expected {expected_sound_units}."
                        )
                        # Pad with silence
                        for _ in range(expected_sound_units - sound_units_processed):
                            temp_decoded_mono_frames.append(
                                np.zeros(512, dtype=np.float32)
                            )

            else:  # Raw .atrac1 input
                if not args.raw_channels:
                    print(
                        "Error: --raw-channels is required for decoding raw .atrac1 files."
                    )
                    return
                n_channels_decoded = args.raw_channels
                # frame_rate_decoded is already set from args.raw_samplerate
                print(
                    f"Raw ATRAC1 Input: '{args.input}', Channels: {n_channels_decoded}, Sample Rate: {frame_rate_decoded} Hz"
                )

                decoder = Atrac1Decoder()

                with open(args.input, "rb") as f_in:
                    compressed_data_full = f_in.read()

                num_sound_units = len(compressed_data_full) // SOUND_UNIT_SIZE
                if len(compressed_data_full) % SOUND_UNIT_SIZE != 0:
                    pass  # Silently ignore trailing data

                for i in range(num_sound_units):
                    start_idx = i * SOUND_UNIT_SIZE
                    end_idx = (i + 1) * SOUND_UNIT_SIZE
                    compressed_sound_unit = compressed_data_full[start_idx:end_idx]
                    decoded_samples = decoder.decode_frame(compressed_sound_unit)
                    if decoded_samples is not None and decoded_samples.size > 0:
                        temp_decoded_mono_frames.append(decoded_samples)
                    else:
                        temp_decoded_mono_frames.append(np.zeros(512, dtype=np.float32))

            # Combine and interleave decoded mono frames
            final_decoded_array = np.array([], dtype=np.float32)
            if not temp_decoded_mono_frames:
                print("Warning: No audio data decoded.")
            elif n_channels_decoded == 1:
                final_decoded_array = np.concatenate(temp_decoded_mono_frames)
            else:  # Stereo
                if len(temp_decoded_mono_frames) % 2 != 0:
                    print(
                        "Warning: Odd number of sound units for stereo decoding. Last unit might be ignored or cause issues."
                    )
                    # Optionally pad temp_decoded_mono_frames with a silent frame if desired
                    # temp_decoded_mono_frames.append(np.zeros(512, dtype=np.float32))

                l_frames = [
                    temp_decoded_mono_frames[i]
                    for i in range(0, len(temp_decoded_mono_frames), 2)
                ]
                r_frames = [
                    temp_decoded_mono_frames[i]
                    for i in range(1, len(temp_decoded_mono_frames), 2)
                ]

                if not l_frames and not r_frames:
                    print("Warning: No L/R frames to assemble for stereo output.")
                elif not r_frames and l_frames:  # Only L frames, treat as mono
                    print(
                        "Warning: Only L channel frames found for stereo. Outputting as mono."
                    )
                    final_decoded_array = np.concatenate(l_frames)
                    n_channels_decoded = 1  # Correct for WAV output
                elif l_frames and r_frames:
                    l_channel_data = np.concatenate(l_frames)
                    r_channel_data = np.concatenate(r_frames)

                    min_len = min(len(l_channel_data), len(r_channel_data))
                    if len(l_channel_data) != len(r_channel_data):
                        print(
                            f"Warning: L/R channels have different lengths after decoding ({len(l_channel_data)} vs {len(r_channel_data)}). Truncating to shorter."
                        )

                    final_decoded_array = np.empty(min_len * 2, dtype=np.float32)
                    final_decoded_array[0::2] = l_channel_data[:min_len]
                    final_decoded_array[1::2] = r_channel_data[:min_len]
                else:  # Should not happen if l_frames or r_frames exist
                    print(
                        "Warning: Could not assemble stereo audio from decoded frames."
                    )

            # Convert decoded float samples back to int16 for WAV writing
            final_decoded_int16 = np.clip(
                final_decoded_array * (2**15 - 1), -(2**15), (2**15 - 1)
            ).astype(np.int16)

            with wave.open(args.output, "wb") as wav_out:
                wav_out.setnchannels(n_channels_decoded)
                wav_out.setsampwidth(2)  # 16-bit PCM
                wav_out.setframerate(frame_rate_decoded)
                wav_out.writeframes(final_decoded_int16.tobytes())

            print("Decoding complete.")

        except FileNotFoundError:
            raise
        except (AeaReaderError, AeaWriterError) as e:
            raise
        except Exception as e:
            raise


if __name__ == "__main__":
    main()
