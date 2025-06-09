"""
Implements bitstream handling for ATRAC1 compressed frames.
This includes reading and writing frame data like control info,
word lengths, scale factors, and quantized spectral coefficients.
"""

from typing import List, TYPE_CHECKING, Optional
from .scaling_quantization import BitstreamSignedValues
from ..common import constants

if TYPE_CHECKING:
    from .codec_data import Atrac1CodecData


class Atrac1FrameData:
    """Holds the unpacked data of a single ATRAC1 frame."""

    def __init__(self):
        self.bsm_low: int = 0
        self.bsm_mid: int = 0
        self.bsm_high: int = 0
        self.bfu_amount_idx: int = 0
        self.num_active_bfus: int = 0

        self.word_lengths: List[int] = []
        self.scale_factor_indices: List[int] = []
        self.quantized_mantissas: List[List[int]] = []


class TBitStream:
    """
    Handles packing and unpacking of ATRAC1 compressed frames (212 bytes).
    """

    def __init__(self, stream_bytes: Optional[bytes] = None):
        """
        Initializes the bitstream.

        Args:
            stream_bytes: Bytes to initialize the stream for reading.
                          If None, initializes an empty stream for writing.
        """
        self.buffer: bytearray = (
            bytearray(stream_bytes) if stream_bytes is not None else bytearray()
        )
        self.bit_position: int = 0
        self.byte_position: int = 0

    def _ensure_buffer_write(self, num_bits_to_write: int):
        """Ensures buffer has enough space for writing num_bits_to_write."""
        num_bytes_needed = (
            self.byte_position * 8 + self.bit_position + num_bits_to_write + 7
        ) // 8
        if num_bytes_needed > len(self.buffer):
            self.buffer.extend([0] * (num_bytes_needed - len(self.buffer)))

    def _ensure_buffer_read(self, num_bits_to_read: int) -> bool:
        """Checks if buffer has enough data for reading num_bits_to_read."""
        bits_available = (len(self.buffer) - self.byte_position) * 8 - self.bit_position
        return bits_available >= num_bits_to_read

    def write_bits(self, value: int, num_bits: int):
        """
        Writes 'num_bits' from 'value' to the bitstream.
        Bits are written from MSB of value to LSB.
        Stream is filled MSB first for each byte.
        """
        if num_bits < 0 or num_bits > 32:  # Max 32 bits for typical int
            raise ValueError("Number of bits must be between 0 and 32")
        if num_bits == 0:
            return

        self._ensure_buffer_write(num_bits)

        for i in range(num_bits - 1, -1, -1):  # Iterate from MSB of value to LSB
            bit = (value >> i) & 1
            if bit:
                self.buffer[self.byte_position] |= 1 << (7 - self.bit_position)

            self.bit_position += 1
            if self.bit_position == 8:
                self.bit_position = 0
                self.byte_position += 1
                if self.byte_position == len(self.buffer) and i > 0:
                    self.buffer.append(0)

    def read_bits(self, num_bits: int) -> int:
        """
        Reads 'num_bits' from the bitstream.
        Bits are read MSB first from the stream.
        """
        if num_bits < 0 or num_bits > 32:
            raise ValueError("Number of bits must be between 0 and 32")
        if num_bits == 0:
            return 0

        if not self._ensure_buffer_read(num_bits):
            raise EOFError("Not enough bits in stream to read")

        value = 0
        for _ in range(num_bits):
            bit = (self.buffer[self.byte_position] >> (7 - self.bit_position)) & 1
            value = (value << 1) | bit

            self.bit_position += 1
            if self.bit_position == 8:
                self.bit_position = 0
                self.byte_position += 1
        return value

    def get_bytes(self) -> bytes:
        """Returns the current buffer content as bytes."""
        return bytes(self.buffer)

    def pad_to_byte_boundary(self):
        """Pads with zero bits until the next byte boundary if not already aligned."""
        if self.bit_position != 0:
            bits_to_pad = 8 - self.bit_position
            self.write_bits(0, bits_to_pad)

    def pad_to_size(self, target_size_bytes: int):
        """Pads the stream with zero bytes until it reaches target_size_bytes."""
        self.pad_to_byte_boundary()  # Align to byte first
        while len(self.buffer) < target_size_bytes:
            self.buffer.append(0)
        if len(self.buffer) > target_size_bytes:
            self.buffer = self.buffer[:target_size_bytes]


class Atrac1BitstreamWriter:
    """Writes ATRAC1 frame data to a bitstream."""

    def __init__(self, codec_data: "Atrac1CodecData"):
        self.codec_data: "Atrac1CodecData" = codec_data  # Needed for specs_per_block

    def write_frame(self, frame_data: Atrac1FrameData) -> bytes:
        """
        Packs an Atrac1FrameData object into a 212-byte ATRAC1 frame.
        
        ATRAC1 Format:
        1. Bits 0-7: BSM data for TBlockSizeMod::Parse (2-bsm_low, 2-bsm_mid, 3-bsm_high, reserved)
        2. Bits 8-10: BFU amount index  
        3. Bits 11-15: Reserved
        4. Bits 16+: Word lengths, scale factors, mantissas
        """
        stream = TBitStream()

        # Write BSM values for TBlockSizeMod::Parse (bits 0-7)
        # Parse does: actual = 2-raw (for low/mid), actual = 3-raw (for high)
        # So to encode our BSM values, we write: 2-bsm, 2-bsm, 3-bsm
        bsm_low_encoded = 2 - frame_data.bsm_low
        bsm_mid_encoded = 2 - frame_data.bsm_mid  
        bsm_high_encoded = 3 - frame_data.bsm_high
        
        stream.write_bits(bsm_low_encoded, 2)   # Bits 0-1
        stream.write_bits(bsm_mid_encoded, 2)   # Bits 2-3
        stream.write_bits(bsm_high_encoded, 2)  # Bits 4-5
        stream.write_bits(0, 2)                 # Bits 6-7 (reserved)
        
        # Write BFU amount index for dequantizer (bits 8-10)
        stream.write_bits(frame_data.bfu_amount_idx, 3)
        
        # Write reserved bits (bits 11-15)
        stream.write_bits(0, 2)  # Bits 11-12
        stream.write_bits(0, 3)  # Bits 13-15

        num_active_bfus = frame_data.num_active_bfus

        # Write word lengths (transformed for atracdenc compatibility)
        for i in range(num_active_bfus):
            # Store word_length - 1 in bitstream (atracdenc format)
            stored_word_length = frame_data.word_lengths[i] - 1 if frame_data.word_lengths[i] > 0 else 0
            stream.write_bits(stored_word_length, constants.BITS_PER_IDWL)

        # Write scale factor indices
        for i in range(num_active_bfus):
            stream.write_bits(
                frame_data.scale_factor_indices[i], constants.BITS_PER_IDSF
            )

        # Write quantized spectral coefficients (mantissas)
        for i in range(num_active_bfus):
            word_len = frame_data.word_lengths[i]
            if word_len > 0:
                if (
                    i >= len(self.codec_data.specs_per_block)
                    or i
                    >= len(
                        frame_data.quantized_mantissas
                    )  # Ensures frame_data.quantized_mantissas[i] is safe to access
                ):
                    raise ValueError(
                        f"Invalid BFU index {i} for accessing mantissa data or specs_per_block."
                    )

                num_coeffs_in_bfu = self.codec_data.specs_per_block[i]
                if len(frame_data.quantized_mantissas[i]) != num_coeffs_in_bfu:
                    raise ValueError(
                        f"Mismatch in mantissa count for BFU {i}. "
                        f"Expected {num_coeffs_in_bfu}, "
                        f"got {len(frame_data.quantized_mantissas[i])}"
                    )

                for mantissa in frame_data.quantized_mantissas[i]:
                    encoded_mantissa = BitstreamSignedValues.encode_signed(
                        mantissa, word_len
                    )
                    stream.write_bits(encoded_mantissa, word_len)

        stream.pad_to_size(constants.SOUND_UNIT_SIZE)
        return stream.get_bytes()


class Atrac1BitstreamReader:
    """Reads ATRAC1 frame data from a bitstream."""

    def __init__(self, codec_data: "Atrac1CodecData"):
        self.codec_data: "Atrac1CodecData" = (
            codec_data  # Needed for specs_per_block and BFU_AMOUNT_TAB
        )

    def read_frame(self, frame_bytes: bytes) -> Atrac1FrameData:
        """
        Unpacks a 212-byte ATRAC1 frame into an Atrac1FrameData object.
        """
        if len(frame_bytes) != constants.SOUND_UNIT_SIZE:
            raise ValueError(
                f"Frame bytes length must be {constants.SOUND_UNIT_SIZE}, "
                f"got {len(frame_bytes)}"
            )

        stream = TBitStream(frame_bytes)
        frame_data = Atrac1FrameData()

        # Read BSM values from first 8 bits (with atracdenc transformation)
        bsm_low_raw = stream.read_bits(2)
        bsm_mid_raw = stream.read_bits(2)
        bsm_high_raw = stream.read_bits(2)
        stream.read_bits(2)  # Reserved bits
        
        # Apply atracdenc BSM transformation: actual = (2 or 3) - raw
        frame_data.bsm_low = 2 - bsm_low_raw
        frame_data.bsm_mid = 2 - bsm_mid_raw  
        frame_data.bsm_high = 3 - bsm_high_raw
        
        # Read BFU amount from bits 8-10
        frame_data.bfu_amount_idx = stream.read_bits(3)
        
        # Read reserved bits 11-15
        stream.read_bits(2)  # Bits 11-12
        stream.read_bits(3)  # Bits 13-15

        # Determine num_active_bfus from bfu_amount_idx
        # This requires BFU_AMOUNT_TAB from spectral_mapping.py, assumed to be in codec_data
        if not hasattr(self.codec_data, "bfu_amount_tab"):
            raise AttributeError("CodecData missing 'bfu_amount_tab'")
        if frame_data.bfu_amount_idx >= len(self.codec_data.bfu_amount_tab):
            raise ValueError(f"Invalid bfu_amount_idx: {frame_data.bfu_amount_idx}")
        frame_data.num_active_bfus = self.codec_data.bfu_amount_tab[
            frame_data.bfu_amount_idx
        ]
        num_active_bfus = frame_data.num_active_bfus

        # Read word lengths and apply atracdenc transformation
        stored_word_lengths = [
            stream.read_bits(constants.BITS_PER_IDWL) for _ in range(num_active_bfus)
        ]
        # atracdenc transformation: actual_word_length = !!stored + stored
        frame_data.word_lengths = [
            (1 if stored > 0 else 0) + stored for stored in stored_word_lengths
        ]

        # Read scale factor indices
        frame_data.scale_factor_indices = [
            stream.read_bits(constants.BITS_PER_IDSF) for _ in range(num_active_bfus)
        ]

        # Read quantized spectral coefficients (mantissas)
        frame_data.quantized_mantissas = []
        for i in range(num_active_bfus):
            # Use stored word length for reading bits (what's actually in the bitstream)
            stored_word_len = stored_word_lengths[i]
            actual_word_len = frame_data.word_lengths[i]  # For decoder logic
            
            if i >= len(self.codec_data.specs_per_block):
                raise ValueError(f"Invalid BFU index {i} for specs_per_block")
            num_coeffs_in_bfu = self.codec_data.specs_per_block[i]
            
            bfu_mantissas: List[int] = []
            if stored_word_len > 0:
                # Read mantissa values using the actual word length from bitstream
                for _ in range(num_coeffs_in_bfu):
                    encoded_mantissa = stream.read_bits(actual_word_len)
                    mantissa = BitstreamSignedValues.decode_signed(
                        encoded_mantissa, actual_word_len
                    )
                    bfu_mantissas.append(mantissa)
            else:
                # For inactive BFUs, pad with zeros to maintain proper spectrum size
                bfu_mantissas = [0] * num_coeffs_in_bfu
            frame_data.quantized_mantissas.append(bfu_mantissas)

        return frame_data
