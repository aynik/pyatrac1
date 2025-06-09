"""
Tests for the ATRAC1 bitstream module.
"""

import pytest
from pyatrac1.core.bitstream import TBitStream, Atrac1FrameData, Atrac1BitstreamWriter, Atrac1BitstreamReader
from pyatrac1.core.codec_data import Atrac1CodecData
from pyatrac1.common import constants


class TestAtrac1FrameData:
    """Test cases for Atrac1FrameData class."""

    def test_init(self):
        """Test frame data initialization."""
        frame_data = Atrac1FrameData()
        
        assert frame_data.bsm_low == 0
        assert frame_data.bsm_mid == 0
        assert frame_data.bsm_high == 0
        assert frame_data.bfu_amount_idx == 0
        assert frame_data.num_active_bfus == 0
        assert frame_data.word_lengths == []
        assert frame_data.scale_factor_indices == []
        assert frame_data.quantized_mantissas == []


class TestTBitStream:
    """Test cases for TBitStream class."""

    def test_init_empty(self):
        """Test initialization with no bytes."""
        bs = TBitStream()
        assert bs.buffer == bytearray()
        assert bs.byte_position == 0
        assert bs.bit_position == 0

    def test_init_with_bytes(self):
        """Test initialization with byte data."""
        test_bytes = b'\x00\xFF\x55'
        bs = TBitStream(test_bytes)
        assert bs.buffer == bytearray(test_bytes)
        assert bs.byte_position == 0
        assert bs.bit_position == 0

    def test_write_bits_invalid_num_bits_negative(self):
        """Test writing with negative number of bits."""
        bs = TBitStream()
        with pytest.raises(ValueError, match="Number of bits must be between 0 and 32"):
            bs.write_bits(42, -1)

    def test_write_bits_invalid_num_bits_too_large(self):
        """Test writing with too many bits."""
        bs = TBitStream()
        with pytest.raises(ValueError, match="Number of bits must be between 0 and 32"):
            bs.write_bits(42, 33)

    def test_write_bits_zero_bits(self):
        """Test writing zero bits."""
        bs = TBitStream()
        bs.write_bits(42, 0)  # Should do nothing
        assert bs.buffer == bytearray()

    def test_read_bits_invalid_num_bits_negative(self):
        """Test reading with negative number of bits."""
        bs = TBitStream(b'\xFF')
        with pytest.raises(ValueError, match="Number of bits must be between 0 and 32"):
            bs.read_bits(-1)

    def test_read_bits_invalid_num_bits_too_large(self):
        """Test reading with too many bits."""
        bs = TBitStream(b'\xFF')
        with pytest.raises(ValueError, match="Number of bits must be between 0 and 32"):
            bs.read_bits(33)

    def test_read_bits_zero_bits(self):
        """Test reading zero bits."""
        bs = TBitStream(b'\xFF')
        result = bs.read_bits(0)
        assert result == 0

    def test_read_bits_eof(self):
        """Test reading when not enough bits available."""
        bs = TBitStream(b'\xFF')  # Only 8 bits available
        with pytest.raises(EOFError, match="Not enough bits in stream to read"):
            bs.read_bits(16)  # Try to read 16 bits

    def test_write_read_round_trip_single_bit(self):
        """Test writing and reading single bits."""
        bs = TBitStream()
        
        # Write some single bits
        bs.write_bits(1, 1)
        bs.write_bits(0, 1)
        bs.write_bits(1, 1)
        bs.write_bits(1, 1)
        
        # Reset position for reading
        bs.byte_position = 0
        bs.bit_position = 0
        
        # Read them back
        assert bs.read_bits(1) == 1
        assert bs.read_bits(1) == 0
        assert bs.read_bits(1) == 1
        assert bs.read_bits(1) == 1

    def test_write_read_round_trip_multiple_bits(self):
        """Test writing and reading multiple bits."""
        bs = TBitStream()
        
        # Write various values
        bs.write_bits(0b1010, 4)
        bs.write_bits(0b110011, 6)
        bs.write_bits(0b11111111, 8)
        
        # Reset position for reading
        bs.byte_position = 0
        bs.bit_position = 0
        
        # Read them back
        assert bs.read_bits(4) == 0b1010
        assert bs.read_bits(6) == 0b110011
        assert bs.read_bits(8) == 0b11111111

    def test_write_bits_buffer_expansion(self):
        """Test that buffer expands correctly during writing."""
        bs = TBitStream()
        
        # Write enough data to force buffer expansion
        for i in range(20):
            bs.write_bits(0xFF, 8)  # Write 20 bytes
        
        assert len(bs.buffer) >= 20

    def test_write_bits_cross_byte_boundary(self):
        """Test writing across byte boundaries."""
        bs = TBitStream()
        
        # Write 12 bits (crosses byte boundary)
        bs.write_bits(0b101010101010, 12)
        
        # Should result in buffer expansion at the right time
        assert len(bs.buffer) >= 2

    def test_get_bytes_empty(self):
        """Test getting bytes from empty stream."""
        bs = TBitStream()
        result = bs.get_bytes()
        assert result == b''

    def test_get_bytes_with_data(self):
        """Test getting bytes with data."""
        bs = TBitStream()
        bs.write_bits(0xFF, 8)
        bs.write_bits(0x00, 8)
        bs.write_bits(0x55, 8)
        
        result = bs.get_bytes()
        assert result == b'\xFF\x00\x55'

    def test_get_bytes_partial_byte(self):
        """Test getting bytes when last byte is partial."""
        bs = TBitStream()
        bs.write_bits(0xFF, 8)
        bs.write_bits(0b1010, 4)  # Only 4 bits in second byte
        
        result = bs.get_bytes()
        # Second byte should have 0b10100000 = 0xA0
        assert result == b'\xFF\xA0'

    def test_read_beyond_buffer(self):
        """Test reading beyond available data."""
        bs = TBitStream(b'\xFF\x00')  # 16 bits available
        
        # Read first 16 bits successfully
        assert bs.read_bits(8) == 0xFF
        assert bs.read_bits(8) == 0x00
        
        # Try to read more - should raise EOFError
        with pytest.raises(EOFError):
            bs.read_bits(1)

    def test_large_value_write_read(self):
        """Test writing and reading large values."""
        bs = TBitStream()
        
        # Write maximum 32-bit value
        large_value = 0xFFFFFFFF
        bs.write_bits(large_value, 32)
        
        # Reset and read back
        bs.byte_position = 0
        bs.bit_position = 0
        
        result = bs.read_bits(32)
        assert result == large_value

    def test_bit_position_tracking(self):
        """Test that bit positions are tracked correctly."""
        bs = TBitStream()
        
        # Write 3 bits
        bs.write_bits(0b101, 3)
        assert bs.bit_position == 3
        assert bs.byte_position == 0
        
        # Write 6 more bits (total 9, should advance to next byte)
        bs.write_bits(0b111000, 6)
        assert bs.bit_position == 1  # 9 bits total, 1 bit into second byte
        assert bs.byte_position == 1

    def test_ensure_buffer_edge_cases(self):
        """Test buffer management edge cases."""
        bs = TBitStream()
        
        # Write data that forces multiple buffer expansions
        bs.write_bits(0xAAAA, 16)
        bs.write_bits(0x5555, 16)
        bs.write_bits(0xFFFF, 16)
        
        # Verify data integrity
        bs.byte_position = 0
        bs.bit_position = 0
        
        assert bs.read_bits(16) == 0xAAAA
        assert bs.read_bits(16) == 0x5555
        assert bs.read_bits(16) == 0xFFFF


class TestAtrac1BitstreamWriter:
    """Test cases for Atrac1BitstreamWriter class."""

    def test_init(self):
        """Test writer initialization."""
        codec_data = Atrac1CodecData()
        writer = Atrac1BitstreamWriter(codec_data)
        assert writer.codec_data is codec_data

    def test_write_frame_basic(self):
        """Test basic frame writing."""
        codec_data = Atrac1CodecData()
        writer = Atrac1BitstreamWriter(codec_data)
        
        frame_data = Atrac1FrameData()
        frame_data.bsm_low = 1
        frame_data.bsm_mid = 2
        frame_data.bsm_high = 3
        frame_data.bfu_amount_idx = 4
        frame_data.num_active_bfus = codec_data.bfu_amount_tab[4]
        frame_data.word_lengths = [4] * frame_data.num_active_bfus
        frame_data.scale_factor_indices = [10] * frame_data.num_active_bfus
        
        # Create correct sized mantissas for each BFU
        frame_data.quantized_mantissas = []
        for i in range(frame_data.num_active_bfus):
            num_specs = codec_data.specs_per_block[i]
            mantissas = [j % 16 for j in range(num_specs)]  # Create test data
            frame_data.quantized_mantissas.append(mantissas)
        
        result = writer.write_frame(frame_data)
        
        # Should return exactly SOUND_UNIT_SIZE bytes
        assert isinstance(result, bytes)
        assert len(result) == constants.SOUND_UNIT_SIZE


class TestAtrac1BitstreamReader:
    """Test cases for Atrac1BitstreamReader class."""

    def test_init(self):
        """Test reader initialization."""
        codec_data = Atrac1CodecData()
        reader = Atrac1BitstreamReader(codec_data)
        assert reader.codec_data is codec_data

    def test_read_frame_invalid_size(self):
        """Test reading frame with invalid size."""
        codec_data = Atrac1CodecData()
        reader = Atrac1BitstreamReader(codec_data)
        
        # Frame too small
        small_frame = b'\x00' * 100
        with pytest.raises((ValueError, IndexError)):
            reader.read_frame(small_frame)

    def test_write_read_frame_round_trip(self):
        """Test writing and reading a frame."""
        codec_data = Atrac1CodecData()
        writer = Atrac1BitstreamWriter(codec_data)
        reader = Atrac1BitstreamReader(codec_data)
        
        # Create frame data
        frame_data = Atrac1FrameData()
        frame_data.bsm_low = 1
        frame_data.bsm_mid = 0
        frame_data.bsm_high = 2
        frame_data.bfu_amount_idx = 5
        frame_data.num_active_bfus = codec_data.bfu_amount_tab[5]
        frame_data.word_lengths = [3] * frame_data.num_active_bfus
        frame_data.scale_factor_indices = [15] * frame_data.num_active_bfus
        
        # Create correct sized mantissas for each BFU
        frame_data.quantized_mantissas = []
        for i in range(frame_data.num_active_bfus):
            num_specs = codec_data.specs_per_block[i]
            mantissas = [1, 2] * (num_specs // 2) + [1] * (num_specs % 2)  # Fill correctly
            frame_data.quantized_mantissas.append(mantissas[:num_specs])
        
        # Write frame
        frame_bytes = writer.write_frame(frame_data)
        
        # Read frame back
        read_frame_data = reader.read_frame(frame_bytes)
        
        # Verify key fields match
        assert read_frame_data.bsm_low == frame_data.bsm_low
        assert read_frame_data.bsm_mid == frame_data.bsm_mid
        assert read_frame_data.bsm_high == frame_data.bsm_high
        assert read_frame_data.bfu_amount_idx == frame_data.bfu_amount_idx

    def test_read_frame_minimal_data(self):
        """Test reading frame with minimal valid data."""
        codec_data = Atrac1CodecData()
        reader = Atrac1BitstreamReader(codec_data)
        
        # Create minimal valid frame (all zeros)
        minimal_frame = b'\x00' * constants.SOUND_UNIT_SIZE
        
        # Should be able to read without error
        frame_data = reader.read_frame(minimal_frame)
        assert isinstance(frame_data, Atrac1FrameData)

    def test_read_frame_maximum_bfu_amount(self):
        """Test reading frame with maximum BFU amount."""
        codec_data = Atrac1CodecData()
        writer = Atrac1BitstreamWriter(codec_data)
        reader = Atrac1BitstreamReader(codec_data)
        
        # Use maximum BFU amount index
        max_bfu_idx = len(codec_data.bfu_amount_tab) - 1
        
        frame_data = Atrac1FrameData()
        frame_data.bfu_amount_idx = max_bfu_idx
        frame_data.num_active_bfus = codec_data.bfu_amount_tab[max_bfu_idx]
        frame_data.word_lengths = [2] * frame_data.num_active_bfus
        frame_data.scale_factor_indices = [0] * frame_data.num_active_bfus
        
        # Create correct sized mantissas for each BFU
        frame_data.quantized_mantissas = []
        for i in range(frame_data.num_active_bfus):
            num_specs = codec_data.specs_per_block[i]
            mantissas = [0] * num_specs
            frame_data.quantized_mantissas.append(mantissas)
        
        # Write and read back
        frame_bytes = writer.write_frame(frame_data)
        read_frame_data = reader.read_frame(frame_bytes)
        
        assert read_frame_data.bfu_amount_idx == max_bfu_idx

    def test_bitstream_endianness_consistency(self):
        """Test that bitstream maintains consistent endianness."""
        bs = TBitStream()
        
        # Write a specific bit pattern
        bs.write_bits(0b10101010, 8)
        bs.write_bits(0b11110000, 8)
        
        # Convert to bytes and back
        data = bs.get_bytes()
        bs2 = TBitStream(data)
        
        # Read back should match
        assert bs2.read_bits(8) == 0b10101010
        assert bs2.read_bits(8) == 0b11110000

    def test_bitstream_partial_byte_handling(self):
        """Test handling of partial bytes in bitstream."""
        bs = TBitStream()
        
        # Write partial byte (5 bits)
        bs.write_bits(0b10110, 5)
        
        # Get bytes (should pad with zeros)
        data = bs.get_bytes()
        assert len(data) == 1
        
        # Read back
        bs2 = TBitStream(data)
        result = bs2.read_bits(5)
        assert result == 0b10110

    def test_frame_data_with_zero_word_lengths(self):
        """Test frame data with zero word lengths (which means no mantissa data)."""
        codec_data = Atrac1CodecData()
        writer = Atrac1BitstreamWriter(codec_data)
        reader = Atrac1BitstreamReader(codec_data)
        
        frame_data = Atrac1FrameData()
        frame_data.bfu_amount_idx = 2
        frame_data.num_active_bfus = codec_data.bfu_amount_tab[2]
        frame_data.word_lengths = [0] * frame_data.num_active_bfus  # Zero word length means no mantissa data
        frame_data.scale_factor_indices = [0] * frame_data.num_active_bfus
        
        # Create correct sized mantissas but they won't be written due to zero word lengths
        frame_data.quantized_mantissas = []
        for i in range(frame_data.num_active_bfus):
            num_specs = codec_data.specs_per_block[i]
            mantissas = [0] * num_specs
            frame_data.quantized_mantissas.append(mantissas)
        
        # Should handle zero word lengths gracefully
        frame_bytes = writer.write_frame(frame_data)
        read_frame_data = reader.read_frame(frame_bytes)
        
        assert len(frame_bytes) == constants.SOUND_UNIT_SIZE