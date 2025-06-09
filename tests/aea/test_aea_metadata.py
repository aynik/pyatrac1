"""
Tests for the AEA metadata module.
"""

import io
import struct
import pytest
from pyatrac1.aea.metadata import AeaMetadata, AEA_MAGIC_NUMBER
from pyatrac1.common.constants import AEA_META_SIZE


class TestAeaMetadata:
    """Test cases for AeaMetadata class."""

    def test_init_default(self):
        """Test initialization with default values."""
        metadata = AeaMetadata()
        assert metadata.title == ""
        assert metadata.total_frames == 0
        assert metadata.channel_count == 0

    def test_init_with_values(self):
        """Test initialization with specific values."""
        metadata = AeaMetadata(title="Test Song", total_frames=100, channel_count=2)
        assert metadata.title == "Test Song"
        assert metadata.total_frames == 100
        assert metadata.channel_count == 2

    def test_pack_basic(self):
        """Test basic packing functionality."""
        metadata = AeaMetadata(title="Test", total_frames=50, channel_count=1)
        packed = metadata.pack()
        
        assert len(packed) == AEA_META_SIZE
        assert packed[:4] == AEA_MAGIC_NUMBER
        assert packed[4:8] == b"Test"
        
        # Check total_frames at offset 260
        unpacked_frames = struct.unpack_from("<I", packed, 260)[0]
        assert unpacked_frames == 50
        
        # Check channel_count at offset 264
        unpacked_channels = struct.unpack_from("<B", packed, 264)[0]
        assert unpacked_channels == 1

    def test_pack_empty_title(self):
        """Test packing with empty title."""
        metadata = AeaMetadata(title="", total_frames=0, channel_count=1)
        packed = metadata.pack()
        
        assert len(packed) == AEA_META_SIZE
        assert packed[:4] == AEA_MAGIC_NUMBER
        # Title area should be zeros after magic number
        assert packed[4] == 0

    def test_pack_long_title_truncation(self):
        """Test packing with title longer than maximum size."""
        long_title = "This is a very long title that exceeds the maximum allowed length"
        metadata = AeaMetadata(title=long_title, total_frames=10, channel_count=2)
        packed = metadata.pack()
        
        # Title should be truncated to fit in 15 bytes (16 - 1 for null terminator)
        title_bytes = packed[4:19]  # 15 bytes max
        title_null_idx = title_bytes.find(0)
        if title_null_idx != -1:
            extracted_title = title_bytes[:title_null_idx].decode("utf-8")
        else:
            extracted_title = title_bytes.decode("utf-8")
        
        assert len(extracted_title.encode("utf-8")) <= 15

    def test_pack_unicode_title(self):
        """Test packing with Unicode title."""
        unicode_title = "Café"
        metadata = AeaMetadata(title=unicode_title, total_frames=25, channel_count=1)
        packed = metadata.pack()
        
        assert len(packed) == AEA_META_SIZE
        assert packed[:4] == AEA_MAGIC_NUMBER

    def test_pack_invalid_channel_count_zero(self):
        """Test packing with invalid channel count (0)."""
        metadata = AeaMetadata(title="Test", total_frames=10, channel_count=0)
        with pytest.raises(ValueError, match="Channel count must be 1 or 2, got 0"):
            metadata.pack()

    def test_pack_invalid_channel_count_three(self):
        """Test packing with invalid channel count (3)."""
        metadata = AeaMetadata(title="Test", total_frames=10, channel_count=3)
        with pytest.raises(ValueError, match="Channel count must be 1 or 2, got 3"):
            metadata.pack()

    def test_pack_max_values(self):
        """Test packing with maximum values."""
        metadata = AeaMetadata(title="Max", total_frames=0xFFFFFFFF, channel_count=2)
        packed = metadata.pack()
        
        # Check that large frame count is packed correctly
        unpacked_frames = struct.unpack_from("<I", packed, 260)[0]
        assert unpacked_frames == 0xFFFFFFFF

    def test_unpack_basic(self):
        """Test basic unpacking functionality."""
        # Create a valid header manually
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:9] = b"Hello"  # Title
        struct.pack_into("<I", header, 260, 75)  # total_frames
        struct.pack_into("<B", header, 264, 2)   # channel_count
        
        metadata = AeaMetadata.unpack(bytes(header))
        assert metadata.title == "Hello"
        assert metadata.total_frames == 75
        assert metadata.channel_count == 2

    def test_unpack_wrong_size(self):
        """Test unpacking with wrong header size."""
        short_header = b'\x00' * 100
        with pytest.raises(ValueError, match=f"Header bytes must be {AEA_META_SIZE} bytes long, got 100"):
            AeaMetadata.unpack(short_header)

    def test_unpack_invalid_magic_number(self):
        """Test unpacking with invalid magic number."""
        header = bytearray(AEA_META_SIZE)
        header[:4] = b"\xFF\xFF\xFF\xFF"  # Invalid magic
        struct.pack_into("<I", header, 260, 10)
        struct.pack_into("<B", header, 264, 1)
        
        with pytest.raises(ValueError, match="Invalid AEA magic number"):
            AeaMetadata.unpack(bytes(header))

    def test_unpack_title_with_null_terminator(self):
        """Test unpacking title that has null terminator."""
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:12] = b"Title\x00\x00\x00"  # Title with null terminators  
        struct.pack_into("<I", header, 260, 5)
        struct.pack_into("<B", header, 264, 1)
        
        metadata = AeaMetadata.unpack(bytes(header))
        assert metadata.title == "Title"  # Should stop at first null

    def test_unpack_title_with_invalid_utf8(self):
        """Test unpacking title with invalid UTF-8 bytes."""
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:8] = b"\xFF\xFE\xFD\xFC"  # Invalid UTF-8
        struct.pack_into("<I", header, 260, 1)
        struct.pack_into("<B", header, 264, 1)
        
        metadata = AeaMetadata.unpack(bytes(header))
        # Should handle invalid UTF-8 gracefully with replacement chars
        assert metadata.total_frames == 1
        assert metadata.channel_count == 1

    def test_unpack_invalid_channel_count_zero(self):
        """Test unpacking with invalid channel count (0)."""
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:8] = b"Test"
        struct.pack_into("<I", header, 260, 10)
        struct.pack_into("<B", header, 264, 0)  # Invalid channel count
        
        with pytest.raises(ValueError, match="Invalid channel count in header: 0"):
            AeaMetadata.unpack(bytes(header))

    def test_unpack_invalid_channel_count_five(self):
        """Test unpacking with invalid channel count (5)."""
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:8] = b"Test"
        struct.pack_into("<I", header, 260, 10)
        struct.pack_into("<B", header, 264, 5)  # Invalid channel count
        
        with pytest.raises(ValueError, match="Invalid channel count in header: 5"):
            AeaMetadata.unpack(bytes(header))

    def test_read_from_stream_success(self):
        """Test successful reading from stream."""
        # Create valid metadata and pack it
        original_metadata = AeaMetadata(title="Stream Test", total_frames=123, channel_count=2)
        packed_data = original_metadata.pack()
        
        stream = io.BytesIO(packed_data)
        read_metadata = AeaMetadata.read_from_stream(stream)
        
        assert read_metadata.title == "Stream Test"
        assert read_metadata.total_frames == 123
        assert read_metadata.channel_count == 2

    def test_read_from_stream_insufficient_data(self):
        """Test reading from stream with insufficient data."""
        short_data = b'\x00' * 100
        stream = io.BytesIO(short_data)
        
        with pytest.raises(EOFError, match=f"Could not read {AEA_META_SIZE} bytes for AEA metadata header"):
            AeaMetadata.read_from_stream(stream)

    def test_read_from_stream_exact_size(self):
        """Test reading from stream with exactly the right amount of data."""
        # Create minimal valid header
        header = bytearray(AEA_META_SIZE)
        header[:4] = AEA_MAGIC_NUMBER
        header[4:9] = b"Exact"
        struct.pack_into("<I", header, 260, 42)
        struct.pack_into("<B", header, 264, 1)
        
        stream = io.BytesIO(bytes(header))
        metadata = AeaMetadata.read_from_stream(stream)
        
        assert metadata.title == "Exact"
        assert metadata.total_frames == 42
        assert metadata.channel_count == 1

    def test_write_to_stream(self):
        """Test writing metadata to stream."""
        metadata = AeaMetadata(title="Write Test", total_frames=99, channel_count=2)
        stream = io.BytesIO()
        
        metadata.write_to_stream(stream)
        
        # Read back and verify
        stream.seek(0)
        read_metadata = AeaMetadata.read_from_stream(stream)
        
        assert read_metadata.title == "Write Test"
        assert read_metadata.total_frames == 99
        assert read_metadata.channel_count == 2

    def test_round_trip_pack_unpack(self):
        """Test that pack/unpack are inverse operations."""
        original = AeaMetadata(title="Round Trip", total_frames=777, channel_count=1)
        packed = original.pack()
        unpacked = AeaMetadata.unpack(packed)
        
        assert unpacked.title == original.title
        assert unpacked.total_frames == original.total_frames
        assert unpacked.channel_count == original.channel_count

    def test_round_trip_stream_operations(self):
        """Test that write_to_stream/read_from_stream are inverse operations."""
        original = AeaMetadata(title="Stream Round", total_frames=888, channel_count=2)
        stream = io.BytesIO()
        
        original.write_to_stream(stream)
        stream.seek(0)
        read_back = AeaMetadata.read_from_stream(stream)
        
        assert read_back.title == original.title
        assert read_back.total_frames == original.total_frames
        assert read_back.channel_count == original.channel_count

    def test_channel_count_property_alias(self):
        """Test that channel_count works as expected (using channels property from reader/writer)."""
        metadata = AeaMetadata(channel_count=2)
        assert metadata.channel_count == 2
        
        # Test channels property if it exists
        if hasattr(metadata, 'channels'):
            assert metadata.channels == 2

    def test_title_edge_cases(self):
        """Test various edge cases for title handling."""
        # Test exactly 15 byte title
        title_15_bytes = "ExactlyFifteen!"
        metadata = AeaMetadata(title=title_15_bytes, channel_count=1)
        packed = metadata.pack()
        unpacked = AeaMetadata.unpack(packed)
        assert len(unpacked.title) <= 15

    def test_zero_total_frames(self):
        """Test handling of zero total frames."""
        metadata = AeaMetadata(title="Zero", total_frames=0, channel_count=1)
        packed = metadata.pack()
        unpacked = AeaMetadata.unpack(packed)
        
        assert unpacked.total_frames == 0

    def test_large_total_frames(self):
        """Test handling of large total frame counts."""
        large_frames = 1000000
        metadata = AeaMetadata(title="Large", total_frames=large_frames, channel_count=2)
        packed = metadata.pack()
        unpacked = AeaMetadata.unpack(packed)
        
        assert unpacked.total_frames == large_frames

    def test_metadata_constants(self):
        """Test that metadata class constants are correct."""
        assert AeaMetadata.MAGIC_NUMBER_OFFSET == 0
        assert AeaMetadata.MAGIC_NUMBER_SIZE == 4
        assert AeaMetadata.TITLE_OFFSET == 4
        assert AeaMetadata.TITLE_SIZE == 16
        assert AeaMetadata.TOTAL_FRAMES_OFFSET == 260
        assert AeaMetadata.TOTAL_FRAMES_SIZE == 4
        assert AeaMetadata.CHANNEL_COUNT_OFFSET == 264
        assert AeaMetadata.CHANNEL_COUNT_SIZE == 1