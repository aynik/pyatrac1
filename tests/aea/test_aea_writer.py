"""
Tests for the AEA writer module.
"""

import io
import pytest
from pyatrac1.aea.aea_writer import AeaWriter, AeaWriterError
from pyatrac1.aea.metadata import AeaMetadata
from pyatrac1.common.constants import AEA_META_SIZE, SOUND_UNIT_SIZE


class TestAeaWriter:
    """Test cases for AeaWriter class."""

    def test_init_with_valid_stream_mono(self):
        """Test initialization with valid stream for mono audio."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1, title="Test Song")
        
        assert writer.metadata.channels == 1
        assert writer.metadata.title == "Test Song"
        assert writer.metadata.total_frames == 0
        assert writer._frame_count == 0
        assert writer._first_write_done is True
        assert writer._close_on_exit is False
        
        # Check that initial header and 5 dummy frames were written (atracdenc compatible)
        assert stream.tell() == AEA_META_SIZE + 5 * SOUND_UNIT_SIZE

    def test_init_with_valid_stream_stereo(self):
        """Test initialization with valid stream for stereo audio."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=2, title="Stereo Test")
        
        assert writer.metadata.channels == 2
        assert writer.metadata.title == "Stereo Test"

    def test_init_with_invalid_channel_count_zero(self):
        """Test initialization with invalid channel count (0)."""
        stream = io.BytesIO()
        with pytest.raises(AeaWriterError, match="Channel count must be 1 or 2, got 0"):
            AeaWriter(stream, channel_count=0)

    def test_init_with_invalid_channel_count_three(self):
        """Test initialization with invalid channel count (3)."""
        stream = io.BytesIO()
        with pytest.raises(AeaWriterError, match="Channel count must be 1 or 2, got 3"):
            AeaWriter(stream, channel_count=3)

    def test_init_with_string_path_permission_error(self, tmp_path):
        """Test initialization with file path that cannot be opened."""
        # Create a directory where we try to write a file
        invalid_path = tmp_path / "readonly_dir" / "test.aea"
        invalid_path.parent.mkdir()
        invalid_path.parent.chmod(0o444)  # Read-only directory
        
        try:
            with pytest.raises(AeaWriterError, match="Failed to open AEA file for writing"):
                AeaWriter(str(invalid_path), channel_count=1)
        finally:
            invalid_path.parent.chmod(0o755)  # Restore permissions for cleanup

    def test_title_truncation(self):
        """Test that title is truncated to 15 characters."""
        stream = io.BytesIO()
        long_title = "This is a very long title that exceeds 15 characters"
        writer = AeaWriter(stream, channel_count=1, title=long_title)
        
        assert len(writer.metadata.title) <= 15
        assert writer.metadata.title == long_title[:15]

    def test_write_frame_success(self):
        """Test successful frame writing."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        frame_data = b'\x01' * SOUND_UNIT_SIZE
        writer.write_frame(frame_data)
        
        assert writer._frame_count == 1
        # Check that frame was written after header and 5 dummy frames
        expected_pos = AEA_META_SIZE + 5 * SOUND_UNIT_SIZE + SOUND_UNIT_SIZE
        assert stream.tell() == expected_pos

    def test_write_multiple_frames(self):
        """Test writing multiple frames."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        frame1 = b'\x01' * SOUND_UNIT_SIZE
        frame2 = b'\x02' * SOUND_UNIT_SIZE
        frame3 = b'\x03' * SOUND_UNIT_SIZE
        
        writer.write_frame(frame1)
        writer.write_frame(frame2)
        writer.write_frame(frame3)
        
        assert writer._frame_count == 3

    def test_write_frame_invalid_size_too_small(self):
        """Test writing frame with invalid size (too small)."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        invalid_frame = b'\x01' * (SOUND_UNIT_SIZE - 1)
        with pytest.raises(AeaWriterError, match=f"Frame bytes length must be {SOUND_UNIT_SIZE}"):
            writer.write_frame(invalid_frame)

    def test_write_frame_invalid_size_too_large(self):
        """Test writing frame with invalid size (too large)."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        invalid_frame = b'\x01' * (SOUND_UNIT_SIZE + 1)
        with pytest.raises(AeaWriterError, match=f"Frame bytes length must be {SOUND_UNIT_SIZE}"):
            writer.write_frame(invalid_frame)

    def test_write_frame_without_init(self):
        """Test writing frame when initial setup not done."""
        stream = io.BytesIO()
        writer = AeaWriter.__new__(AeaWriter)  # Create without calling __init__
        writer._first_write_done = False
        
        frame_data = b'\x01' * SOUND_UNIT_SIZE
        with pytest.raises(AeaWriterError, match="Initial header and dummy frame not written"):
            writer.write_frame(frame_data)

    def test_write_frame_io_error(self):
        """Test writing frame with IO error."""
        # Create a mock stream that raises IOError on write
        class FailingStream:
            def __init__(self):
                self.closed = False
                
            def write(self, data):
                raise IOError("Mock write error")
                
            def seek(self, pos):
                pass
                
            def tell(self):
                return 0
                
            def flush(self):
                pass
        
        writer = AeaWriter.__new__(AeaWriter)
        writer.stream = FailingStream()
        writer._first_write_done = True
        
        frame_data = b'\x01' * SOUND_UNIT_SIZE
        with pytest.raises(AeaWriterError, match="Failed to write audio frame"):
            writer.write_frame(frame_data)

    def test_finalize_success(self):
        """Test successful finalization."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1, title="Test")
        
        # Write some frames
        frame1 = b'\x01' * SOUND_UNIT_SIZE
        frame2 = b'\x02' * SOUND_UNIT_SIZE
        writer.write_frame(frame1)
        writer.write_frame(frame2)
        
        # Store current position
        pos_before_finalize = stream.tell()
        
        writer.finalize()
        
        # Check that total_frames was updated
        assert writer.metadata.total_frames == 2
        
        # Check that stream position was restored
        assert stream.tell() == pos_before_finalize
        
        # Verify metadata was written correctly
        stream.seek(0)
        read_metadata = AeaMetadata.read_from_stream(stream)
        assert read_metadata.total_frames == 2
        assert read_metadata.title == "Test"

    def test_finalize_io_error(self):
        """Test finalization with IO error."""
        # Create a mock stream that fails on tell()
        class FailingStream:
            def __init__(self):
                self.closed = False
                
            def tell(self):
                raise IOError("Mock tell error")
                
            def seek(self, pos):
                pass
                
            def write(self, data):
                pass
                
            def flush(self):
                pass
        
        writer = AeaWriter.__new__(AeaWriter)
        writer.stream = FailingStream()
        writer.metadata = AeaMetadata(channel_count=1)
        writer._frame_count = 1
        
        with pytest.raises(AeaWriterError, match="Failed to finalize AEA metadata header"):
            writer.finalize()

    def test_close_with_finalization(self):
        """Test closing writer with automatic finalization."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        writer._close_on_exit = True  # Simulate file-based writer
        
        frame_data = b'\x01' * SOUND_UNIT_SIZE
        writer.write_frame(frame_data)
        
        writer.close()
        
        # Check that metadata was finalized
        assert writer.metadata.total_frames == 1
        assert stream.closed

    def test_close_external_stream(self):
        """Test closing with external stream (shouldn't close stream)."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        frame_data = b'\x01' * SOUND_UNIT_SIZE
        writer.write_frame(frame_data)
        
        writer.close()
        
        # Check that metadata was finalized but stream wasn't closed
        assert writer.metadata.total_frames == 1
        assert not stream.closed

    def test_close_already_closed_stream(self):
        """Test closing when stream is already closed."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1)
        
        stream.close()
        
        # Should not raise an exception
        writer.close()

    def test_context_manager_success(self):
        """Test context manager functionality."""
        stream = io.BytesIO()
        
        with AeaWriter(stream, channel_count=2, title="Context Test") as writer:
            frame_data = b'\x01' * SOUND_UNIT_SIZE
            writer.write_frame(frame_data)
            writer.write_frame(frame_data)
        
        # Check that finalization happened
        stream.seek(0)
        metadata = AeaMetadata.read_from_stream(stream)
        assert metadata.total_frames == 2
        assert metadata.title == "Context Test"
        assert metadata.channels == 2

    def test_context_manager_with_exception(self):
        """Test context manager with exception handling."""
        stream = io.BytesIO()
        
        with pytest.raises(ValueError):
            with AeaWriter(stream, channel_count=1) as writer:
                frame_data = b'\x01' * SOUND_UNIT_SIZE
                writer.write_frame(frame_data)
                raise ValueError("Test exception")
        
        # Check that finalization still happened despite exception
        stream.seek(0)
        metadata = AeaMetadata.read_from_stream(stream)
        assert metadata.total_frames == 1

    def test_initial_header_position_validation(self):
        """Test validation of stream position after initial header write."""
        # Create a mock stream that doesn't advance position properly
        class MockStream:
            def __init__(self):
                self.data = bytearray()
                self.pos = 0
                self.closed = False
                
            def write(self, data):
                # Don't actually advance position to trigger error
                self.data.extend(data)
                
            def seek(self, pos):
                self.pos = pos
                
            def tell(self):
                return self.pos  # Return wrong position
                
            def flush(self):
                pass
                
            def close(self):
                self.closed = True
        
        mock_stream = MockStream()
        
        with pytest.raises(AeaWriterError, match="Stream position after writing initial header and 5 dummy frames is incorrect"):
            AeaWriter(mock_stream, channel_count=1)

    def test_initial_header_io_error(self):
        """Test IO error during initial header write."""
        # Create a stream that raises IOError on write
        class FailingStream:
            def write(self, data):
                raise IOError("Mock IO error")
                
            def seek(self, pos):
                pass
                
            def tell(self):
                return 0
        
        failing_stream = FailingStream()
        
        with pytest.raises(AeaWriterError, match="Failed to write initial AEA header or dummy frame"):
            AeaWriter(failing_stream, channel_count=1)

    def test_empty_title(self):
        """Test initialization with empty title."""
        stream = io.BytesIO()
        writer = AeaWriter(stream, channel_count=1, title="")
        
        assert writer.metadata.title == ""

    def test_exactly_15_char_title(self):
        """Test initialization with exactly 15 character title."""
        stream = io.BytesIO()
        title_15_chars = "Exactly15Chars!"
        writer = AeaWriter(stream, channel_count=1, title=title_15_chars)
        
        assert writer.metadata.title == title_15_chars
        assert len(writer.metadata.title) == 15