"""
Tests for the AEA reader module.
"""

import io
import pytest
from pyatrac1.aea.aea_reader import AeaReader, AeaReaderError
from pyatrac1.aea.metadata import AeaMetadata
from pyatrac1.common.constants import AEA_META_SIZE, SOUND_UNIT_SIZE


class TestAeaReader:
    """Test cases for AeaReader class."""

    def test_init_with_valid_stream(self):
        """Test initialization with a valid binary stream."""
        # Create minimal valid AEA file content
        metadata = AeaMetadata()
        metadata.title = "Test Song"
        metadata.channels = 2
        metadata.total_frames = 1
        
        stream_data = bytearray(AEA_META_SIZE + 2 * SOUND_UNIT_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        # Add dummy frame + one actual frame
        dummy_frame = b'\x00' * SOUND_UNIT_SIZE
        actual_frame = b'\x01' * SOUND_UNIT_SIZE
        stream_data[AEA_META_SIZE:AEA_META_SIZE + SOUND_UNIT_SIZE] = dummy_frame
        stream_data[AEA_META_SIZE + SOUND_UNIT_SIZE:] = actual_frame
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        assert reader.metadata is not None
        assert reader.metadata.title == "Test Song"
        assert reader.metadata.channels == 2
        assert reader.metadata.total_frames == 1
        assert reader._close_on_exit is False

    def test_init_with_string_path_file_not_found(self):
        """Test initialization with non-existent file path."""
        with pytest.raises(AeaReaderError, match="Failed to open AEA file"):
            AeaReader("nonexistent_file.aea")

    def test_init_with_invalid_metadata(self):
        """Test initialization with invalid metadata."""
        # Create stream with insufficient data for metadata
        stream = io.BytesIO(b'\x00' * 10)
        with pytest.raises(AeaReaderError, match="Failed to read or parse AEA metadata header"):
            AeaReader(stream)

    def test_get_metadata_success(self):
        """Test successful metadata retrieval."""
        metadata = AeaMetadata()
        metadata.title = "Test"
        metadata.channels = 1
        metadata.total_frames = 0
        
        stream_data = bytearray(AEA_META_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        retrieved_metadata = reader.get_metadata()
        assert retrieved_metadata.title == "Test"
        assert retrieved_metadata.channels == 1

    def test_get_metadata_not_loaded(self):
        """Test metadata retrieval when not loaded."""
        stream = io.BytesIO(b'\x00' * AEA_META_SIZE)
        reader = AeaReader.__new__(AeaReader)  # Create without calling __init__
        reader.metadata = None
        
        with pytest.raises(AeaReaderError, match="Metadata not loaded"):
            reader.get_metadata()

    def test_get_total_file_frames(self):
        """Test total frame calculation."""
        metadata = AeaMetadata()
        metadata.total_frames = 5
        metadata.channels = 2
        
        stream_data = bytearray(AEA_META_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        assert reader._get_total_file_frames() == 5

    def test_get_total_file_frames_no_metadata(self):
        """Test total frame calculation without metadata."""
        reader = AeaReader.__new__(AeaReader)
        reader.metadata = None
        
        with pytest.raises(AeaReaderError, match="Metadata not loaded, cannot calculate total frames"):
            reader._get_total_file_frames()

    def test_frames_iterator_success(self):
        """Test successful frame iteration."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 2
        
        stream_data = bytearray(AEA_META_SIZE + 3 * SOUND_UNIT_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        # Add dummy frame + two actual frames
        dummy_frame = b'\x00' * SOUND_UNIT_SIZE
        frame1 = b'\x01' * SOUND_UNIT_SIZE
        frame2 = b'\x02' * SOUND_UNIT_SIZE
        
        offset = AEA_META_SIZE
        stream_data[offset:offset + SOUND_UNIT_SIZE] = dummy_frame
        offset += SOUND_UNIT_SIZE
        stream_data[offset:offset + SOUND_UNIT_SIZE] = frame1
        offset += SOUND_UNIT_SIZE
        stream_data[offset:offset + SOUND_UNIT_SIZE] = frame2
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        frames = list(reader.frames())
        assert len(frames) == 2
        assert frames[0] == frame1
        assert frames[1] == frame2

    def test_frames_iterator_no_metadata(self):
        """Test frame iteration without metadata."""
        reader = AeaReader.__new__(AeaReader)
        reader.metadata = None
        
        with pytest.raises(AeaReaderError, match="Metadata not loaded"):
            list(reader.frames())

    def test_frames_iterator_truncated_dummy_frame(self):
        """Test frame iteration with truncated dummy frame."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 1
        
        stream_data = bytearray(AEA_META_SIZE + SOUND_UNIT_SIZE - 10)  # Truncated
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        with pytest.raises(AeaReaderError, match="Failed to read full dummy frame"):
            list(reader.frames())

    def test_frames_iterator_zero_frames_with_short_dummy(self):
        """Test frame iteration with zero frames and short dummy frame."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 0
        
        stream_data = bytearray(AEA_META_SIZE + 10)  # Short dummy frame
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        frames = list(reader.frames())
        assert len(frames) == 0

    def test_frames_iterator_truncated_audio_frame(self):
        """Test frame iteration with truncated audio frame."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 1
        
        stream_data = bytearray(AEA_META_SIZE + SOUND_UNIT_SIZE + 100)  # Truncated audio frame
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        # Add full dummy frame
        dummy_frame = b'\x00' * SOUND_UNIT_SIZE
        stream_data[AEA_META_SIZE:AEA_META_SIZE + SOUND_UNIT_SIZE] = dummy_frame
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        with pytest.raises(AeaReaderError, match="Failed to read full audio frame.*File might be truncated"):
            list(reader.frames())

    def test_get_total_samples(self):
        """Test total sample calculation."""
        metadata = AeaMetadata()
        metadata.total_frames = 5
        metadata.channels = 2
        
        stream_data = bytearray(AEA_META_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        reader = AeaReader(stream)
        
        assert reader.get_total_samples() == 5 * 512  # 512 samples per frame

    def test_get_total_samples_no_metadata(self):
        """Test total sample calculation without metadata."""
        reader = AeaReader.__new__(AeaReader)
        reader.metadata = None
        
        with pytest.raises(AeaReaderError, match="Metadata not loaded"):
            reader.get_total_samples()

    def test_close_stream_opened_by_reader(self):
        """Test closing stream that was opened by reader."""
        reader = AeaReader.__new__(AeaReader)
        reader.stream = io.BytesIO(b'\x00' * 100)
        reader._close_on_exit = True
        
        reader.close()
        assert reader.stream.closed

    def test_close_external_stream(self):
        """Test not closing external stream."""
        stream = io.BytesIO(b'\x00' * 100)
        reader = AeaReader.__new__(AeaReader)
        reader.stream = stream
        reader._close_on_exit = False
        
        reader.close()
        assert not stream.closed

    def test_context_manager(self):
        """Test context manager functionality."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 0
        
        stream_data = bytearray(AEA_META_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        
        with AeaReader(stream) as reader:
            assert reader.metadata is not None
            assert reader.metadata.channels == 1

    def test_context_manager_with_exception(self):
        """Test context manager with exception handling."""
        metadata = AeaMetadata()
        metadata.channels = 1
        metadata.total_frames = 0
        
        stream_data = bytearray(AEA_META_SIZE)
        metadata_bytes = metadata.to_bytes()
        stream_data[:len(metadata_bytes)] = metadata_bytes
        
        stream = io.BytesIO(stream_data)
        
        with pytest.raises(ValueError):
            with AeaReader(stream) as reader:
                assert reader.metadata is not None
                raise ValueError("Test exception")

    def test_metadata_position_validation(self):
        """Test validation of stream position after metadata read."""
        # Create stream where metadata read doesn't reach expected position
        stream = io.BytesIO(b'\x00' * AEA_META_SIZE)
        
        # Mock the metadata reading to not advance stream properly
        original_read_from_stream = AeaMetadata.read_from_stream
        
        def mock_read_from_stream(stream):
            # Read less than expected, leaving stream at wrong position
            stream.read(100)  # Read only 100 bytes instead of full metadata
            metadata = AeaMetadata()
            return metadata
        
        AeaMetadata.read_from_stream = mock_read_from_stream
        
        try:
            with pytest.raises(AeaReaderError, match="Stream position after metadata read is incorrect"):
                AeaReader(stream)
        finally:
            AeaMetadata.read_from_stream = original_read_from_stream