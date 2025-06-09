"""
Handles reading of AEA (AtracDEnc Audio) files, including metadata
and compressed audio frames.
Based on spec.txt section 6.
"""

from typing import BinaryIO, Optional, Iterator, Type
from types import TracebackType
from .metadata import AeaMetadata
from ..common.constants import AEA_META_SIZE, SOUND_UNIT_SIZE


class AeaReaderError(Exception):
    """Custom exception for AEA reader errors."""

    pass


class AeaReader:
    """
    Reads AEA file metadata and provides an iterator for audio frames.
    """

    def __init__(self, filepath_or_stream: str | BinaryIO):
        """
        Initializes the AEA reader.

        Args:
            filepath_or_stream: Path to the AEA file or an already open binary stream.
        """
        if isinstance(filepath_or_stream, str):
            try:
                self.stream: BinaryIO = open(filepath_or_stream, "rb")
            except IOError as e:
                raise AeaReaderError(
                    f"Failed to open AEA file: {filepath_or_stream}"
                ) from e
            self._close_on_exit = True
        else:
            self.stream: BinaryIO = filepath_or_stream
            self._close_on_exit = False

        self.metadata: Optional[AeaMetadata] = None
        self._initial_frame_offset: int = 0
        self._read_metadata()

    def _read_metadata(self):
        """Reads and parses the AEA metadata header."""
        try:
            self.stream.seek(0)
            self.metadata = AeaMetadata.read_from_stream(self.stream)
            self._initial_frame_offset = self.stream.tell()  # Position after metadata
            if self._initial_frame_offset != AEA_META_SIZE:
                raise AeaReaderError(
                    f"Stream position after metadata read is incorrect: "
                    f"{self._initial_frame_offset}, expected {AEA_META_SIZE}"
                )
        except (IOError, EOFError, ValueError) as e:
            raise AeaReaderError("Failed to read or parse AEA metadata header.") from e

    def get_metadata(self) -> AeaMetadata:
        """Returns the parsed AEA metadata."""
        if self.metadata is None:
            raise AeaReaderError("Metadata not loaded.")
        return self.metadata

    def _get_total_file_frames(self) -> int:  # pylint: disable=unused-private-member
        """
        Calculates the total number of audio frames in the file,
        accounting for the dummy frame.
        The file size based calculation was removed as it was unused.
        """
        if self.metadata is None:
            raise AeaReaderError("Metadata not loaded, cannot calculate total frames.")

        return self.metadata.total_frames

    def frames(self) -> Iterator[bytes]:
        """
        Yields actual compressed audio frames (212 bytes each), skipping the dummy frame.
        """
        if self.metadata is None:
            raise AeaReaderError("Metadata not loaded.")

        self.stream.seek(self._initial_frame_offset)

        dummy_frame = self.stream.read(SOUND_UNIT_SIZE)
        if len(dummy_frame) != SOUND_UNIT_SIZE:
            if self.metadata.total_frames == 0 and len(dummy_frame) < SOUND_UNIT_SIZE:
                return
            raise AeaReaderError(
                f"Failed to read full dummy frame. Expected {SOUND_UNIT_SIZE} bytes, got {len(dummy_frame)}"
            )

        for _ in range(self.metadata.total_frames):
            frame_bytes = self.stream.read(SOUND_UNIT_SIZE)
            if len(frame_bytes) != SOUND_UNIT_SIZE:
                raise AeaReaderError(
                    f"Failed to read full audio frame. Expected {SOUND_UNIT_SIZE} bytes, "
                    f"got {len(frame_bytes)}. File might be truncated."
                )
            yield frame_bytes

    def get_total_samples(self) -> int:
        """
        Calculates the total number of PCM samples in the AEA file per channel.
        """
        if self.metadata is None:
            raise AeaReaderError("Metadata not loaded.")
        return self.metadata.total_frames * 512

    def close(self):
        """Closes the stream if it was opened by this reader."""
        if self._close_on_exit and self.stream:
            if not self.stream.closed:
                self.stream.close()
            # self.stream = None # type: ignore

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.close()
        return False  # Do not suppress exceptions
