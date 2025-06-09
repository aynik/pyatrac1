"""
Handles writing of AEA (AtracDEnc Audio) files, including metadata,
the initial dummy frame, and compressed audio frames.
Based on spec.txt section 6.
"""

from typing import BinaryIO, Type, Optional
from types import TracebackType
from .metadata import AeaMetadata
from ..common.constants import AEA_META_SIZE, SOUND_UNIT_SIZE


class AeaWriterError(Exception):
    """Custom exception for AEA writer errors."""

    pass


class AeaWriter:
    """
    Writes AEA file metadata and compressed audio frames.
    """

    def __init__(
        self, filepath_or_stream: str | BinaryIO, channel_count: int, title: str = ""
    ):
        """
        Initializes the AEA writer.

        Args:
            filepath_or_stream: Path to the AEA file to create/overwrite or an
                                already open binary stream for writing.
            channel_count: Number of audio channels (1 or 2).
            title: Optional title for the AEA metadata (max 15 chars).
        """
        if not 1 <= channel_count <= 2:
            raise AeaWriterError(f"Channel count must be 1 or 2, got {channel_count}")

        if isinstance(filepath_or_stream, str):
            try:
                self.stream: BinaryIO = open(filepath_or_stream, "wb")
            except IOError as e:
                raise AeaWriterError(
                    f"Failed to open AEA file for writing: {filepath_or_stream}"
                ) from e
            self._close_on_exit = True
        else:
            self.stream: BinaryIO = filepath_or_stream
            self._close_on_exit = False

        self.metadata: AeaMetadata = AeaMetadata(
            title=title[:15], total_frames=0, channel_count=channel_count
        )
        self._frame_count: int = 0
        self._first_write_done: bool = False  # To track if initial setup is done
        self._write_initial_header_and_dummy()

    def _write_initial_header_and_dummy(self):
        """
        Writes a placeholder metadata header and the initial dummy frame.
        The header will be updated later with the correct total_frames.
        """
        try:
            self.stream.seek(0)
            self.metadata.write_to_stream(self.stream)

            # Write 5 dummy frames as expected by atracdenc calculation: (file_size - header) / 212 / channels - 5
            dummy_frame = bytearray(SOUND_UNIT_SIZE)
            for i in range(5):
                self.stream.write(dummy_frame)

            expected_pos = AEA_META_SIZE + (5 * SOUND_UNIT_SIZE)
            if self.stream.tell() != expected_pos:
                raise AeaWriterError(
                    f"Stream position after writing initial header and 5 dummy frames is incorrect. "
                    f"Expected {expected_pos}, got {self.stream.tell()}."
                )
            self._first_write_done = True
        except IOError as e:
            raise AeaWriterError(
                "Failed to write initial AEA header or dummy frame."
            ) from e

    def write_frame(self, frame_bytes: bytes):
        """
        Writes a compressed audio frame (212 bytes) to the stream.
        """
        if not self._first_write_done:
            # Should not happen if constructor logic is sound
            raise AeaWriterError(
                "Initial header and dummy frame not written. Call internal setup first."
            )
        if len(frame_bytes) != SOUND_UNIT_SIZE:
            raise AeaWriterError(
                f"Frame bytes length must be {SOUND_UNIT_SIZE}, got {len(frame_bytes)}"
            )

        try:
            self.stream.write(frame_bytes)
            self._frame_count += 1
        except IOError as e:
            raise AeaWriterError("Failed to write audio frame.") from e

    def finalize(self):
        """
        Updates the metadata header with the final total_frames count
        and ensures all data is flushed.
        """
        self.metadata.total_frames = self._frame_count
        current_pos = None
        try:
            current_pos = self.stream.tell()
            self.stream.seek(0)
            self.metadata.write_to_stream(self.stream)
            self.stream.flush()
        except IOError as e:
            raise AeaWriterError("Failed to finalize AEA metadata header.") from e
        finally:
            if current_pos is not None:
                try:
                    self.stream.seek(current_pos)
                except IOError:
                    pass

    def close(self):
        """
        Finalizes the AEA file and closes the stream if opened by this writer.
        """
        if self.stream and not self.stream.closed:
            try:
                self.finalize()
            finally:
                if self._close_on_exit:
                    self.stream.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.close()
        return False
