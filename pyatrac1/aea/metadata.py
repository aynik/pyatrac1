"""
Handles the AEA (AtracDEnc Audio) file format metadata header.
Based on spec.txt section 6.1.
"""

import struct
from typing import BinaryIO
from ..common.constants import AEA_META_SIZE

AEA_MAGIC_NUMBER = b"\x00\x08\x00\x00"


class AeaMetadata:
    """
    Represents and handles the AEA metadata header.
    """

    MAGIC_NUMBER_OFFSET = 0
    MAGIC_NUMBER_SIZE = 4
    TITLE_OFFSET = 4
    TITLE_SIZE = 16
    TOTAL_FRAMES_OFFSET = 260
    TOTAL_FRAMES_SIZE = 4
    CHANNEL_COUNT_OFFSET = 264
    CHANNEL_COUNT_SIZE = 1

    def __init__(self, title: str = "", total_frames: int = 0, channel_count: int = 0):
        self.title = title
        self.total_frames = total_frames
        self.channel_count = channel_count

    @property
    def channels(self) -> int:
        """Alias for channel_count for compatibility."""
        return self.channel_count

    @channels.setter
    def channels(self, value: int):
        """Setter for channels property."""
        self.channel_count = value

    def pack(self) -> bytes:
        """
        Packs the metadata into a 2048-byte header.
        """
        header = bytearray(AEA_META_SIZE)

        header[
            self.MAGIC_NUMBER_OFFSET : self.MAGIC_NUMBER_OFFSET + self.MAGIC_NUMBER_SIZE
        ] = AEA_MAGIC_NUMBER

        title_bytes = self.title.encode("utf-8")[: self.TITLE_SIZE - 1]
        header[self.TITLE_OFFSET : self.TITLE_OFFSET + len(title_bytes)] = title_bytes

        struct.pack_into("<I", header, self.TOTAL_FRAMES_OFFSET, self.total_frames)

        if not 1 <= self.channel_count <= 2:
            raise ValueError(f"Channel count must be 1 or 2, got {self.channel_count}")
        struct.pack_into("<B", header, self.CHANNEL_COUNT_OFFSET, self.channel_count)

        return bytes(header)

    def to_bytes(self) -> bytes:
        """Alias for pack() method for compatibility."""
        return self.pack()

    @classmethod
    def unpack(cls, header_bytes: bytes) -> "AeaMetadata":
        """
        Unpacks a 2048-byte header into an AeaMetadata object.
        """
        if len(header_bytes) != AEA_META_SIZE:
            raise ValueError(
                f"Header bytes must be {AEA_META_SIZE} bytes long, got {len(header_bytes)}"
            )

        magic = header_bytes[
            cls.MAGIC_NUMBER_OFFSET : cls.MAGIC_NUMBER_OFFSET + cls.MAGIC_NUMBER_SIZE
        ]
        if magic != AEA_MAGIC_NUMBER:
            raise ValueError(
                f"Invalid AEA magic number. Expected {AEA_MAGIC_NUMBER!r}, got {magic!r}"
            )

        title_raw = header_bytes[cls.TITLE_OFFSET : cls.TITLE_OFFSET + cls.TITLE_SIZE]
        title = title_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")

        (total_frames,) = struct.unpack_from(
            "<I", header_bytes, cls.TOTAL_FRAMES_OFFSET
        )

        (channel_count,) = struct.unpack_from(
            "<B", header_bytes, cls.CHANNEL_COUNT_OFFSET
        )
        if not 1 <= channel_count <= 2:
            raise ValueError(f"Invalid channel count in header: {channel_count}")

        return cls(title=title, total_frames=total_frames, channel_count=channel_count)

    @classmethod
    def read_from_stream(cls, stream: BinaryIO) -> "AeaMetadata":
        """Reads and unpacks the metadata header from a binary stream."""
        header_bytes = stream.read(AEA_META_SIZE)
        if len(header_bytes) != AEA_META_SIZE:
            raise EOFError(
                f"Could not read {AEA_META_SIZE} bytes for AEA metadata header."
            )
        return cls.unpack(header_bytes)

    def write_to_stream(self, stream: BinaryIO):
        """Packs and writes the metadata header to a binary stream."""
        header_bytes = self.pack()
        stream.write(header_bytes)
