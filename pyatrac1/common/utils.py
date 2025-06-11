from typing import List, TypeVar

T = TypeVar("T")

"""
Common utility functions for the pyatrac1 project.
"""


def swap_array(arr: List[T]) -> List[T]:
    """
    Reverses the order of elements in a list.

    Args:
        arr: The input list.

    Returns:
        A new list with elements in reversed order.
    """
    return arr[::-1]


def bfu_to_band(bfu_idx: int) -> int:
    """
    Determines the frequency band (low, mid, high) for a given BFU index.
    Matches the logic of BfuToBand in atracdenc.

    Args:
        bfu_idx: The BFU index.

    Returns:
        An integer representing the band:
        0 for low band (BFU 0-19)
        1 for mid band (BFU 20-35)
        2 for high band (BFU 36 and above)
    """
    if bfu_idx < 20:
        return 0  # low band
    elif bfu_idx < 36:
        return 1  # mid band
    else:
        return 2  # high band
