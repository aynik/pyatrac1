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
