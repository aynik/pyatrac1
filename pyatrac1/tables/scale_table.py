import math
from typing import List


def generate_scale_table() -> List[float]:
    """
    Generates the ScaleTable for ATRAC1 codec as per the technical specification.
    Formula: ScaleTable[i] = pow(2.0, (i / 3.0 - 21.0)) for i from 0 to 63.
    """
    scale_table: List[float] = []
    for i in range(64):
        value: float = math.pow(2.0, (i / 3.0 - 21.0))
        scale_table.append(value)
    return scale_table


ATRAC1_SCALE_TABLE: List[float] = generate_scale_table()
