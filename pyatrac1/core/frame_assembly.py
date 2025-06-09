# This file is being refactored.
# The FrameAssembler and FrameDisassembler classes, along with their
# internal BitStreamWriter and BitStreamReader, have been removed
# as their functionality is consolidated into pyatrac1.core.bitstream.py
# and other relevant modules.

# Imports that were used by the removed classes:
# import numpy as np
# from typing import List, Dict, Any, Tuple
# from pyatrac1.core.mdct import BlockSizeMode
# from .scaling_quantization import BitstreamSignedValues
# from pyatrac1.common.constants import (
#     ATRAC1_SOUND_UNIT_SIZE_BYTES,
#     ATRAC1_BITS_PER_BFU_AMOUNT_TAB_IDX,
#     ATRAC1_SPECS_PER_BLOCK,
#     ATRAC1_BFU_AMOUNT_TAB,
# )
