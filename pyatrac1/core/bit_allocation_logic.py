"""
Implements the bit allocation logic for ATRAC1, including dynamic bit
allocation based on psychoacoustic models and bit boosting.
"""

from typing import List, Tuple, TYPE_CHECKING

from ..common.constants import SOUND_UNIT_SIZE, MAX_BFUS
from ..tables.bit_allocation import (
    FIXED_BIT_ALLOC_TABLE_LONG,
    FIXED_BIT_ALLOC_TABLE_SHORT,
    BIT_BOOST_MASK,
)

if TYPE_CHECKING:
    from .codec_data import Atrac1CodecData, ScaledBlock


class Atrac1SimpleBitAlloc:
    """
    Handles the core bit allocation calculation for ATRAC1, aiming to fit
    spectral data within a fixed bit budget per frame.
    """

    def __init__(self, codec_data: "Atrac1CodecData"):
        """
        Initializes the bit allocator.
        Args:
            codec_data: An object containing necessary codec tables and state,
                        like ATHLong and specs_per_block.
        """
        self.codec_data: "Atrac1CodecData" = codec_data

    def calc_bits_allocation(
        self,
        scaled_blocks: List["ScaledBlock"],
        is_long_block: bool,
        ath_scaled: List[float],
        analize_scale_factor_spread: float,  # pylint: disable=unused-argument
        shift: int,
    ) -> Tuple[List[int], int]:
        """
        Calculates the bit allocation for each Basic Frequency Unit (BFU).

        Args:
            scaled_blocks: List of scaled blocks, one for each BFU.
            is_long_block: True if long MDCT blocks are used, False for short.
            ath_scaled: Absolute Threshold of Hearing values, scaled by loudness, for each BFU.
            analize_scale_factor_spread: Value indicating tonal vs. noise-like characteristics.
            shift: A shift value used in the iterative optimization process to adjust
                   the overall bit allocation.

        Returns:
            A tuple containing:
                - A list of integers representing bits allocated per BFU (word lengths).
                  The list length is num_active_bfus.
                - Total bits used by mantissas with this allocation.
        """
        bits_per_bfu: List[int] = [0] * MAX_BFUS  # Initialize for max possible BFUs

        fixed_alloc_table = (
            FIXED_BIT_ALLOC_TABLE_LONG if is_long_block else FIXED_BIT_ALLOC_TABLE_SHORT
        )

        num_active_bfus = len(scaled_blocks)

        for i in range(num_active_bfus):
            if i >= len(fixed_alloc_table):
                bits_per_bfu[i] = 0
                continue

            if scaled_blocks[i].max_energy < ath_scaled[i]:
                bits_per_bfu[i] = 0
            else:
                allocated = max(0, fixed_alloc_table[i] - shift)
                bits_per_bfu[i] = allocated

            if bits_per_bfu[i] == 1:
                bits_per_bfu[i] = 0

            bits_per_bfu[i] = min(bits_per_bfu[i], 15)  # Max word length (0-15)

        actual_mantissa_bits = 0
        for i in range(num_active_bfus):
            if bits_per_bfu[i] > 0:
                actual_mantissa_bits += (
                    bits_per_bfu[i] * self.codec_data.specs_per_block[i]
                )

        # Return only the portion of bits_per_bfu relevant to num_active_bfus
        return bits_per_bfu[:num_active_bfus], actual_mantissa_bits

    def perform_iterative_allocation(
        self,
        scaled_blocks: List["ScaledBlock"],  # List of active scaled blocks
        is_long_block: bool,
        ath_scaled: List[float],  # List of active ATH values
        analize_scale_factor_spread: float,
        num_active_bfus: int,
        bits_available_for_mantissas: int,
    ) -> Tuple[List[int], int]:
        """
        Iteratively finds an optimal shift value to fit the allocated bits
        within the available budget.
        Returns the best allocation (word lengths for all MAX_BFUS, where non-active are 0)
        and the total mantissa bits used by that allocation for active BFUs.
        """
        # Stores the best allocation found so far that is <= target
        best_under_budget_allocation: List[int] = [0] * MAX_BFUS
        best_under_budget_mantissa_bits = -1

        # Stores the best allocation found if all are > target (closest to target)
        closest_over_budget_allocation: List[int] = [0] * MAX_BFUS
        closest_over_budget_mantissa_bits = float("inf")

        for shift_try in range(-8, 17):  # Example range for shift
            # calc_bits_allocation expects lists for active BFUs
            current_bits_per_bfu_active, current_total_mantissa_bits = (
                self.calc_bits_allocation(
                    scaled_blocks,  # Already sliced to num_active_bfus
                    is_long_block,
                    ath_scaled,  # Already sliced to num_active_bfus
                    analize_scale_factor_spread,
                    shift_try,
                )
            )

            # Pad current_bits_per_bfu_active to MAX_BFUS length for consistent storage
            full_current_bits_per_bfu = [0] * MAX_BFUS
            if num_active_bfus > 0:  # Ensure current_bits_per_bfu_active is not empty
                full_current_bits_per_bfu[:num_active_bfus] = (
                    current_bits_per_bfu_active
                )

            if current_total_mantissa_bits <= bits_available_for_mantissas:
                # This allocation is within budget
                if current_total_mantissa_bits > best_under_budget_mantissa_bits:
                    # It's better (uses more bits without exceeding) than previous under-budget ones
                    best_under_budget_mantissa_bits = current_total_mantissa_bits
                    best_under_budget_allocation = list(full_current_bits_per_bfu)
            else:
                # This allocation is over budget, track the closest one (only consider non-zero allocations)
                if current_total_mantissa_bits > 0 and current_total_mantissa_bits < closest_over_budget_mantissa_bits:
                    closest_over_budget_mantissa_bits = current_total_mantissa_bits
                    closest_over_budget_allocation = list(full_current_bits_per_bfu)

        if best_under_budget_mantissa_bits > 0:
            # We found a meaningful allocation within budget (non-zero bits)
            return best_under_budget_allocation, best_under_budget_mantissa_bits

        # Check for zero budget case - should always return 0 bits
        if bits_available_for_mantissas == 0:
            return [0] * MAX_BFUS, 0

        # All allocations were over budget, use the one closest to the target
        if closest_over_budget_mantissa_bits == float("inf"):
            # No valid allocation found, return all zeros
            return [0] * MAX_BFUS, 0
        # Ensure the returned type is int
        return closest_over_budget_allocation, int(closest_over_budget_mantissa_bits)


class BitsBooster:
    """
    Handles the distribution of surplus bits to eligible Basic Frequency Units (BFUs)
    after the initial bit allocation.
    """

    def __init__(self, codec_data: "Atrac1CodecData"):
        self.codec_data: "Atrac1CodecData" = codec_data

    def apply_boost(
        self,
        current_bits_per_bfu: List[int],  # Should be MAX_BFUS long
        surplus_bits: int,
        num_active_bfus: int,
    ) -> Tuple[List[int], int]:
        """
        Distributes surplus bits to eligible BFUs.
        current_bits_per_bfu is assumed to be MAX_BFUS long.
        """
        boosted_bits_per_bfu = list(current_bits_per_bfu)  # Make a copy
        bits_consumed_by_boost = 0

        # Priority 1: BFUs needing 2 bits (currently 0, eligible for boost)
        for i in range(num_active_bfus):
            if surplus_bits <= 0:
                break
            # BIT_BOOST_MASK is MAX_BFUS long
            if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] == 0:
                cost_for_bfu = 2 * self.codec_data.specs_per_block[i]
                if cost_for_bfu <= surplus_bits and (boosted_bits_per_bfu[i] + 2 <= 15):
                    boosted_bits_per_bfu[i] += 2
                    surplus_bits -= cost_for_bfu
                    bits_consumed_by_boost += cost_for_bfu

        # Priority 2: BFUs needing 1 bit (from an existing allocation)
        for i in range(num_active_bfus):
            if surplus_bits <= 0:
                break
            if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] > 0:
                cost_for_bfu = 1 * self.codec_data.specs_per_block[i]
                if cost_for_bfu <= surplus_bits and (boosted_bits_per_bfu[i] + 1 <= 15):
                    boosted_bits_per_bfu[i] += 1
                    surplus_bits -= cost_for_bfu
                    bits_consumed_by_boost += cost_for_bfu

        return boosted_bits_per_bfu, bits_consumed_by_boost


if __name__ == "__main__":

    class MockCodecDataImpl:
        def __init__(self):
            # self.ath_long = [10.0] * MAX_BFUS # Example ATH values
            self.specs_per_block = [
                8,
                8,
                8,
                8,
                4,
                4,
                4,
                4,
                8,
                8,
                8,
                8,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,  # Low (20)
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                9,
                9,
                9,
                9,
                10,
                10,
                10,
                10,  # Mid (16)
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                12,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,  # High (16)
            ]  # Total 52 BFUs
            # Ensure specs_per_block is MAX_BFUS long if constants change
            if len(self.specs_per_block) < MAX_BFUS:
                self.specs_per_block.extend(
                    [0] * (MAX_BFUS - len(self.specs_per_block))
                )
            elif len(self.specs_per_block) > MAX_BFUS:
                self.specs_per_block = self.specs_per_block[:MAX_BFUS]

    mock_data_instance = MockCodecDataImpl()
    allocator_instance = Atrac1SimpleBitAlloc(mock_data_instance)  # type: ignore
    booster_instance = BitsBooster(mock_data_instance)  # type: ignore

    example_scaled_blocks_list = [
        ScaledBlock(max_energy=100.0, scaled_values=[], scale_factor_index=0)
        for _ in range(MAX_BFUS)  # Create for all possible BFUs
    ]
    example_ath_scaled_list = [5.0] * MAX_BFUS  # Create for all possible BFUs
    example_num_bfus_active = 36

    # Using constants from pyatrac1.common.constants (or define them if not available)
    # These should be imported or defined if used here, for now, hardcoding for example
    # If pyatrac1.common.constants defines BITS_PER_IDWL etc., use those.
    try:
        from ..common.constants import (
            BITS_PER_IDWL,
            BITS_PER_IDSF,
            BITS_PER_BFU_AMOUNT_TAB_IDX,
        )

        bits_per_idwl_const = BITS_PER_IDWL
        bits_per_idsf_const = BITS_PER_IDSF
        bits_per_bfu_amount_tab_idx_const = BITS_PER_BFU_AMOUNT_TAB_IDX
    except ImportError:  # Fallback if running script directly and relative imports fail
        bits_per_idwl_const = 4
        bits_per_idsf_const = 6
        bits_per_bfu_amount_tab_idx_const = 3

    bsm_total_bits = 6  # 2 bits per band * 3 bands

    header_other_bits = bsm_total_bits + bits_per_bfu_amount_tab_idx_const
    wl_header_bits = example_num_bfus_active * bits_per_idwl_const
    sf_header_bits = example_num_bfus_active * bits_per_idsf_const

    total_frame_bits_val = SOUND_UNIT_SIZE * 8
    available_mantissa_bits_calc = (
        total_frame_bits_val - header_other_bits - wl_header_bits - sf_header_bits
    )

    # Pass only the active scaled_blocks and ath_scaled to perform_iterative_allocation
    word_lengths_full, total_mantissa_bits_used_val = (
        allocator_instance.perform_iterative_allocation(
            example_scaled_blocks_list[:example_num_bfus_active],
            is_long_block=True,
            ath_scaled=example_ath_scaled_list[:example_num_bfus_active],
            analize_scale_factor_spread=0.5,
            num_active_bfus=example_num_bfus_active,
            bits_available_for_mantissas=available_mantissa_bits_calc,
        )
    )
