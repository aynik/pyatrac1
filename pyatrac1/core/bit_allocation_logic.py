"""
Implements the bit allocation logic for ATRAC1, including dynamic bit
allocation based on psychoacoustic models and bit boosting.
"""

from typing import List, Tuple, TYPE_CHECKING

from ..common.constants import SOUND_UNIT_SIZE, MAX_BFUS, BITS_PER_IDWL, BITS_PER_IDSF # Added BITS_PER_IDWL, BITS_PER_IDSF
from ..core.mdct import BlockSizeMode
from ..common.utils import bfu_to_band
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
        block_size_mode: BlockSizeMode, # Changed parameter
        ath_scaled: List[float],
        analize_scale_factor_spread: float,
        shift: int,
    ) -> Tuple[List[int], int]:
        """
        Calculates the bit allocation for each Basic Frequency Unit (BFU).

        Args:
            scaled_blocks: List of scaled blocks, one for each BFU.
            block_size_mode: Object indicating if bands are using short or long blocks.
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

        num_active_bfus = len(scaled_blocks)

        for i in range(num_active_bfus):
            band = bfu_to_band(i)
            is_short_for_current_bfu = False
            if band == 0 and block_size_mode.low_band_short:
                is_short_for_current_bfu = True
            elif band == 1 and block_size_mode.mid_band_short:
                is_short_for_current_bfu = True
            elif band == 2 and block_size_mode.high_band_short:
                is_short_for_current_bfu = True

            # Determine fixed allocation part based on whether the BFU is short or long
            fixed_val = (FIXED_BIT_ALLOC_TABLE_SHORT[i]
                         if is_short_for_current_bfu
                         else FIXED_BIT_ALLOC_TABLE_LONG[i])

            # Ensure index is within bounds for the chosen table (should be guaranteed by num_active_bfus < MAX_BFUS)
            # However, the original code had a check `if i >= len(fixed_alloc_table):`,
            # which is implicitly handled if FIXED_BIT_ALLOC_TABLE_SHORT/LONG are MAX_BFUS long.
            # Assuming num_active_bfus <= MAX_BFUS and tables cover up to MAX_BFUS.

            if not is_short_for_current_bfu and scaled_blocks[i].max_energy < ath_scaled[i]:
                bits_per_bfu[i] = 0
            else:
                # Use atracdenc-compatible formula: 
                # spread * (ScaleFactorIndex/3.2) + (1.0 - spread) * fix - shift
                fixed_part = fixed_val # Use the dynamically determined fixed_val
                scale_factor_part = scaled_blocks[i].scale_factor_index / 3.2
                tmp = (analize_scale_factor_spread * scale_factor_part + 
                       (1.0 - analize_scale_factor_spread) * fixed_part - shift)
                
                if tmp > 16:
                    bits_per_bfu[i] = 16
                elif tmp < 2:
                    bits_per_bfu[i] = 0
                else:
                    bits_per_bfu[i] = int(tmp)

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
        scaled_blocks: List["ScaledBlock"], # Full list, up to MAX_BFUS based on initial_bfu_amount_idx
        block_size_mode: BlockSizeMode,
        ath_scaled: List[float], # Full list, ^ same as scaled_blocks
        analize_scale_factor_spread: float,
        initial_bfu_amount_idx: int,
        total_frame_bits: int,
        header_bits: int,
        trailer_bits: int,
    ) -> Tuple[List[int], int, int]:
        """
        Iteratively finds an optimal bit allocation and BFU count.
        This implements the logic from atracdenc's PerformIterativeBitAlloc.
        Returns:
            - final_wl_max_bfus: Word lengths (MAX_BFUS long).
            - final_mantissa_bits: Actual mantissa bits used for the active BFUs.
            - final_bfu_idx_used: The final bfu_amount_idx used.
        """
        current_bfu_idx = initial_bfu_amount_idx
        final_wl_max_bfus: List[int] = [0] * MAX_BFUS
        final_mantissa_bits = 0
        final_bfu_idx_used = initial_bfu_amount_idx

        max_outer_loops = 8  # As in atracdenc
        for _outer_loop_count in range(max_outer_loops):
            num_active_bfus = self.codec_data.bfu_amount_tab[current_bfu_idx]

            # Slice inputs for the current number of active BFUs
            iter_scaled_blocks = scaled_blocks[:num_active_bfus]
            iter_ath_scaled = ath_scaled[:num_active_bfus]

            wl_header_bits = num_active_bfus * BITS_PER_IDWL
            sf_header_bits = num_active_bfus * BITS_PER_IDSF

            bits_available_for_mantissas = (
                total_frame_bits - header_bits - trailer_bits -
                wl_header_bits - sf_header_bits
            )

            target_mantissa_bits = bits_available_for_mantissas
            # min_acceptable_mantissa_bits in C++ is target - 110 (0.5 * 220)
            min_acceptable_mantissa_bits = max(0, target_mantissa_bits - 110)

            # Inner binary search for shift parameter
            best_shift = 3.0  # Initial guess, as in C++
            min_shift, max_shift = -3.0, 15.0

            best_alloc_for_current_num_bfus: List[int] = [] # Active length
            best_bits_for_current_num_bfus = -1

            max_inner_iters = 30 # Similar to C++
            for _inner_loop_count in range(max_inner_iters):
                current_alloc_active, current_bits_active = self.calc_bits_allocation(
                    iter_scaled_blocks,
                    block_size_mode,
                    iter_ath_scaled,
                    analize_scale_factor_spread,
                    int(round(best_shift)), # shift is int
                )

                # Store this result as potentially the best for this num_active_bfus
                # if it's the first or better than previous under min_acceptable
                if best_bits_for_current_num_bfus == -1 or \
                   (current_bits_active < min_acceptable_mantissa_bits and \
                    current_bits_active > best_bits_for_current_num_bfus) : # if under, try to get closer from below
                    best_alloc_for_current_num_bfus = list(current_alloc_active)
                    best_bits_for_current_num_bfus = current_bits_active

                if min_acceptable_mantissa_bits <= current_bits_active <= target_mantissa_bits:
                    # Found a good allocation within the target range
                    best_alloc_for_current_num_bfus = list(current_alloc_active)
                    best_bits_for_current_num_bfus = current_bits_active
                    break # Exit inner binary search loop

                # Binary search adjustment for shift
                delta = max_shift - min_shift
                if delta < 0.1: # Convergence condition
                    # If not in range, best_alloc/bits will hold the last tried or closest from below
                    break

                if current_bits_active > target_mantissa_bits:
                    min_shift = best_shift
                elif current_bits_active < min_acceptable_mantissa_bits:
                    max_shift = best_shift

                best_shift = (min_shift + max_shift) / 2.0
            # End of inner binary search loop

            # Ensure best_alloc_for_current_num_bfus is initialized if loop didn't run once
            if not best_alloc_for_current_num_bfus and num_active_bfus > 0 :
                 # This can happen if target_mantissa_bits is extremely low or negative
                 # Default to zero allocation for safety
                best_alloc_for_current_num_bfus = [0] * num_active_bfus
                best_bits_for_current_num_bfus = 0
            elif num_active_bfus == 0: # Handle case of no active BFUs
                best_alloc_for_current_num_bfus = []
                best_bits_for_current_num_bfus = 0


            candidate_wl_max_bfus = [0] * MAX_BFUS
            if num_active_bfus > 0:
                candidate_wl_max_bfus[:num_active_bfus] = best_alloc_for_current_num_bfus

            # Pass the active portion of word lengths to check_bfu_usage
            # The length of best_alloc_for_current_num_bfus is num_active_bfus
            new_bfu_idx, bfu_changed = self.check_bfu_usage(current_bfu_idx, best_alloc_for_current_num_bfus)

            if bfu_changed:
                current_bfu_idx = new_bfu_idx
                final_wl_max_bfus = list(candidate_wl_max_bfus)
                final_mantissa_bits = best_bits_for_current_num_bfus
                final_bfu_idx_used = current_bfu_idx
                # Continue outer loop to re-evaluate with new num_active_bfus
            else:
                # No change in BFU count, this is our final allocation
                final_wl_max_bfus = list(candidate_wl_max_bfus)
                final_mantissa_bits = best_bits_for_current_num_bfus
                final_bfu_idx_used = current_bfu_idx
                break # Exit outer loop

        return final_wl_max_bfus, final_mantissa_bits, final_bfu_idx_used

    def get_max_used_bfu_idx(self, word_lengths: List[int], current_bfu_amount_idx: int) -> int:
        """
        Determines the smallest bfu_amount_idx that still covers all active BFUs.
        word_lengths corresponds to the number of BFUs active for current_bfu_amount_idx.
        """
        # test_idx is an index into self.codec_data.bfu_amount_tab
        test_idx = current_bfu_amount_idx
        while test_idx > 0:
            # Number of BFUs defined by the tier *above* test_idx-1
            num_bfus_for_tier_upper_bound = self.codec_data.bfu_amount_tab[test_idx]
            # Number of BFUs defined by the tier test_idx-1 itself
            num_bfus_for_tier_lower_bound = self.codec_data.bfu_amount_tab[test_idx - 1]

            tier_has_non_zero_allocations = False

            # Iterate through the BFU indices that define this specific tier.
            # This tier consists of BFUs from index num_bfus_for_tier_lower_bound
            # up to num_bfus_for_tier_upper_bound - 1.
            start_bfu_in_tier = num_bfus_for_tier_lower_bound

            # word_lengths has length corresponding to num_bfus for current_bfu_amount_idx.
            # We only check indices within word_lengths.
            # The upper bound for checking is min(tier's natural end, word_lengths' actual end).
            # Example: current_bfu_amount_idx = 7 (52 BFUs), len(word_lengths) = 52.
            # test_idx = 7. upper_bound = 52. lower_bound = bfu_amount_tab[6] = 48.
            # Tier is BFUs 48, 49, 50, 51. Check word_lengths[48] to word_lengths[51].
            end_bfu_in_tier_inclusive = min(num_bfus_for_tier_upper_bound - 1, len(word_lengths) - 1)

            for i in range(start_bfu_in_tier, end_bfu_in_tier_inclusive + 1):
                if word_lengths[i] != 0:
                    tier_has_non_zero_allocations = True
                    break

            if tier_has_non_zero_allocations:
                # This tier (from test_idx-1 up to test_idx) contains non-zero allocations.
                # So, test_idx is the correct bfu_amount_idx to return.
                return test_idx
            else:
                # This tier is all zeros. Try the next lower bfu_amount_idx.
                test_idx -= 1

        # If loop finishes, test_idx is 0.
        # This means all tiers from 1 up to current_bfu_amount_idx were empty.
        # So, the allocations must be confined to the BFUs covered by bfu_amount_tab[0],
        # or all allocations are zero. In either case, 0 is the correct index.
        return test_idx

    def check_bfu_usage(self, current_bfu_amount_idx: int, current_word_lengths: List[int]) -> Tuple[int, bool]:
        """
        Checks if the number of active BFUs can be reduced based on actual bit allocations.
        current_word_lengths is a list of bit allocations, its length is determined by
        self.codec_data.bfu_amount_tab[current_bfu_amount_idx].
        """
        # current_word_lengths contains allocations for BFUs 0 up to N-1,
        # where N = self.codec_data.bfu_amount_tab[current_bfu_amount_idx].
        actual_max_bfu_idx = self.get_max_used_bfu_idx(current_word_lengths, current_bfu_amount_idx)

        changed = False
        new_bfu_amount_idx = current_bfu_amount_idx

        if actual_max_bfu_idx < current_bfu_amount_idx:
            changed = True
            new_bfu_amount_idx = actual_max_bfu_idx

        return (new_bfu_amount_idx, changed)


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
                if cost_for_bfu <= surplus_bits and (boosted_bits_per_bfu[i] + 2 <= 16):
                    boosted_bits_per_bfu[i] += 2
                    surplus_bits -= cost_for_bfu
                    bits_consumed_by_boost += cost_for_bfu

        # Priority 2: BFUs needing 1 bit (from an existing allocation)
        for i in range(num_active_bfus):
            if surplus_bits <= 0:
                break
            if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] > 0:
                cost_for_bfu = 1 * self.codec_data.specs_per_block[i]
                if cost_for_bfu <= surplus_bits and (boosted_bits_per_bfu[i] + 1 <= 16):
                    boosted_bits_per_bfu[i] += 1
                    surplus_bits -= cost_for_bfu
                    bits_consumed_by_boost += cost_for_bfu

        return boosted_bits_per_bfu, bits_consumed_by_boost


# if __name__ == "__main__":
#
#     class MockCodecDataImpl:
#         def __init__(self):
#             # self.ath_long = [10.0] * MAX_BFUS # Example ATH values
#             self.specs_per_block = [
#                 8,
#                 8,
#                 8,
#                 8,
#                 4,
#                 4,
#                 4,
#                 4,
#                 8,
#                 8,
#                 8,
#                 8,
#                 6,
#                 6,
#                 6,
#                 6,
#                 6,
#                 6,
#                 6,
#                 6,  # Low (20)
#                 6,
#                 6,
#                 6,
#                 6,
#                 7,
#                 7,
#                 7,
#                 7,
#                 9,
#                 9,
#                 9,
#                 9,
#                 10,
#                 10,
#                 10,
#                 10,  # Mid (16)
#                 12,
#                 12,
#                 12,
#                 12,
#                 12,
#                 12,
#                 12,
#                 12,
#                 20,
#                 20,
#                 20,
#                 20,
#                 20,
#                 20,
#                 20,
#                 20,  # High (16)
#             ]  # Total 52 BFUs
#             # Ensure specs_per_block is MAX_BFUS long if constants change
#             if len(self.specs_per_block) < MAX_BFUS:
#                 self.specs_per_block.extend(
#                     [0] * (MAX_BFUS - len(self.specs_per_block))
#                 )
#             elif len(self.specs_per_block) > MAX_BFUS:
#                 self.specs_per_block = self.specs_per_block[:MAX_BFUS]
#
#             self.bfu_amount_tab = [ # Example, should match constants.py
#                 12, 16, 20, 24, 28, 32, 36, MAX_BFUS
#             ]
#
#
#     mock_data_instance = MockCodecDataImpl()
#     allocator_instance = Atrac1SimpleBitAlloc(mock_data_instance)  # type: ignore
#     booster_instance = BitsBooster(mock_data_instance)  # type: ignore
#
#     example_scaled_blocks_list = [
#         ScaledBlock(max_energy=100.0, scaled_values=[], scale_factor_index=0)
#         for _ in range(MAX_BFUS)  # Create for all possible BFUs
#     ]
#     example_ath_scaled_list = [5.0] * MAX_BFUS  # Create for all possible BFUs
#     # example_num_bfus_active = 36 # This is now determined by initial_bfu_amount_idx
#     initial_bfu_amount_idx_example = 6 # Corresponds to 36 BFUs in a typical table
#
#     try:
#         from ..common.constants import (
#             BITS_PER_BFU_AMOUNT_TAB_IDX,
#             FRAME_TRAILER_BITS
#         )
#     except ImportError:
#         BITS_PER_BFU_AMOUNT_TAB_IDX = 3
#         FRAME_TRAILER_BITS = 0 # Example
#
#
#     header_control_bits_example = 2 + 2 + 2 + 2 + BITS_PER_BFU_AMOUNT_TAB_IDX + 2 + 3
#     total_frame_bits_example = SOUND_UNIT_SIZE * 8
#     trailer_bits_example = FRAME_TRAILER_BITS
#
#     mock_bsm = BlockSizeMode(low_band_short=False, mid_band_short=False, high_band_short=False)
#
#     word_lengths_full, total_mantissa_bits_used_val, final_bfu_idx_val = (
#         allocator_instance.perform_iterative_allocation(
#             example_scaled_blocks_list, # Pass full list
#             block_size_mode=mock_bsm,
#             ath_scaled=example_ath_scaled_list, # Pass full list
#             analize_scale_factor_spread=0.5,
#             initial_bfu_amount_idx=initial_bfu_amount_idx_example,
#             total_frame_bits=total_frame_bits_example,
#             header_bits=header_control_bits_example,
#             trailer_bits=trailer_bits_example
#         )
#     )
#     print(f"Final BFU Idx: {final_bfu_idx_val}, Mantissa Bits: {total_mantissa_bits_used_val}")
#     # print(f"Word Lengths: {word_lengths_full}")
