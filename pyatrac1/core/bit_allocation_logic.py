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

from ..core.mdct import BlockSizeMode # For type hinting and usage

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

    @staticmethod
    def bfu_to_band(bfu_index: int) -> int:
        """Maps a BFU index (0-51) to a QMF band index (0-2)."""
        if bfu_index < 20:  # BFUs 0-19
            return 0  # Low band
        elif bfu_index < 36:  # BFUs 20-35
            return 1  # Mid band
        else:  # BFUs 36-51
            return 2  # High band

    def calc_bits_allocation(
        self,
        scaled_blocks: List["ScaledBlock"], # List of active scaled blocks
        block_size_mode: "BlockSizeMode",
        ath_scaled: List[float], # List of active ATH values, scaled by loudness
        analize_scale_factor_spread: float,
        shift: int,
    ) -> Tuple[List[int], int]:
        """
        Calculates the bit allocation for each Basic Frequency Unit (BFU).

        Args:
            scaled_blocks: List of scaled blocks, one for each active BFU.
            block_size_mode: Object indicating short/long mode for each QMF band.
            ath_scaled: Absolute Threshold of Hearing values, scaled by loudness, for each active BFU.
            analize_scale_factor_spread: Value indicating tonal vs. noise-like characteristics.
            shift: A shift value used in the iterative optimization process to adjust
                   the overall bit allocation.

        Returns:
            A tuple containing:
                - A list of integers representing bits allocated per BFU (word lengths).
                  The list length is num_active_bfus.
                - Total bits used by mantissas with this allocation.
        """
        num_active_bfus = len(scaled_blocks)
        # Initialize for all possible BFUs, then slice at the end if needed,
        # but this function returns for active_bfus length.
        bits_per_bfu: List[int] = [0] * num_active_bfus


        for i in range(num_active_bfus):
            band_for_bfu = Atrac1SimpleBitAlloc.bfu_to_band(i)
            # block_size_mode.LogCount[band] is 0 for long, >0 for short (e.g. 2 or 3)
            is_short_block_for_bfu = bool(block_size_mode.log_count[band_for_bfu])

            fixed_alloc_table_for_bfu = (
                FIXED_BIT_ALLOC_TABLE_SHORT if is_short_block_for_bfu else FIXED_BIT_ALLOC_TABLE_LONG
            )

            if i >= len(fixed_alloc_table_for_bfu): # Should not happen if MAX_BFUS is consistent
                bits_per_bfu[i] = 0
                continue

            if not is_short_block_for_bfu and scaled_blocks[i].max_energy < ath_scaled[i]:
                bits_per_bfu[i] = 0
            else:
                # Use atracdenc-compatible formula:
                # spread * (ScaleFactorIndex/3.2) + (1.0 - spread) * fix - shift
                fixed_part = fixed_alloc_table[i]
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

    def _get_max_used_bfu_id(self, bits_per_each_block: List[int], current_bfu_tab_idx: int) -> int:
        """
        Helper to find the optimal BFU amount table index based on current allocation.
        bits_per_each_block is for current_num_active_bfus.
        current_bfu_tab_idx is 0-7.
        Returns new bfu_tab_idx (0-7).
        """
        # This needs self.codec_data.bfu_amount_tab and self.codec_data.fast_bfu_num_search
        bfu_amount_table = self.codec_data.bfu_amount_tab # e.g., [20, 28, 32, 36, 40, 44, 48, 52]

        new_bfu_tab_idx = current_bfu_tab_idx
        # Number of BFUs currently considered based on current_bfu_tab_idx
        num_bfus_for_current_idx = bfu_amount_table[current_bfu_tab_idx]

        if num_bfus_for_current_idx > len(bits_per_each_block):
            # This can happen if bits_per_each_block was from a smaller bfu_num
            # For safety, or adjust logic based on how C++ handles this.
            # C++ GetMaxUsedBfuId asserts bfuNum == bitsPerEachBlock.size() if idx != 0.
            # This implies bitsPerEachBlock should always be sized for bfuAmountTab[idx].
            # The CalcBitsAllocation in C++ is called with bfuNum, so its result is that size.
            # Python calc_bits_allocation gets len(scaled_blocks) which is num_active_bfus.
            # This means bits_per_each_block here might be shorter than bfu_amount_table[new_bfu_tab_idx].
            # For this helper, assume bits_per_each_block is already correctly sized for num_bfus_for_current_idx.
            pass


        idx_check = new_bfu_tab_idx
        while True:
            num_bfus_at_idx_check = bfu_amount_table[idx_check]

            # Check if the current number of active bfus in bits_per_each_block is less than this table entry.
            # This means we should potentially reduce bfu_tab_idx.
            if num_bfus_at_idx_check > len(bits_per_each_block):
                 if idx_check > 0:
                    idx_check -= 1
                    continue
                 else: # Already at smallest BFU count
                    break


            if idx_check == 0: # At the smallest BFU count, cannot reduce further
                new_bfu_tab_idx = 0
                break

            # Count trailing zeros in the relevant part of bits_per_each_block
            # The part to check is from bfu_amount_table[idx_check-1] to bfu_amount_table[idx_check]-1
            relevant_slice_start = bfu_amount_table[idx_check-1]
            num_to_check_in_slice = bfu_amount_table[idx_check] - bfu_amount_table[idx_check-1]

            trailing_zeros_in_slice = 0
            # Iterate backwards over the slice of bits_per_each_block that corresponds to the
            # difference between bfu_amount_table[idx_check] and bfu_amount_table[idx_check-1]
            for k in range(num_to_check_in_slice):
                bfu_actual_index = bfu_amount_table[idx_check] - 1 - k
                if bfu_actual_index < len(bits_per_each_block) and bits_per_each_block[bfu_actual_index] == 0:
                    trailing_zeros_in_slice += 1
                else:
                    break # Non-zero found, or out of bounds of current allocation

            if trailing_zeros_in_slice >= num_to_check_in_slice:
                # All elements in this top slice are zero, so try reducing bfu_tab_idx
                idx_check -= 1
                new_bfu_tab_idx = idx_check # Tentatively update
            else:
                # Found a non-zero element in this slice, so this idx_check is appropriate
                new_bfu_tab_idx = idx_check
                break

        return new_bfu_tab_idx

    def perform_iterative_allocation(
        self,
        # Arguments required from caller
        scaled_blocks_full: List["ScaledBlock"],  # Full list for MAX_BFUS
        block_size_mode: "BlockSizeMode",
        ath_scaled_full: List[float],  # Full list for MAX_BFUS, loudness applied
        analize_scale_factor_spread: float,
        # Initial settings, potentially from codec/frame settings
        initial_bfu_idx_const: int, # 0 for auto, 1-8 for fixed (maps to table index 0-7)
        fast_bfu_num_search: bool,
        # Overall bit budget for the frame
        frame_total_mantissa_bits_budget: int
    ) -> Tuple[List[int], int, int]: # Returns allocation (MAX_BFUS), actual_mantissa_bits, final_num_active_bfus
        """
        Iteratively finds an optimal shift value and BFU count.
        Aligns with C++ TAtrac1SimpleBitAlloc::Write.
        """

        # Determine initial bfu_tab_idx (0-7)
        # C++: bfuIdx = BfuIdxConst ? BfuIdxConst - 1 : 7; (7 is max index)
        # initial_bfu_idx_const is 1-based in prompt, map to 0-based for table
        current_bfu_tab_idx = (initial_bfu_idx_const - 1) if initial_bfu_idx_const > 0 else 7
        auto_bfu_count_mode = (initial_bfu_idx_const == 0)

        final_allocation_padded = [0] * MAX_BFUS
        final_mantissa_bits_used = 0

        # Outer loop for adjusting BFU count (similar to C++ `for (;;)` with bfuNumChanged)
        while True:
            bfu_num_changed_in_iteration = False
            current_num_active_bfus = self.codec_data.bfu_amount_tab[current_bfu_tab_idx]

            # Slice inputs for the current number of active BFUs
            current_scaled_blocks = scaled_blocks_full[:current_num_active_bfus]
            current_ath_scaled = ath_scaled_full[:current_num_active_bfus]

            # Calculate available bits for mantissas for the current_num_active_bfus
            # This needs to be dynamic based on current_num_active_bfus
            # Assuming header_fixed_overhead_bits = 16 (from previous alignment)
            header_fixed_overhead_bits = 16
            wl_header_bits = current_num_active_bfus * self.codec_data.bits_per_idwl
            sf_header_bits = current_num_active_bfus * self.codec_data.bits_per_idsf

            # frame_total_mantissa_bits_budget is the budget for mantissas only,
            # calculated by the caller by subtracting all headers.
            # So, bits_available_for_mantissas = frame_total_mantissa_bits_budget.
            # No, the C++ TAtrac1SimpleBitAlloc::Write calculates bitsAvaliablePerBfus *inside* the loop,
            # because bitsPerEachBlock.size() (which is current_num_active_bfus) changes.
            bits_available_for_mantissas_current = frame_total_mantissa_bits_budget - \
                (wl_header_bits + sf_header_bits) # This is wrong.
                                                # The budget passed should be total for frame.

            # Let's rename frame_total_mantissa_bits_budget to reflect it's the budget for the whole sound unit.
            # SoundUnitSize * 8 - FixedOverhead - WLOverhead - SFOverhead
            bits_available_for_mantissas_calc = (SOUND_UNIT_SIZE * 8) - header_fixed_overhead_bits - \
                                              wl_header_bits - sf_header_bits


            max_mantissa_bits_allowed = bits_available_for_mantissas_calc
            min_mantissa_bits_target = bits_available_for_mantissas_calc - 110 # From C++

            # Bisection for shift
            min_shift = -3.0  # C++ TAtrac1SimpleBitAlloc::Write
            max_shift = 15.0  # C++ TAtrac1SimpleBitAlloc::Write
            current_shift = 3.0 # C++ TAtrac1SimpleBitAlloc::Write

            # Store the allocation from the current shift search iteration
            # This will be the one that meets criteria or last one if bisection finishes
            current_best_alloc_active = []
            current_best_mantissa_bits = -1

            while max_shift - min_shift >= 0.1:
                # Perform allocation with current_shift
                # calc_bits_allocation returns allocation for current_num_active_bfus
                tmp_alloc_active, tmp_mantissa_bits = self.calc_bits_allocation(
                    current_scaled_blocks,
                    block_size_mode,
                    current_ath_scaled,
                    analize_scale_factor_spread,
                    int(round(current_shift)) # Shift is integer
                )

                current_best_alloc_active = tmp_alloc_active
                current_best_mantissa_bits = tmp_mantissa_bits

                if tmp_mantissa_bits < min_mantissa_bits_target:
                    # Too few bits used, try to use more by decreasing shift
                    max_shift = current_shift
                    current_shift -= (current_shift - min_shift) / 2.0
                elif tmp_mantissa_bits > max_mantissa_bits_allowed:
                    # Too many bits used, try to use less by increasing shift
                    min_shift = current_shift
                    current_shift += (max_shift - current_shift) / 2.0
                else:
                    # Target range met
                    break # Shift search successful for this BFU count

            # After bisection loop, current_best_alloc_active and current_best_mantissa_bits hold the result
            # Now, check if BFU count needs to be adjusted (if in auto mode)
            if auto_bfu_count_mode:
                # current_best_alloc_active is sized for current_num_active_bfus
                new_bfu_tab_idx = self._get_max_used_bfu_id(current_best_alloc_active, current_bfu_tab_idx)

                if new_bfu_tab_idx < current_bfu_tab_idx:
                    bfu_num_changed_in_iteration = True
                    current_bfu_tab_idx = new_bfu_tab_idx if fast_bfu_num_search else (current_bfu_tab_idx - 1)
                    # Continue outer loop with new current_bfu_tab_idx

            if not bfu_num_changed_in_iteration:
                # If BFU count didn't change (or not in auto mode), this is our final allocation
                final_allocation_padded = [0] * MAX_BFUS
                final_allocation_padded[:current_num_active_bfus] = current_best_alloc_active
                final_mantissa_bits_used = current_best_mantissa_bits
                break # Exit outer BFU adjustment loop
            # If bfu_num_changed_in_iteration is true, outer loop continues

        # Return final padded allocation, bits used, and the actual number of active BFUs determined
        final_num_active_bfus = self.codec_data.bfu_amount_tab[current_bfu_tab_idx]
        # Ensure final_mantissa_bits_used is non-negative, default to 0 if it's -1 (e.g. no valid alloc found)
        return final_allocation_padded, max(0, final_mantissa_bits_used), final_num_active_bfus


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

        # C++ TBitsBooster::ApplyBoost iterates based on a pre-sorted BitsBoostMap
        # (cost -> bfu_index) and has a MinKey check.
        # The loop continues while surplus >= MinKey and iterations make changes.
        # This Python version will build a list of potential boosts and iterate.

        min_possible_cost = float('inf')
        for i in range(num_active_bfus):
             if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] < 16:
                cost_per_spec = self.codec_data.specs_per_block[i]
                if cost_per_spec > 0: # Ensure valid cost
                    n_bits_to_add = 2 if boosted_bits_per_bfu[i] == 0 else 1
                    actual_cost = n_bits_to_add * cost_per_spec
                    min_possible_cost = min(min_possible_cost, actual_cost)

        if min_possible_cost == float('inf'): # No boostable BFUs
            return boosted_bits_per_bfu, bits_consumed_by_boost

        while surplus_bits >= min_possible_cost:
            boost_occurred_in_pass = False

            # Create a list of current boost opportunities, sorted by cost.
            # Each item: (cost_of_increment, bfu_index, num_bits_to_increment_by)
            # This mimics iterating through the C++ BitsBoostMap implicitly.
            potential_boosts = []
            for i in range(num_active_bfus):
                if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] < 16:
                    cost_per_spec = self.codec_data.specs_per_block[i]
                    if cost_per_spec == 0: continue # Cannot boost if specs_per_block is 0

                    n_bits_to_add = 2 if boosted_bits_per_bfu[i] == 0 else 1

                    # Ensure that adding these bits does not exceed 16
                    if boosted_bits_per_bfu[i] + n_bits_to_add > 16:
                        # If adding 2 bits exceeds, try adding 1 bit if current is 0.
                        # (This case is unlikely if WL cannot exceed 16, but good for safety)
                        if n_bits_to_add == 2 and boosted_bits_per_bfu[i] == 0 and \
                           boosted_bits_per_bfu[i] + 1 <= 16:
                           n_bits_to_add = 1
                        else:
                            continue # Cannot boost this BFU further

                    actual_cost = n_bits_to_add * cost_per_spec
                    potential_boosts.append((actual_cost, i, n_bits_to_add))

            if not potential_boosts: break # No more BFUs can be boosted

            # Sort by cost primarily, then by BFU index as a tie-breaker (optional, but good for consistency)
            potential_boosts.sort(key=lambda x: (x[0], x[1]))

            # Iterate through sorted potential boosts for this pass
            # C++ iterates multiple times if surplus allows (the `it != maxIt` part implies considering all affordable ones)
            # This Python loop will try to apply boosts one by one, cheapest first in this pass.
            for cost, bfu_idx, n_add in potential_boosts:
                if cost <= surplus_bits:
                    # Check again as a previous boost in the same pass might have changed eligibility
                    if boosted_bits_per_bfu[bfu_idx] < 16: # Still possible to boost
                        # Re-check n_add based on current boosted_bits_per_bfu[bfu_idx]
                        # as it might have been incremented by 1 in a previous step of this same pass
                        # if multiple entries for same bfu_idx were possible (not with current list build)
                        current_val = boosted_bits_per_bfu[bfu_idx]
                        n_bits_actually_added = n_add
                        if current_val == 0 and n_add == 2 and current_val + 2 > 16: # e.g. if max was 1
                            n_bits_actually_added = 1
                        elif current_val > 0 and n_add == 2: # Should not happen if n_add is set to 1 for current_val > 0
                             n_bits_actually_added = 1

                        if current_val + n_bits_actually_added > 16: # Final check
                            continue

                        # Re-calculate cost based on n_bits_actually_added
                        actual_cost_for_this_bfu = n_bits_actually_added * self.codec_data.specs_per_block[bfu_idx]

                        if actual_cost_for_this_bfu <= surplus_bits: # Check cost again
                            boosted_bits_per_bfu[bfu_idx] += n_bits_actually_added
                            surplus_bits -= actual_cost_for_this_bfu
                            bits_consumed_by_boost += actual_cost_for_this_bfu
                            boost_occurred_in_pass = True
                            # Do not break here, allow other BFUs to be boosted in the same pass if they are also cheap
                # else:
                    # This candidate (and subsequent ones due to sort) is too expensive for remaining surplus
                    # break # from this inner loop over potential_boosts for the current pass

            if not boost_occurred_in_pass:
                break # Exit outer while loop if no boost was made in a full pass

            # Update min_possible_cost for the next iteration of the while loop
            min_possible_cost = float('inf')
            for i in range(num_active_bfus):
                if i < MAX_BFUS and BIT_BOOST_MASK[i] == 1 and boosted_bits_per_bfu[i] < 16:
                    cost_per_spec = self.codec_data.specs_per_block[i]
                    if cost_per_spec > 0:
                        n_bits_to_add = 2 if boosted_bits_per_bfu[i] == 0 else 1
                        if boosted_bits_per_bfu[i] + n_bits_to_add > 16 : # e.g. at 15, try to add 2
                             if boosted_bits_per_bfu[i] == 15 and n_bits_to_add == 2: n_bits_to_add = 1
                             else: continue # Cannot add even 1 bit if already 16 or n_add makes it >16

                        actual_cost = n_bits_to_add * cost_per_spec
                        min_possible_cost = min(min_possible_cost, actual_cost)
            if min_possible_cost == float('inf'): break


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

    # Using constants from pyatrac1.common.constants
    # For the example, assume they are part of mock_data_instance or defined globally
    mock_data_instance.bits_per_idwl = 4
    mock_data_instance.bits_per_idsf = 6
    mock_data_instance.bfu_amount_tab = [20, 28, 32, 36, 40, 44, 48, 52]


    # Frame budget calculation now happens inside perform_iterative_allocation
    # We need to pass the total frame budget for mantissas, or rather, the overall frame budget
    # Let's assume SOUND_UNIT_SIZE * 8 is the total budget for everything.
    # The perform_iterative_allocation will then derive mantissa budget based on current_num_active_bfus.
    # The C++ TAtrac1SimpleBitAlloc::Write takes scaledBlocks, blockSize, loudness.
    # It calculates spread. It has BfuIdxConst and FastBfuNumSearch as members.
    # Python equivalent would be:
    # initial_bfu_idx_param = 0 # Auto
    # fast_search_param = False

    # The `bits_available_for_mantissas` argument in the original Python `perform_iterative_allocation`
    # was the budget *after* all headers for a *fixed* num_active_bfus.
    # The new version recalculates this internally based on a potentially changing num_active_bfus.
    # So, we need to pass the total frame bit budget (SOUND_UNIT_SIZE * 8) or equivalent.
    # Let's keep the example simple and assume the outer structure calls this.
    # The original call was:
    # allocator_instance.perform_iterative_allocation(
    #   scaled_blocks[:num_active], block_size_mode, ath_scaled[:num_active], spread, num_active, budget_for_mantissas_at_num_active)
    # New call:
    # allocator_instance.perform_iterative_allocation(
    #   scaled_blocks_full, block_size_mode, ath_scaled_full, spread, initial_bfu_idx_const, fast_bfu_num_search, total_frame_budget)

    # For the example, let's simulate the inputs perform_iterative_allocation now expects
    initial_bfu_idx_const_param = 0 # 0 for auto, corresponds to C++ default of 7 (max BFUs)
    fast_bfu_num_search_param = False

    # The budget passed to perform_iterative_allocation is total bits for the sound unit.
    # The function itself will subtract header bits based on the *current* bfu count in its loop.
    total_frame_bits_budget_param = SOUND_UNIT_SIZE * 8

    word_lengths_full, total_mantissa_bits_used_val, final_num_active_bfus_val = (
        allocator_instance.perform_iterative_allocation(
            example_scaled_blocks_list, # Full list
            block_size_mode=example_block_size_mode,
            ath_scaled=example_ath_scaled_list, # Full list
            analize_scale_factor_spread=0.5,
            initial_bfu_idx_const=initial_bfu_idx_const_param,
            fast_bfu_num_search=fast_bfu_num_search_param,
            frame_total_mantissa_bits_budget=total_frame_bits_budget_param # This name is confusing, it's total frame budget
        )
    )
    # Create a dummy BlockSizeMode for the example
    example_block_size_mode = BlockSizeMode(low_band_short=False, mid_band_short=False, high_band_short=False) # All long

    word_lengths_full, total_mantissa_bits_used_val = (
        allocator_instance.perform_iterative_allocation(
            example_scaled_blocks_list[:example_num_bfus_active],
            block_size_mode=example_block_size_mode, # Use BlockSizeMode
            ath_scaled=example_ath_scaled_list[:example_num_bfus_active],
            analize_scale_factor_spread=0.5,
            num_active_bfus=example_num_bfus_active,
            bits_available_for_mantissas=available_mantissa_bits_calc,
        )
    )
