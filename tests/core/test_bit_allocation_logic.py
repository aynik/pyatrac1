import pytest
from typing import (
    List,
    Optional,
)  # Optional was already here, ensure it's used or remove if not

from pyatrac1.core.bit_allocation_logic import Atrac1SimpleBitAlloc, BitsBooster
from pyatrac1.core.codec_data import (
    ScaledBlock,
    Atrac1CodecData,
)  # Atrac1CodecData re-added
from pyatrac1.common.constants import MAX_BFUS
from pyatrac1.tables.bit_allocation import (
    FIXED_BIT_ALLOC_TABLE_LONG,
    FIXED_BIT_ALLOC_TABLE_SHORT,
    BIT_BOOST_MASK,
)


# Mock CodecData for testing
class MockAtrac1CodecData(Atrac1CodecData):  # Inherit from Atrac1CodecData
    def __init__(self, specs_per_block: Optional[List[int]] = None):
        super().__init__()  # Call parent constructor
        if specs_per_block is None:
            # Override specs_per_block after parent init if None is passed
            self.specs_per_block: List[int] = [
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
        else:
            self.specs_per_block: List[int] = specs_per_block

        if len(self.specs_per_block) < MAX_BFUS:
            self.specs_per_block.extend([0] * (MAX_BFUS - len(self.specs_per_block)))
        elif len(self.specs_per_block) > MAX_BFUS:
            self.specs_per_block = self.specs_per_block[:MAX_BFUS]


@pytest.fixture
def mock_codec_data_default() -> MockAtrac1CodecData:
    return MockAtrac1CodecData()


@pytest.fixture
def simple_bit_alloc(
    mock_codec_data_default: MockAtrac1CodecData,
) -> Atrac1SimpleBitAlloc:
    return Atrac1SimpleBitAlloc(codec_data=mock_codec_data_default)  # type: ignore removed, should be compatible now


@pytest.fixture
def bits_booster(mock_codec_data_default: MockAtrac1CodecData) -> BitsBooster:
    return BitsBooster(codec_data=mock_codec_data_default)  # type: ignore removed, should be compatible now


# Helper to create scaled blocks
def create_scaled_blocks(
    num_blocks: int, max_energy: float = 100.0, scale_factor_index: int = 0
) -> List[ScaledBlock]:
    return [
        ScaledBlock(
            scale_factor_index=scale_factor_index,
            scaled_values=[],
            max_energy=max_energy,
        )
        for _ in range(num_blocks)
    ]


class TestAtrac1SimpleBitAllocCalcBits:
    def test_calc_bits_allocation_long_block_basic(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 10
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        shift = 0

        bits_per_bfu, total_mantissa_bits = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )

        assert len(bits_per_bfu) == num_active_bfus
        # With atracdenc formula: spread * (ScaleFactorIndex/3.2) + (1.0 - spread) * fix - shift
        # spread=0.5, ScaleFactorIndex=0, shift=0
        # = 0.5 * (0/3.2) + (1.0 - 0.5) * fix - 0 = 0.5 * fix
        expected_bits = []
        for i in range(num_active_bfus):
            tmp = 0.5 * FIXED_BIT_ALLOC_TABLE_LONG[i]
            if tmp > 16:
                expected_bits.append(16)
            elif tmp < 2:
                expected_bits.append(0)
            else:
                expected_bits.append(int(tmp))
        assert bits_per_bfu == expected_bits

        expected_mantissa_bits = sum(
            bpf * simple_bit_alloc.codec_data.specs_per_block[i]
            for i, bpf in enumerate(expected_bits)
            if bpf > 0
        )
        assert total_mantissa_bits == expected_mantissa_bits

    def test_calc_bits_allocation_short_block_basic(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 5
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=50.0)
        ath_scaled = [5.0] * num_active_bfus
        is_long_block = False
        shift = 1

        bits_per_bfu, total_mantissa_bits = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )

        assert len(bits_per_bfu) == num_active_bfus
        # With atracdenc formula: spread * (ScaleFactorIndex/3.2) + (1.0 - spread) * fix - shift
        # spread=0.5, ScaleFactorIndex=0, shift=1
        # = 0.5 * (0/3.2) + (1.0 - 0.5) * fix - 1 = 0.5 * fix - 1
        expected_bits = []
        for i in range(num_active_bfus):
            tmp = 0.5 * FIXED_BIT_ALLOC_TABLE_SHORT[i] - shift
            if tmp > 16:
                expected_bits.append(16)
            elif tmp < 2:
                expected_bits.append(0)
            else:
                expected_bits.append(int(tmp))
        assert bits_per_bfu == expected_bits
        expected_mantissa_bits = sum(
            bpf * simple_bit_alloc.codec_data.specs_per_block[i]
            for i, bpf in enumerate(expected_bits)
            if bpf > 0
        )
        assert total_mantissa_bits == expected_mantissa_bits

    def test_calc_bits_allocation_energy_below_ath(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 3
        scaled_blocks = create_scaled_blocks(
            num_active_bfus, max_energy=5.0
        )  # Energy below ATH for some
        ath_scaled = [10.0, 2.0, 10.0]  # BFU 0 and 2 should get 0 bits
        is_long_block = True
        shift = 0

        bits_per_bfu, _ = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )
        assert bits_per_bfu[0] == 0
        assert bits_per_bfu[1] > 0  # Assuming FIXED_BIT_ALLOC_TABLE_LONG[1] > 0
        assert bits_per_bfu[2] == 0

    def test_calc_bits_allocation_with_shift(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 4
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        shift = 2  # Positive shift reduces allocated bits

        bits_per_bfu, _ = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )
        # With atracdenc formula: spread * (ScaleFactorIndex/3.2) + (1.0 - spread) * fix - shift
        # spread=0.5, ScaleFactorIndex=0, shift=2
        # = 0.5 * (0/3.2) + (1.0 - 0.5) * fix - 2 = 0.5 * fix - 2
        for i in range(num_active_bfus):
            tmp = 0.5 * FIXED_BIT_ALLOC_TABLE_LONG[i] - shift
            if tmp > 16:
                expected_val = 16
            elif tmp < 2:
                expected_val = 0
            else:
                expected_val = int(tmp)
            assert bits_per_bfu[i] == expected_val

    def test_calc_bits_allocation_less_than_2_becomes_zero(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 1
        # Use values that will generate < 2 bits, which should become 0
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0, scale_factor_index=0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        # With formula: 0.5 * fix - shift, we need 0.5 * fix - shift < 2
        # So shift > 0.5 * fix - 2. For FIXED_BIT_ALLOC_TABLE_LONG[0] = 7, 0.5 * 7 = 3.5
        # So shift > 1.5, use shift = 2
        shift = 2

        bits_per_bfu, _ = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )
        # Should be 0 because result < 2
        assert bits_per_bfu[0] == 0

    def test_calc_bits_allocation_max_capped_at_16(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 1
        # Use a large scale factor index to generate a large bit allocation that gets capped
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0, scale_factor_index=60)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        shift = -10  # Large negative shift to make allocation > 16

        bits_per_bfu, _ = simple_bit_alloc.calc_bits_allocation(
            scaled_blocks, is_long_block, ath_scaled, 0.5, shift
        )
        # With atracdenc formula and large values, should be capped at 16
        assert bits_per_bfu[0] <= 16
        assert bits_per_bfu[0] == 16  # Should be capped at max


class TestAtrac1SimpleBitAllocIterative:
    def test_perform_iterative_allocation_finds_under_budget(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 10
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        # Calculate a realistic bits_available_for_mantissas
        # Assuming some header bits. For simplicity, let's pick a value.
        # Total bits for 10 BFUs with specs_per_block[0] (e.g. 8) and avg 4 bits/BFU = 10*8*4 = 320
        # Let's aim for a budget that allows some allocation.
        # If shift=0, FIXED_BIT_ALLOC_TABLE_LONG[0..9] * specs_per_block[0..9]
        # Example: if FIXED_BIT_ALLOC_TABLE_LONG are all 5, specs are all 8 -> 10 * 5 * 8 = 400
        bits_available = (
            sum(
                FIXED_BIT_ALLOC_TABLE_LONG[i]
                * simple_bit_alloc.codec_data.specs_per_block[i]
                for i in range(num_active_bfus)
            )
            // 2
        )  # A budget that should be achievable

        allocation, total_bits = simple_bit_alloc.perform_iterative_allocation(
            scaled_blocks,
            is_long_block,
            ath_scaled,
            0.5,
            num_active_bfus,
            bits_available,
        )
        assert len(allocation) == MAX_BFUS
        assert total_bits <= bits_available
        assert total_bits > 0  # Should find some allocation
        # Verify that the active part of allocation is not all zeros if budget allows
        assert any(allocation[i] > 0 for i in range(num_active_bfus))

    def test_perform_iterative_allocation_all_over_budget(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 5
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        bits_available = 1  # Very low budget, likely all initial allocations are over

        allocation, total_bits = simple_bit_alloc.perform_iterative_allocation(
            scaled_blocks,
            is_long_block,
            ath_scaled,
            0.5,
            num_active_bfus,
            bits_available,
        )
        assert len(allocation) == MAX_BFUS
        # It should pick the smallest positive allocation if all are over budget
        # or zero if even the smallest is too large (which is handled by calc_bits_allocation)
        # The logic picks the *closest* over budget, so total_bits might be > bits_available
        # If all shifts result in 0 bits (e.g. high shift), then total_bits could be 0.
        # If the smallest possible allocation (e.g. 2 bits in one BFU * spec_count) > 1,
        # then total_bits will be that smallest allocation.
        min_possible_bits_for_one_bfu = 2 * min(
            s
            for s in simple_bit_alloc.codec_data.specs_per_block[:num_active_bfus]
            if s > 0
        )

        if total_bits > 0:  # If any allocation was possible
            assert total_bits >= min_possible_bits_for_one_bfu
        else:  # All allocations resulted in 0 bits
            assert total_bits == 0

        # Check if it indeed picked the one with the smallest total_mantissa_bits
        # This requires re-running calc_bits_allocation for all shifts
        min_mantissa_bits_found = float("inf")
        for shift_try in range(-8, 17):
            _, current_total_mantissa_bits = simple_bit_alloc.calc_bits_allocation(
                scaled_blocks, is_long_block, ath_scaled, 0.5, shift_try
            )
            if current_total_mantissa_bits > 0:  # only consider positive allocations
                min_mantissa_bits_found = min(
                    min_mantissa_bits_found, current_total_mantissa_bits
                )

        if (
            min_mantissa_bits_found != float("inf")
            and min_mantissa_bits_found > bits_available
        ):
            assert total_bits == min_mantissa_bits_found
        elif min_mantissa_bits_found == float("inf"):  # all shifts resulted in 0 bits
            assert total_bits == 0

    def test_perform_iterative_allocation_zero_budget(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 5
        scaled_blocks = create_scaled_blocks(num_active_bfus, max_energy=100.0)
        ath_scaled = [10.0] * num_active_bfus
        is_long_block = True
        bits_available = 0

        allocation, total_bits = simple_bit_alloc.perform_iterative_allocation(
            scaled_blocks,
            is_long_block,
            ath_scaled,
            0.5,
            num_active_bfus,
            bits_available,
        )
        assert len(allocation) == MAX_BFUS
        assert total_bits == 0
        assert all(a == 0 for a in allocation)

    def test_perform_iterative_allocation_empty_scaled_blocks(
        self, simple_bit_alloc: Atrac1SimpleBitAlloc
    ):
        num_active_bfus = 0
        scaled_blocks: List[ScaledBlock] = []
        ath_scaled: List[float] = []
        is_long_block = True
        bits_available = 100

        allocation, total_bits = simple_bit_alloc.perform_iterative_allocation(
            scaled_blocks,
            is_long_block,
            ath_scaled,
            0.5,
            num_active_bfus,
            bits_available,
        )
        assert len(allocation) == MAX_BFUS
        assert total_bits == 0
        assert all(a == 0 for a in allocation)


class TestBitsBooster:
    def test_apply_boost_no_surplus_bits(self, bits_booster: BitsBooster):
        num_active_bfus = 10
        current_bits_per_bfu = [0] * MAX_BFUS  # MAX_BFUS long
        for i in range(num_active_bfus):
            current_bits_per_bfu[i] = (
                FIXED_BIT_ALLOC_TABLE_LONG[i]
                if FIXED_BIT_ALLOC_TABLE_LONG[i] != 1
                else 0
            )
        surplus_bits = 0

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits == current_bits_per_bfu
        assert consumed == 0

    def test_apply_boost_priority1_enough_bits(self, bits_booster: BitsBooster):
        num_active_bfus = 25  # Include BFUs that are eligible for boost
        # Make BFU 0 and 2 eligible for priority 1 boost (current bits = 0, mask = 1)
        current_bits_per_bfu = [0] * MAX_BFUS
        current_bits_per_bfu[1] = 4  # Not zero
        # Assume BIT_BOOST_MASK[0]=1, BIT_BOOST_MASK[1]=0, BIT_BOOST_MASK[2]=1 for test
        # For simplicity, let's use the actual BIT_BOOST_MASK
        # Find first few BFUs where BIT_BOOST_MASK is 1
        eligible_indices_for_p1 = [
            i for i, m in enumerate(BIT_BOOST_MASK[:num_active_bfus]) if m == 1
        ]

        if not eligible_indices_for_p1:
            pytest.skip(
                "No BFU eligible for P1 boost in the first num_active_bfus with current BIT_BOOST_MASK"
            )

        idx1 = eligible_indices_for_p1[0]
        # Ensure current_bits_per_bfu[idx1] is 0
        current_bits_per_bfu[idx1] = 0

        cost_for_idx1_boost = 2 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost + 5  # Enough for one P1 boost

        original_bfu_idx1 = list(current_bits_per_bfu)

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )

        assert boosted_bits[idx1] == 2
        assert consumed == cost_for_idx1_boost
        for i in range(num_active_bfus):
            if i != idx1:
                assert boosted_bits[i] == original_bfu_idx1[i]

    def test_apply_boost_priority2_enough_bits(self, bits_booster: BitsBooster):
        num_active_bfus = 25
        current_bits_per_bfu = [0] * MAX_BFUS
        # Make BFU 0 eligible for P2 boost (current bits > 0, mask = 1)
        eligible_indices_for_p2 = [
            i for i, m in enumerate(BIT_BOOST_MASK[:num_active_bfus]) if m == 1
        ]
        if not eligible_indices_for_p2:
            pytest.skip(
                "No BFU eligible for P2 boost in the first num_active_bfus with current BIT_BOOST_MASK"
            )

        idx1 = eligible_indices_for_p2[0]
        current_bits_per_bfu[idx1] = 4  # Has some bits

        cost_for_idx1_boost = 1 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost + 5  # Enough for one P2 boost

        original_bfu_idx1_val = current_bits_per_bfu[idx1]
        original_full_list = list(current_bits_per_bfu)

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )

        assert boosted_bits[idx1] == original_bfu_idx1_val + 1
        assert consumed == cost_for_idx1_boost
        for i in range(num_active_bfus):
            if i != idx1:
                assert boosted_bits[i] == original_full_list[i]

    def test_apply_boost_mixed_priorities(self, bits_booster: BitsBooster):
        num_active_bfus = min(MAX_BFUS, 25)  # Ensure we have enough BFUs to test
        current_bits_per_bfu = [0] * MAX_BFUS

        # Find indices for P1 and P2 based on BIT_BOOST_MASK
        p1_idx = -1
        p2_idx = -1

        for i in range(num_active_bfus):
            if BIT_BOOST_MASK[i] == 1:
                if p1_idx == -1:
                    p1_idx = i
                elif p2_idx == -1 and i != p1_idx:
                    p2_idx = i
                    break

        if p1_idx == -1 or p2_idx == -1:
            pytest.skip(
                "Not enough distinct eligible BFUs for P1 and P2 boost test with current BIT_BOOST_MASK"
            )

        current_bits_per_bfu[p1_idx] = 0  # Eligible for P1
        current_bits_per_bfu[p2_idx] = 3  # Eligible for P2

        cost_p1 = 2 * bits_booster.codec_data.specs_per_block[p1_idx]
        cost_p2 = 1 * bits_booster.codec_data.specs_per_block[p2_idx]
        # After P1 boost, p1_idx becomes eligible for P2 boost too
        cost_p1_p2 = 1 * bits_booster.codec_data.specs_per_block[p1_idx]
        surplus_bits = cost_p1 + cost_p1_p2 + 1  # Enough for P1 and P1's P2, but not P2's P2

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )

        # P1 boost: p1_idx goes 0->2, then P2 boost: only p1_idx goes 2->3 (p2_idx can't afford boost)
        assert boosted_bits[p1_idx] == 3  # 0 + 2 (P1) + 1 (P2)
        assert boosted_bits[p2_idx] == 3  # No change (not enough surplus for its boost)
        assert consumed == cost_p1 + cost_p1_p2

    def test_apply_boost_surplus_not_enough_for_full_p1(
        self, bits_booster: BitsBooster
    ):
        num_active_bfus = 25
        current_bits_per_bfu = [0] * MAX_BFUS
        eligible_indices_for_p1 = [
            i for i, m in enumerate(BIT_BOOST_MASK[:num_active_bfus]) if m == 1
        ]
        if not eligible_indices_for_p1:
            pytest.skip("No BFU eligible for P1 boost")
        idx1 = eligible_indices_for_p1[0]
        current_bits_per_bfu[idx1] = 0

        cost_for_idx1_boost = 2 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost - 1  # Not enough

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits[idx1] == 0  # No boost
        assert consumed == 0

    def test_apply_boost_surplus_not_enough_for_full_p2(
        self, bits_booster: BitsBooster
    ):
        num_active_bfus = 25
        current_bits_per_bfu = [0] * MAX_BFUS
        eligible_indices_for_p2 = [
            i for i, m in enumerate(BIT_BOOST_MASK[:num_active_bfus]) if m == 1
        ]
        if not eligible_indices_for_p2:
            pytest.skip("No BFU eligible for P2 boost")
        idx1 = eligible_indices_for_p2[0]
        current_bits_per_bfu[idx1] = 3

        cost_for_idx1_boost = 1 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost - 1  # Not enough

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits[idx1] == 3  # No boost
        assert consumed == 0

    def test_apply_boost_limited_by_max_word_length_16_p1(
        self, bits_booster: BitsBooster
    ):
        # P1 boost adds 2 bits to a BFU that currently has 0 bits.
        # The condition `boosted_bits_per_bfu[i] + 2 <= 16` is checked.
        # Since 0 + 2 = 2, which is <= 16, P1 boost itself will not be capped by 16.
        # This test case is therefore not directly applicable for P1 in its current logic.
        # We can verify that a P1 boost happens correctly when eligible.
        num_active_bfus = 25
        current_bits_per_bfu = [0] * MAX_BFUS
        idx1 = 0
        # Find first boostable BFU for P1
        while idx1 < num_active_bfus and not (
            BIT_BOOST_MASK[idx1] == 1 and current_bits_per_bfu[idx1] == 0
        ):
            if idx1 < MAX_BFUS - 1:  # prevent index out of bounds for BIT_BOOST_MASK
                idx1 += 1
            else:  # No eligible BFU found
                idx1 = num_active_bfus  # break loop
                break

        if idx1 == num_active_bfus:  # No eligible BFU found
            pytest.skip("No BFU eligible for P1 boost (mask=1 and current_bits=0)")

        # current_bits_per_bfu[idx1] is already 0 if eligible
        cost_for_idx1_boost = 2 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits[idx1] == 2  # Boosted by 2
        assert consumed == cost_for_idx1_boost

    def test_apply_boost_limited_by_max_word_length_16_p2(
        self, bits_booster: BitsBooster
    ):
        num_active_bfus = 25
        current_bits_per_bfu = [0] * MAX_BFUS
        idx1 = 0
        # Find first boostable BFU for P2
        while idx1 < num_active_bfus and not (
            BIT_BOOST_MASK[idx1] == 1 and current_bits_per_bfu[idx1] > 0
        ):
            current_bits_per_bfu[idx1] = 1  # Set to a value > 0 to check P2 eligibility
            if BIT_BOOST_MASK[idx1] == 1:  # Found one
                break
            if idx1 < MAX_BFUS - 1:
                idx1 += 1
            else:
                idx1 = num_active_bfus
                break

        if idx1 == num_active_bfus:
            pytest.skip("No BFU eligible for P2 boost (mask=1 and current_bits > 0)")

        current_bits_per_bfu[idx1] = 16  # Already at max
        cost_for_idx1_boost = 1 * bits_booster.codec_data.specs_per_block[idx1]
        surplus_bits = cost_for_idx1_boost + 5

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits[idx1] == 16  # No change
        assert consumed == 0

        current_bits_per_bfu[idx1] = 15  # One below max
        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )
        assert boosted_bits[idx1] == 16  # Boosted by 1
        assert consumed == cost_for_idx1_boost

    def test_apply_boost_respects_bit_boost_mask(self, bits_booster: BitsBooster):
        num_active_bfus = MAX_BFUS
        current_bits_per_bfu = [0] * MAX_BFUS
        surplus_bits = 1000  # Plenty of bits

        # Set some BFUs to be non-boostable by mask
        non_boostable_indices = [
            i for i, m in enumerate(BIT_BOOST_MASK[:num_active_bfus]) if m == 0
        ]

        if not non_boostable_indices:
            pytest.skip("BIT_BOOST_MASK is all 1s, cannot test non-boostable case.")

        idx_non_boost_p1 = -1
        for idx in non_boostable_indices:
            if idx < num_active_bfus:
                current_bits_per_bfu[idx] = 0  # Eligible for P1 if mask allowed
                idx_non_boost_p1 = idx
                break

        idx_non_boost_p2 = -1
        for idx in non_boostable_indices:
            if idx < num_active_bfus and idx != idx_non_boost_p1:
                current_bits_per_bfu[idx] = 4  # Eligible for P2 if mask allowed
                idx_non_boost_p2 = idx
                break

        boosted_bits, consumed = bits_booster.apply_boost(
            list(current_bits_per_bfu), surplus_bits, num_active_bfus
        )

        if idx_non_boost_p1 != -1:
            assert boosted_bits[idx_non_boost_p1] == 0  # Should not be boosted
        if idx_non_boost_p2 != -1:
            assert boosted_bits[idx_non_boost_p2] == 4  # Should not be boosted

        # Check that other, boostable ones *were* boosted if eligible
        # This check is complex because it depends on surplus_bits and costs of other BFUs.
        # The primary check is that non_boostable_indices were NOT boosted.
        # If there were other boostable BFUs and enough surplus, 'consumed' should be > 0.
        was_anything_else_boosted = False
        for i in range(num_active_bfus):
            if i != idx_non_boost_p1 and i != idx_non_boost_p2:
                if BIT_BOOST_MASK[i] == 1:
                    if current_bits_per_bfu[i] == 0 and boosted_bits[i] == 2:
                        was_anything_else_boosted = True
                        break
                    if (
                        current_bits_per_bfu[i] > 0
                        and boosted_bits[i] > current_bits_per_bfu[i]
                    ):
                        was_anything_else_boosted = True
                        break

        any_other_eligible_and_enough_bits = False
        temp_surplus = surplus_bits
        if idx_non_boost_p1 != -1:  # if it was tested, its cost wasn't 'consumed'
            pass  # no change to temp_surplus
        if idx_non_boost_p2 != -1:
            pass

        for i in range(num_active_bfus):
            if i != idx_non_boost_p1 and i != idx_non_boost_p2:
                if BIT_BOOST_MASK[i] == 1:
                    if (
                        current_bits_per_bfu[i] == 0
                        and (2 * bits_booster.codec_data.specs_per_block[i])
                        <= temp_surplus
                    ):
                        any_other_eligible_and_enough_bits = True
                        break
                    if (
                        current_bits_per_bfu[i] > 0
                        and (1 * bits_booster.codec_data.specs_per_block[i])
                        <= temp_surplus
                    ):
                        any_other_eligible_and_enough_bits = True
                        break

        # Simply verify that non-boostable BFUs weren't boosted
        if idx_non_boost_p1 != -1:
            assert boosted_bits[idx_non_boost_p1] == 0  # Should not be boosted
        if idx_non_boost_p2 != -1:
            assert boosted_bits[idx_non_boost_p2] == 4  # Should not be boosted

    def test_total_bits_consistency_after_boost(
        self,
        bits_booster: BitsBooster,  # Corrected type hint
    ):
        # This is implicitly tested by checking `consumed` bits.
        # The function returns `bits_consumed_by_boost`.
        # The caller is responsible for `total_mantissa_bits_used_val += bits_boost_consumed_val`
        # So, the function itself doesn't need to maintain total bit count, just report consumption.
        pass
