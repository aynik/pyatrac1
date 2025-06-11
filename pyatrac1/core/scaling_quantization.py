"""
Implements scaling, quantization, and signed bitstream value handling for ATRAC1.
Based on spec.txt sections 3.6 and "Bitstream Handling (Signed Values)".
"""

from typing import List, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .codec_data import Atrac1CodecData
    
from .codec_data import ScaledBlock


class TScaler:
    """
    Handles the scaling of spectral coefficients using the ATRAC1 ScaleTable.
    """

    def __init__(self, codec_data: "Atrac1CodecData"):
        self.codec_data = codec_data

    def scale(self, spectral_coeffs: List[float]) -> "ScaledBlock":
        """
        Normalizes a block of spectral coefficients.

        Args:
            spectral_coeffs: A list of floating-point spectral coefficients for one BFU.

        Returns:
            A ScaledBlock object containing the scale factor index,
            the normalized scaled values, and the max energy of the original block.
        """
        if not spectral_coeffs:
            return ScaledBlock(scale_factor_index=0, scaled_values=[], max_energy=0.0)

        max_abs_spec = 0.0
        max_energy = 0.0
        for coeff in spectral_coeffs:
            abs_coeff = abs(coeff)
            if abs_coeff > max_abs_spec:
                max_abs_spec = abs_coeff
            # Store max(coeff*coeff) to match atracdenc
            energy = coeff * coeff
            if energy > max_energy:
                max_energy = energy

        if max_abs_spec == 0.0:  # All coeffs are zero
            return ScaledBlock(
                scale_factor_index=0,
                scaled_values=[0.0] * len(spectral_coeffs),
                max_energy=0.0,
            )

        # Align with C++: clip max_abs_spec to MAX_SCALE (1.0) before finding scale factor
        if max_abs_spec > 1.0:
            # This matches C++ TAtrac1Data::MAX_SCALE behavior
            # print(f"Scale warning: max_abs_spec ({max_abs_spec}) > 1.0, clipping to 1.0") # Optional: for debugging
            max_abs_spec = 1.0

        chosen_scale_factor_index = 0
        # Default to the largest scale factor if max_abs_spec is smaller than all table entries
        # or if it's 0. However, ScaleTable[0] is usually the smallest non-zero.
        # C++ uses lower_bound, which finds the first element not less than max_abs_spec.
        # If max_abs_spec is 0, it will find the first element in the map (smallest scale factor).
        # If max_abs_spec is > largest value, it would be end().

        # Find the smallest factor in scale_table >= max_abs_spec
        # The scale_table values are sorted and increasing.
        # ScaleTable[0] is the smallest, ScaleTable[63] is the largest (1.0 for Atrac1)

        # If max_abs_spec is 0.0, C++ lower_bound would give ScaleTable[0]
        # If max_abs_spec is > 0, find appropriate factor
        if max_abs_spec == 0.0:
             # All coeffs are zero, already handled, but if not, this would be the path
            chosen_scale_factor_index = 0 # Or an index corresponding to smallest factor
        else:
            # Find first scale_factor >= max_abs_spec
            found_factor = False
            for i, factor_val in enumerate(self.codec_data.scale_table):
                if factor_val >= max_abs_spec:
                    chosen_scale_factor_index = i
                    found_factor = True
                    break
            if not found_factor:
                # This case should ideally not be hit if max_abs_spec <= 1.0 and 1.0 is in table.
                # If max_abs_spec was > 1.0 and clipped, and 1.0 is largest, this won't be hit.
                # If table doesn't contain 1.0 but max_abs_spec is 1.0, use largest.
                chosen_scale_factor_index = len(self.codec_data.scale_table) - 1

        chosen_scale_factor = self.codec_data.scale_table[chosen_scale_factor_index]

        scaled_values: List[float] = []
        if chosen_scale_factor == 0:
            scaled_values = [0.0] * len(spectral_coeffs)
        else:
            for coeff in spectral_coeffs:
                scaled_val = coeff / chosen_scale_factor
                if scaled_val >= 1.0:
                    scaled_val = 0.99999
                elif scaled_val <= -1.0:
                    scaled_val = -0.99999
                scaled_values.append(scaled_val)

        return ScaledBlock(
            scale_factor_index=chosen_scale_factor_index,
            scaled_values=scaled_values,
            max_energy=max_energy,
        )


def _round_half_away_from_zero(x: float) -> int:
    """Replicates C's lrint rounding behavior (round half away from zero)."""
    if x >= 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

def _round_half_to_even(x: float) -> int:
    """Replicates C's lrint rounding behavior (round half to even) using np.rint."""
    # np.rint rounds to the nearest even integer for .5 cases
    # and then we cast to int.
    return int(np.rint(x))

def quantize_mantissas(
    scaled_values: List[float],
    word_length: int,
    perform_energy_adjustment: bool = False,
) -> Tuple[List[int], float, float]:
    """
    Quantizes scaled spectral coefficients into integer mantissas.

    Args:
        scaled_values: List of normalized floating-point spectral coefficients.
        word_length: The number of bits allocated for each mantissa (including sign).
                     If 0, all mantissas are 0.
        perform_energy_adjustment: If True, attempts to adjust rounding to preserve energy.

    Returns:
        A tuple containing:
            - List of integer mantissas.
            - Original energy (sum of squares of scaled_values).
            - Quantized energy (sum of squares of de-quantized mantissas).
    """
    if word_length == 0:
        return [0] * len(scaled_values), 0.0, 0.0

    mantissas: List[int] = []
    original_energy = 0.0
    quantized_energy = 0.0

    if word_length == 1:
        for val in scaled_values:
            original_energy += val * val
            quantized_mantissa = -1 if val < 0 else 0
            mantissas.append(quantized_mantissa)
            dequant_val_at_unity_scale = float(quantized_mantissa)
            quantized_energy += dequant_val_at_unity_scale * dequant_val_at_unity_scale
    else:  # word_length >= 2
        # max_quant_val is (2 to the power of (word_length-1)) - 1.
        # This is the max positive integer for the symmetric part of the range.
        max_quant_val = (1 << (word_length - 1)) - 1

        # Initial quantization
        for val in scaled_values:
            original_energy += val * val
            # Use round half to even, matching C++ lrint via ToInt()
            int_mantissa = _round_half_to_even(val * max_quant_val)
            int_mantissa = max(-max_quant_val, min(int_mantissa, max_quant_val))
            mantissas.append(int_mantissa)

        # Energy adjustment logic (only for word_length > 1, which is true here)
        if perform_energy_adjustment and max_quant_val > 0:
            # max_quant_val is > 0 if word_length >= 2
            dequant_factor = 1.0 / max_quant_val

            # Inner function for Energy Adjustment
            def calculate_q_energy_ea(current_mantissas_ea: List[int]) -> float:
                energy_ea = 0.0
                for m_val_ea in current_mantissas_ea:
                    dequantized_val_ea = m_val_ea * dequant_factor
                    energy_ea += dequantized_val_ea * dequantized_val_ea
                return energy_ea

            current_quantized_energy_ea = calculate_q_energy_ea(mantissas)
            # Max iterations for the adjustment loop as a safeguard
            # Calculate initial quantized_energy from initial mantissas
            current_quantized_energy = 0.0
            if max_quant_val > 0:
                dequant_factor = 1.0 / max_quant_val
                for m_val in mantissas:
                    dequantized_val = m_val * dequant_factor
                    current_quantized_energy += dequantized_val * dequantized_val
            else: # Should not happen for wl >= 2
                current_quantized_energy = 0.0

            # C++ EA logic starts here
            candidates = [] # List of (abs_delta, original_index, t_original)

            for i, scaled_val in enumerate(scaled_values):
                t_original = scaled_val * max_quant_val

                # Calculate delta: t - (trunc(t) + copysign(0.5, t))
                # For t = 0, copysign(0.5, 0) is 0.5. trunc(0)+0.5 = 0.5. delta = -0.5. abs(delta)=0.5. Not candidate.
                # This is fine as t=0 typically means mantissa is 0 and won't be adjusted by this logic.
                ref_point = np.trunc(t_original) + np.copysign(0.5, t_original)
                delta = t_original - ref_point

                if abs(delta) < 0.25:
                    # Store abs(delta), original index, and t_original (scaled_val * max_quant_val)
                    candidates.append((abs(delta), i, t_original))

            if not candidates:
                quantized_energy = current_quantized_energy
                return mantissas, original_energy, quantized_energy

            # Sort candidates by abs(delta) in ascending order
            candidates.sort(key=lambda x: x[0])

            # Iterative adjustment based on sorted candidates
            for _, original_idx, t_original_for_candidate in candidates:
                # Calculate current total quantized energy for comparison (e2 in C++)
                # This needs to be recalculated if a mantissa changed in a previous iteration.
                # Or, update current_quantized_energy incrementally. Let's do incremental.

                # Test conditions for increasing or decreasing magnitude
                current_mantissa_val = mantissas[original_idx]
                m_new = current_mantissa_val # trial new mantissa

                candidate_made_change = False

                if current_quantized_energy < original_energy:
                    # Try to increase quantized energy by increasing mantissa magnitude
                    # Conditions: abs(current_mantissa) < abs(t_original) AND abs(current_mantissa) < max_quant_val - 1
                    if abs(current_mantissa_val) < abs(t_original_for_candidate) and \
                       abs(current_mantissa_val) < max_quant_val -1: # max_quant_val-1 is like C++ (mul-1)

                        if current_mantissa_val > 0:
                            m_new = current_mantissa_val + 1
                        elif current_mantissa_val < 0:
                            m_new = current_mantissa_val - 1
                        else: # current_mantissa_val == 0
                            m_new = 1 if t_original_for_candidate > 0 else -1

                        # Clamp m_new (though conditions might make this redundant)
                        m_new = max(-max_quant_val, min(m_new, max_quant_val))
                        candidate_made_change = True

                elif current_quantized_energy > original_energy:
                    # Try to decrease quantized energy by decreasing mantissa magnitude
                    # Condition: abs(current_mantissa) > abs(t_original)
                    if abs(current_mantissa_val) > abs(t_original_for_candidate):
                        if current_mantissa_val > 0:
                            m_new = current_mantissa_val - 1
                        elif current_mantissa_val < 0:
                            m_new = current_mantissa_val + 1
                        # if current_mantissa_val == 0, no change as per this condition.
                        candidate_made_change = True

                if candidate_made_change and m_new != current_mantissa_val:
                    # Check if this change improves overall energy match
                    old_term_energy = (current_mantissa_val * dequant_factor)**2
                    new_term_energy = (m_new * dequant_factor)**2

                    trial_quantized_energy = current_quantized_energy - old_term_energy + new_term_energy

                    error_before_change = abs(current_quantized_energy - original_energy)
                    error_after_change = abs(trial_quantized_energy - original_energy)

                    if error_after_change < error_before_change:
                        mantissas[original_idx] = m_new
                        current_quantized_energy = trial_quantized_energy

            quantized_energy = current_quantized_energy
        else:
            # No energy adjustment (wl < 2 or EA flag off)
            # Calculate quantized_energy directly from initial mantissas
            if max_quant_val > 0:
                dequant_factor = 1.0 / max_quant_val
                for m_val in mantissas:
                    dequantized_val = m_val * dequant_factor
                    quantized_energy += dequantized_val * dequantized_val
            # If max_quant_val was 0, q_energy remains 0.0 (already initialized)

    return mantissas, original_energy, quantized_energy


class BitstreamSignedValues:
    """
    Handles encoding and decoding of signed integer values for the ATRAC1 bitstream.
    Equivalent to NBitStream::MakeSign logic.
    """

    @staticmethod
    def encode_signed(value: int, num_bits: int) -> int:
        """
        Prepares a signed integer for writing to the bitstream with 'num_bits'.
        This involves ensuring it's represented correctly as an unsigned value
        if negative, fitting within num_bits.
        Example: For num_bits=4, value -1 (0b...1111) should be stored as 0b1111 (15).
                 value -7 (0b...1001) should be stored as 0b1001 (9).
                 value +7 (0b...0111) should be stored as 0b0111 (7).

        Args:
            value: The signed integer mantissa.
            num_bits: The word length for this mantissa.

        Returns:
            An unsigned integer representation suitable for bitstream writing.
        """
        if num_bits <= 0:
            return 0
        if num_bits > 32:  # Practical limit for typical integer operations
            raise ValueError("num_bits too large for standard integer types")

        # Mask to get the relevant number of bits
        mask = (1 << num_bits) - 1

        if value < 0:
            # For negative numbers, we want the two's complement representation
            # truncated to num_bits.
            # e.g., value = -1, num_bits = 4.  -1 is ...1111. Masked with 0xF gives 15 (0b1111).
            # e.g., value = -7, num_bits = 4.  -7 is ...1001. Masked with 0xF gives 9 (0b1001).
            # Python handles negative numbers with infinite leading 1s in two's complement.
            # So, simple masking works.
            return value & mask

        # For positive numbers, just ensure it fits.
        # Max positive for num_bits is (1 << (num_bits - 1)) - 1
        # If value exceeds this, it's an encoding error or needs clamping before this stage.
        # Here, we just take the lower num_bits.
        return value & mask

    @staticmethod
    def decode_signed(unsigned_value: int, num_bits: int) -> int:
        """
        Reconstructs a signed integer from its 'num_bits' representation
        read from the bitstream.
        Example: For num_bits=4, unsigned_value 15 (0b1111) should become -1.
                 unsigned_value 9 (0b1001) should become -7.
                 unsigned_value 7 (0b0111) should become +7.

        Args:
            unsigned_value: The unsigned integer read from the bitstream.
            num_bits: The word length used for this mantissa.

        Returns:
            The reconstructed signed integer.
        """
        if num_bits <= 0:
            return 0
        if num_bits > 32:
            raise ValueError("num_bits too large for standard integer types")

        value = unsigned_value

        # Check the sign bit (the MSB of the num_bits field)
        sign_bit_mask = 1 << (num_bits - 1)
        if (value & sign_bit_mask) != 0:  # If sign bit is set, it's a negative number
            # Perform sign extension: if it was negative, fill higher bits with 1s.
            # This is done by subtracting (1 << num_bits)
            # e.g., num_bits=4, value=15 (0b1111). sign_bit is set. 15 - (1<<4) = 15 - 16 = -1.
            # e.g., num_bits=4, value=9 (0b1001). sign_bit is set. 9 - (1<<4) = 9 - 16 = -7.
            return value - (1 << num_bits)

        # Positive number, return as is
        return value
