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

        chosen_scale_factor_index = 0
        chosen_scale_factor = self.codec_data.scale_table[0]  # Default to smallest

        # Iterate through scale_table to find the smallest factor >= max_abs_spec
        # The scale_table values are increasing.
        for i, factor_val in enumerate(self.codec_data.scale_table):
            if factor_val >= max_abs_spec:
                chosen_scale_factor_index = i
                chosen_scale_factor = factor_val
                break
        else:
            # If max_abs_spec is larger than any value in scale_table,
            # use the largest scale factor.
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
            int_mantissa = _round_half_away_from_zero(val * max_quant_val)
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
            max_ea_loops = len(scaled_values) + 5
            for _ea_loop_count in range(max_ea_loops):
                best_improvement_metric = abs(
                    current_quantized_energy_ea - original_energy
                )
                candidate_index_to_flip = -1
                new_mantissa_for_candidate = 0
                improved_energy_for_candidate = 0.0

                for i in range(len(scaled_values)):
                    s_val_scaled = scaled_values[i] * max_quant_val
                    if s_val_scaled == float(mantissas[i]):  # Already exact
                        continue

                    m_floor = math.floor(s_val_scaled)
                    m_ceil = math.ceil(s_val_scaled)
                    current_m_i = mantissas[i]
                    alt_m_i = 0  # Alternative mantissa

                    if current_m_i == m_floor:
                        alt_m_i = m_ceil
                    elif current_m_i == m_ceil:
                        alt_m_i = m_floor
                    else:
                        # This path should not be hit if current_m_i was from rounding
                        continue

                    clamped_alt_m_i = max(-max_quant_val, min(alt_m_i, max_quant_val))
                    if (
                        clamped_alt_m_i == current_m_i
                    ):  # Alternative is same after clamping
                        continue

                    temp_mantissas_list = list(
                        mantissas
                    )  # Use a different name to avoid confusion
                    temp_mantissas_list[i] = clamped_alt_m_i
                    temp_q_energy = calculate_q_energy_ea(temp_mantissas_list)
                    new_improvement_metric = abs(temp_q_energy - original_energy)

                    if new_improvement_metric < best_improvement_metric:
                        best_improvement_metric = new_improvement_metric
                        candidate_index_to_flip = i
                        new_mantissa_for_candidate = clamped_alt_m_i
                        improved_energy_for_candidate = temp_q_energy

                if candidate_index_to_flip != -1:  # An improvement was found
                    mantissas[candidate_index_to_flip] = new_mantissa_for_candidate
                    current_quantized_energy_ea = improved_energy_for_candidate
                else:  # No improvement found in this pass
                    break
            quantized_energy = current_quantized_energy_ea  # Final q_energy from EA
        else:
            # No energy adjustment, or max_quant_val is 0 (not possible for wl>=2)
            # Calculate quantized_energy directly from initial mantissas
            if max_quant_val > 0:  # True for word_length >= 2
                dequant_factor = 1.0 / max_quant_val
                for m_val in mantissas:
                    dequantized_val = m_val * dequant_factor
                    quantized_energy += dequantized_val * dequantized_val
            # If max_quant_val was 0 (not possible for wl>=2), q_energy remains 0.0

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
