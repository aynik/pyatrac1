"""
Psychoacoustic model constants for the ATRAC1 codec.
These constants and formulas are fundamental to the perceptual coding aspect of ATRAC1.
"""

import math
from pyatrac1.common.constants import NUM_SAMPLES

ATH_LOOKUP_TABLE = [
    (10.0, 9669, 9669, 9626, 9512),
    (12.6, 9353, 9113, 8882, 8676),
    (15.8, 8469, 8243, 7997, 7748),
    (20.0, 7492, 7239, 7000, 6762),
    (25.1, 6529, 6302, 6084, 5900),
    (31.6, 5717, 5534, 5351, 5167),
    (39.8, 5004, 4812, 4638, 4466),
    (50.1, 4310, 4173, 4050, 3922),
    (63.1, 3723, 3577, 3451, 3281),
    (79.4, 3132, 3036, 2902, 2760),
    (100.0, 2658, 2591, 2441, 2301),
    (125.9, 2212, 2125, 2018, 1900),
    (158.5, 1770, 1682, 1594, 1512),
    (199.5, 1430, 1341, 1260, 1198),
    (251.2, 1136, 1057, 998, 943),
    (316.2, 887, 846, 744, 712),
    (398.1, 693, 668, 637, 606),
    (501.2, 580, 555, 529, 502),
    (631.0, 475, 448, 422, 398),
    (794.3, 375, 351, 327, 322),
    (1000.0, 312, 301, 291, 268),
    (1258.9, 246, 215, 182, 146),
    (1584.9, 107, 61, 13, -35),
    (1995.3, -96, -156, -179, -235),
    (2511.9, -295, -350, -401, -421),
    (3162.3, -446, -499, -532, -535),
    (3981.1, -513, -476, -431, -313),
    (5011.9, -179, 8, 203, 403),
    (6309.6, 580, 736, 881, 1022),
    (7943.3, 1154, 1251, 1348, 1421),
    (10000.0, 1479, 1399, 1285, 1193),
    (12589.3, 1287, 1519, 1914, 2369),
    (15848.9, 3352, 4352, 5352, 6352),
    (19952.6, 7352, 8352, 9352, 9999),
    (25118.9, 9999, 9999, 9999, 9999),
]

# Flattened ATH lookup table for Frank's formula, matching C++ static short tab[]
# Each entry from ATH_LOOKUP_TABLE contributes its 4 data values consecutively.
FLAT_ATH_TAB_FRANK: list[int] = []
for entry in ATH_LOOKUP_TABLE:
    FLAT_ATH_TAB_FRANK.extend(entry[1:]) # Add v1, v2, v3, v4


def generate_loudness_curve() -> list[float]:
    """
    Generates the Loudness Curve as per the technical specification (Table 2.7.2).
    This curve models the human ear's non-linear sensitivity to loudness across frequencies.
    It has 512 entries (NUM_SAMPLES).

    Generation Formula:
    1. Map spectral index i to frequency f: f = (i + 3) * 0.5 * 44100 / sz
    2. Compute intermediate loudness t (logarithmic scale): t = std::log10(f) - 3.5
    3. Adjust t with parabolic and linear terms: t = -10 * t * t + 3 - f / 3000
    4. Convert t to linear scale: t = std::pow(10, (0.1 * t))
    """
    loudness_curve: list[float] = []
    sz = NUM_SAMPLES  # Use the imported constant
    sample_rate = 44100.0

    for i in range(sz):
        # 1. Map spectral index i to frequency f
        f = (i + 3) * 0.5 * sample_rate / sz

        # Avoid log10(0) or negative for very low frequencies
        if f <= 0:
            lou_t = -float("inf")
        else:
            # 2. Compute intermediate loudness t (logarithmic scale)
            lou_t = math.log10(f) - 3.5

            # 3. Adjust t with parabolic and linear terms
            lou_t = -10 * lou_t * lou_t + 3 - f / 3000.0

        # 4. Convert t to linear scale (handle -inf from log10(0) case)
        if lou_t == -float("inf"):
            value = 0.0  # Or some other appropriate minimum value
        else:
            value = math.pow(10, (0.1 * lou_t))
        loudness_curve.append(value)

    return loudness_curve
