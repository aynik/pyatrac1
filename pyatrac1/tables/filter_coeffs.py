"""
QMF and MDCT coefficients for the ATRAC1 codec.
These coefficients define the fundamental transformations (QMF and MDCT) within the codec.
"""

import math

TAP_HALF = [
    -1.461907e-05,
    -9.205479e-05,
    -5.6157569e-05,
    3.0117269e-04,
    2.422519e-04,
    -8.5293897e-04,
    -5.205574e-04,
    2.0340169e-03,
    7.8333891e-04,
    -4.2153862e-03,
    -7.5614988e-04,
    7.8402944e-03,
    -6.1169922e-05,
    -1.344162e-02,
    2.4626821e-03,
    2.1736089e-02,
    -7.801671e-03,
    -3.4090221e-02,
    1.880949e-02,
    5.4326009e-02,
    -4.3596379e-02,
    -9.9384367e-02,
    1.3207909e-01,
    4.6424159e-01,
]


def generate_qmf_window() -> list[float]:
    """
    Generates the QmfWindow (48-tap symmetric FIR filter) from TapHalf for QMF analysis and synthesis.
    Derivation: QmfWindow[i] = TapHalf[i] * 2.0 and QmfWindow[sz - 1 - i] = TapHalf[i] * 2.0 for i from 0 to 23,
    where sz is 48.
    """
    sz = 48
    qmf_window = [0.0] * sz
    for i in range(24):
        qmf_window[i] = TAP_HALF[i] * 2.0
        qmf_window[sz - 1 - i] = TAP_HALF[i] * 2.0
    return qmf_window


def generate_sine_window() -> list[float]:
    """
    Generates the SineWindow (32-entry floating-point array) used in MDCT operations for TDAC.
    Formula: SineWindow[i] = sin((i + 0.5) * (M_PI / (2.0 * 32.0))) for i from 0 to 31.
    This simplifies to: sin((i + 0.5) * π / 64.0)
    """
    sine_window: list[float] = []
    for i in range(32):
        value = math.sin((i + 0.5) * (math.pi / 64.0))
        sine_window.append(value)
    return sine_window


HPF_FIR_COEFFS = [
    -8.65163e-18 * 2.0,
    -0.00851586 * 2.0,
    -6.74764e-18 * 2.0,
    0.0209036 * 2.0,
    -3.36639e-17 * 2.0,
    -0.0438162 * 2.0,
    -1.54175e-17 * 2.0,
    0.0931738 * 2.0,
    -5.52212e-17 * 2.0,
    -0.313819 * 2.0,
]

HPF_FIR_LEN = 21
HPF_PREV_BUF_SZ = 20
