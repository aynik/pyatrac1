import unittest
import math
from pyatrac1.tables import filter_coeffs as fc


class TestFilterCoefficients(unittest.TestCase):
    def test_tap_half_values(self):
        # From spec.txt Table 2.5.1
        expected_tap_half = [
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
        self.assertEqual(len(fc.TAP_HALF), 24, "TAP_HALF should have 24 entries")
        for i in range(24):
            self.assertAlmostEqual(
                fc.TAP_HALF[i],
                expected_tap_half[i],
                places=9,
                msg=f"TAP_HALF value at index {i} does not match spec",
            )

    def test_qmf_window_generation(self):
        # From spec.txt Section 2.5.2
        # Derivation: QmfWindow[i] = TapHalf[i] * 2.0 and QmfWindow[sz - 1 - i] = TapHalf[i] * 2.0
        qmf_window = fc.generate_qmf_window()
        self.assertEqual(len(qmf_window), 48, "QMF_WINDOW should have 48 entries")
        sz = 48
        for i in range(24):
            expected_val = fc.TAP_HALF[i] * 2.0
            self.assertAlmostEqual(
                qmf_window[i],
                expected_val,
                places=9,
                msg=f"QMF_WINDOW value at index {i} does not match spec derivation",
            )
            self.assertAlmostEqual(
                qmf_window[sz - 1 - i],
                expected_val,
                places=9,
                msg=f"QMF_WINDOW value at index {sz - 1 - i} does not match spec derivation (symmetric)",
            )

    def test_sine_window_generation(self):
        # From spec.txt Table 2.5.3
        # Formula: SineWindow[i] = sin((i + 0.5) * (M_PI / (2.0 * 32.0)))
        # This simplifies to: sin((i + 0.5) * π / 64.0)
        sine_window = fc.generate_sine_window()
        self.assertEqual(len(sine_window), 32, "SINE_WINDOW should have 32 entries")

        # Verify the implementation follows the mathematical formula exactly
        for i in range(32):
            expected_formula_val = math.sin((i + 0.5) * (math.pi / 64.0))
            self.assertAlmostEqual(
                sine_window[i],
                expected_formula_val,
                places=9,
                msg=f"SINE_WINDOW value at index {i} does not match formula",
            )

        # Verify the window has expected properties for MDCT
        # First value should be small, middle should be reasonably large, last should be close to 1
        self.assertLess(sine_window[0], 0.1, "First value should be small")
        self.assertGreater(sine_window[15], 0.68, "Middle value should be reasonably large")
        self.assertGreater(sine_window[31], 0.99, "Last value should be close to 1")

    def test_hpf_fir_coeffs(self):
        # From spec.txt Table 2.6.1
        expected_hpf_coeffs = [
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
        self.assertEqual(
            len(fc.HPF_FIR_COEFFS), 10, "HPF_FIR_COEFFS should have 10 entries"
        )
        for i in range(10):
            self.assertAlmostEqual(
                fc.HPF_FIR_COEFFS[i],
                expected_hpf_coeffs[i],
                places=9,  # High precision for e-18 values
                msg=f"HPF_FIR_COEFFS value at index {i} does not match spec",
            )

    def test_hpf_fir_params(self):
        # From spec.txt Section 2.6
        self.assertEqual(fc.HPF_FIR_LEN, 21, "HPF_FIR_LEN should be 21")
        self.assertEqual(fc.HPF_PREV_BUF_SZ, 20, "HPF_PREV_BUF_SZ should be 20")


if __name__ == "__main__":
    unittest.main()
