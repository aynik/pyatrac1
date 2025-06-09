import unittest
import math
from pyatrac1.tables import scale_table as st


class TestScaleTable(unittest.TestCase):
    def test_scale_table_generation(self):
        # From spec.txt Section 2.3
        # Formula: ScaleTable[i] = pow(2.0, (i / 3.0 - 21.0)) for i from 0 to 63.
        self.assertEqual(
            len(st.ATRAC1_SCALE_TABLE), 64, "ScaleTable should have 64 entries"
        )

        for i in range(64):
            expected_value = math.pow(2.0, (i / 3.0 - 21.0))
            # Using 9 decimal places for comparison, as spec values have high precision
            self.assertAlmostEqual(
                st.ATRAC1_SCALE_TABLE[i],
                expected_value,
                places=9,
                msg=f"ScaleTable value at index {i} does not match expected formula result",
            )


if __name__ == "__main__":
    unittest.main()
