import unittest
from pyatrac1.tables import spectral_mapping as sm


class TestSpectralMappingTables(unittest.TestCase):
    def test_specs_per_block(self):
        # From spec.txt Table 2.2.1
        expected_specs_per_block = [
            8,
            8,
            8,
            8,  # 0-3
            4,
            4,
            4,
            4,  # 4-7
            8,
            8,
            8,
            8,  # 8-11
            6,
            6,
            6,
            6,  # 12-15
            6,  # 16
            6,
            6,
            6,  # 17-19
            6,
            6,
            6,
            6,  # 20-23
            7,
            7,
            7,
            7,  # 24-27
            9,
            9,
            9,
            9,  # 28-31
            10,
            10,
            10,
            10,  # 32-35
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,  # 36-43
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,  # 44-51
        ]
        self.assertEqual(
            len(sm.SPECS_PER_BLOCK), 52, "SPECS_PER_BLOCK should have 52 entries"
        )
        self.assertListEqual(
            sm.SPECS_PER_BLOCK,
            expected_specs_per_block,
            "SPECS_PER_BLOCK values do not match spec",
        )

    def test_blocks_per_band(self):
        # From spec.txt Table 2.2.2
        expected_blocks_per_band = [
            (0, 20),  # Low
            (20, 36),  # Mid
            (36, 52),  # High
        ]
        self.assertEqual(
            len(sm.BLOCKS_PER_BAND), 3, "BLOCKS_PER_BAND should have 3 entries"
        )
        self.assertListEqual(
            sm.BLOCKS_PER_BAND,
            expected_blocks_per_band,
            "BLOCKS_PER_BAND values do not match spec",
        )

    def test_specs_start_long(self):
        # From spec.txt Table 2.2.3
        expected_specs_start_long = [
            0,
            8,
            16,
            24,
            32,
            36,
            40,
            44,
            48,
            56,
            64,
            72,
            80,
            86,
            92,
            98,
            104,
            110,
            116,
            122,
            128,
            134,
            140,
            146,
            152,
            159,
            166,
            173,
            180,
            189,
            198,
            207,
            216,
            226,
            236,
            246,
            256,
            268,
            280,
            292,
            304,
            316,
            328,
            340,
            352,
            372,
            392,
            412,
            432,
            452,
            472,
            492,
        ]
        self.assertEqual(
            len(sm.SPECS_START_LONG), 52, "SPECS_START_LONG should have 52 entries"
        )
        self.assertListEqual(
            sm.SPECS_START_LONG,
            expected_specs_start_long,
            "SPECS_START_LONG values do not match spec",
        )

    def test_specs_start_short(self):
        # From spec.txt Table 2.2.4
        # Note: The spec table is presented differently, this is the flattened version matching the code.
        expected_specs_start_short = [
            0,
            32,
            64,
            96,  # BFU 0-3 (maps to 4 short blocks each)
            8,
            40,
            72,
            104,  # BFU 4-7
            12,
            44,
            76,
            108,  # BFU 8-11
            20,
            52,
            84,
            116,  # BFU 12-15
            26,
            58,
            90,
            122,  # BFU 16-19 (BFU 16 is single, 17-19 are grouped)
            128,
            160,
            192,
            224,  # BFU 20-23
            134,
            166,
            198,
            230,  # BFU 24-27
            141,
            173,
            205,
            237,  # BFU 28-31
            150,
            182,
            214,
            246,  # BFU 32-35
            256,
            288,
            320,
            352,  # BFU 36 (first of high band) - 39
            384,
            416,
            448,
            480,  # BFU 40 - 43
            268,
            300,
            332,
            364,  # BFU 44 - 47
            396,
            428,
            460,
            492,  # BFU 48 - 51
        ]
        self.assertEqual(
            len(sm.SPECS_START_SHORT), 52, "SPECS_START_SHORT should have 52 entries"
        )
        self.assertListEqual(
            sm.SPECS_START_SHORT,
            expected_specs_start_short,
            "SPECS_START_SHORT values do not match spec",
        )

    def test_bfu_amount_tab(self):
        # From spec.txt Table 2.2.5
        expected_bfu_amount_tab = [20, 28, 32, 36, 40, 44, 48, 52]
        self.assertEqual(
            len(sm.BFU_AMOUNT_TAB), 8, "BFU_AMOUNT_TAB should have 8 entries"
        )
        self.assertListEqual(
            sm.BFU_AMOUNT_TAB,
            expected_bfu_amount_tab,
            "BFU_AMOUNT_TAB values do not match spec",
        )


if __name__ == "__main__":
    unittest.main()
