import unittest
from pyatrac1.core.codec_data import Atrac1CodecData
from pyatrac1.tables.scale_table import generate_scale_table, ATRAC1_SCALE_TABLE
from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.tables.spectral_mapping import SPECS_PER_BLOCK, BFU_AMOUNT_TAB


class TestCodecDataInitialization(unittest.TestCase):
    def test_atrac1_codec_data_initialization(self):
        """
        Tests if Atrac1CodecData correctly initializes its attributes with
        the data from the respective table modules.
        """
        codec_data = Atrac1CodecData()

        # 1. Test scale_table
        # Atrac1CodecData calls generate_scale_table() internally.
        # We can also compare against the pre-generated ATRAC1_SCALE_TABLE from the module.
        expected_scale_table = generate_scale_table()
        self.assertListEqual(
            codec_data.scale_table,
            expected_scale_table,
            "Atrac1CodecData.scale_table does not match generate_scale_table()",
        )
        self.assertListEqual(
            codec_data.scale_table,
            ATRAC1_SCALE_TABLE,
            "Atrac1CodecData.scale_table does not match module's ATRAC1_SCALE_TABLE",
        )

        # 2. Test sine_window
        expected_sine_window = generate_sine_window()
        self.assertListEqual(
            codec_data.sine_window,
            expected_sine_window,
            "Atrac1CodecData.sine_window does not match generate_sine_window()",
        )

        # 3. Test specs_per_block
        self.assertListEqual(
            codec_data.specs_per_block,
            SPECS_PER_BLOCK,
            "Atrac1CodecData.specs_per_block does not match spectral_mapping.SPECS_PER_BLOCK",
        )

        # 4. Test bfu_amount_tab
        self.assertListEqual(
            codec_data.bfu_amount_tab,
            BFU_AMOUNT_TAB,
            "Atrac1CodecData.bfu_amount_tab does not match spectral_mapping.BFU_AMOUNT_TAB",
        )


if __name__ == "__main__":
    unittest.main()
