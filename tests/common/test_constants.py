import unittest
from pyatrac1.common import constants as c


class TestGlobalCodecConstants(unittest.TestCase):
    def test_num_qmf_bands(self):
        self.assertEqual(c.NUM_QMF, 3, "Number of QMF bands should be 3")

    def test_num_samples_per_block(self):
        self.assertEqual(
            c.NUM_SAMPLES,
            512,
            "Samples per processing unit per channel should be 512",
        )

    def test_max_bfus(self):
        self.assertEqual(c.MAX_BFUS, 52, "Maximum Basic Frequency Units should be 52")

    def test_sound_unit_size(self):
        self.assertEqual(
            c.SOUND_UNIT_SIZE, 212, "Bytes per compressed frame should be 212"
        )

    def test_bits_per_bfu_amount_tab_idx(self):
        self.assertEqual(
            c.BITS_PER_BFU_AMOUNT_TAB_IDX, 3, "Bits for BFU amount index should be 3"
        )

    def test_bits_per_idwl(self):
        self.assertEqual(
            c.BITS_PER_IDWL, 4, "Bits for Individual Word Length should be 4"
        )

    def test_bits_per_idsf(self):
        self.assertEqual(
            c.BITS_PER_IDSF, 6, "Bits for Individual Scale Factor index should be 6"
        )

    def test_aea_meta_size(self):
        self.assertEqual(
            c.AEA_META_SIZE, 2048, "Bytes for AEA file metadata header should be 2048"
        )

    def test_qmf_delay_compensation(self):
        self.assertEqual(
            c.DELAY_COMP,
            39,
            "QMF delay compensation in samples should be 39",
        )

    def test_loudness_factor_initial(self):
        self.assertAlmostEqual(
            c.LOUD_FACTOR,
            0.006,
            places=3,
            msg="Initial loudness factor should be 0.006",
        )


if __name__ == "__main__":
    unittest.main()
