import unittest
from pyatrac1.tables import bit_allocation as ba


class TestFixedBitAllocationTables(unittest.TestCase):
    def test_fixed_bit_alloc_table_long(self):
        # From spec.txt Table 2.4.1
        expected_table_long = [
            7,
            7,
            7,  # BFU 0-2
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,  # BFU 3-18
            6,
            6,
            6,  # BFU 19-21
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,  # BFU 22-34
            4,  # BFU 35
            4,
            4,
            4,  # BFU 36-38
            3,
            3,
            3,
            3,
            3,
            3,  # BFU 39-44
            1,
            1,
            1,
            1,  # BFU 45-48
            0,
            0,
            0,  # BFU 49-51
        ]
        self.assertEqual(
            len(ba.FIXED_BIT_ALLOC_TABLE_LONG),
            52,
            "FIXED_BIT_ALLOC_TABLE_LONG should have 52 entries",
        )
        self.assertListEqual(
            ba.FIXED_BIT_ALLOC_TABLE_LONG,
            expected_table_long,
            "FIXED_BIT_ALLOC_TABLE_LONG values do not match spec",
        )

    def test_fixed_bit_alloc_table_short(self):
        # From spec.txt Table 2.4.2
        expected_table_short = [
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,  # BFU 0-23
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,  # BFU 24-35
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,  # BFU 36-43
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # BFU 44-51
        ]
        self.assertEqual(
            len(ba.FIXED_BIT_ALLOC_TABLE_SHORT),
            52,
            "FIXED_BIT_ALLOC_TABLE_SHORT should have 52 entries",
        )
        self.assertListEqual(
            ba.FIXED_BIT_ALLOC_TABLE_SHORT,
            expected_table_short,
            "FIXED_BIT_ALLOC_TABLE_SHORT values do not match spec",
        )

    def test_bit_boost_mask(self):
        # From spec.txt Table 2.4.3
        expected_bit_boost_mask = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # BFU 0-17
            1,
            1,
            1,
            1,
            1,  # BFU 18-22
            0,  # BFU 23
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # BFU 24-31
            1,
            1,
            1,
            1,  # BFU 32-35
            1,
            1,
            1,  # BFU 36-38
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # BFU 39-51
        ]
        self.assertEqual(
            len(ba.BIT_BOOST_MASK), 52, "BIT_BOOST_MASK should have 52 entries"
        )
        self.assertListEqual(
            ba.BIT_BOOST_MASK,
            expected_bit_boost_mask,
            "BIT_BOOST_MASK values do not match spec",
        )


if __name__ == "__main__":
    unittest.main()
