import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode
from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.common.utils import swap_array


@pytest.fixture
def mdct_instance():
    """Fixture to provide an instance of Atrac1MDCT."""
    return Atrac1MDCT()


@pytest.fixture
def base_sine_window_32():
    """Fixture for the base 32-point sine window."""
    return np.array(generate_sine_window(), dtype=np.float32)


# Test BlockSizeMode (optional, but good for sanity check)
def test_block_size_mode_properties():
    bsm_long = BlockSizeMode(False, False, False)
    assert bsm_long.low_mdct_size == 128
    assert bsm_long.mid_mdct_size == 128
    assert bsm_long.high_mdct_size == 256

    bsm_short = BlockSizeMode(True, True, True)
    assert bsm_short.low_mdct_size == 64
    assert bsm_short.mid_mdct_size == 64
    assert bsm_short.high_mdct_size == 64

    bsm_mixed = BlockSizeMode(True, False, True)
    assert bsm_mixed.low_mdct_size == 64
    assert bsm_mixed.mid_mdct_size == 128
    assert bsm_mixed.high_mdct_size == 64


# Test basic MDCT functionality
def test_mdct_basic_functionality(mdct_instance):
    """Test that MDCT processes input without errors."""
    bsm = BlockSizeMode(False, False, False)  # All long windows
    
    # Create input buffers
    low_buf = np.random.randn(256).astype(np.float64) * 0.1
    mid_buf = np.random.randn(256).astype(np.float64) * 0.1  
    hi_buf = np.random.randn(512).astype(np.float64) * 0.1
    
    # Create output specs array
    specs = np.zeros(512, dtype=np.float64)
    
    # Should not raise an exception
    mdct_instance.mdct(specs, low_buf, mid_buf, hi_buf, bsm)
    
    # Check that specs were modified
    assert not np.allclose(specs, 0.0), "MDCT output should not be all zeros"
    assert np.all(np.isfinite(specs)), "MDCT output should not contain NaN or infinity"


def test_imdct_basic_functionality(mdct_instance):
    """Test that IMDCT processes input without errors."""
    bsm = BlockSizeMode(False, False, False)  # All long windows
    
    # Create input specs array with some coefficients
    specs = np.random.randn(512).astype(np.float64) * 0.1
    
    # Create output buffers
    low_buf = np.zeros(256, dtype=np.float64)
    mid_buf = np.zeros(256, dtype=np.float64)
    hi_buf = np.zeros(512, dtype=np.float64)
    
    # Should not raise an exception
    mdct_instance.imdct(specs, bsm, low_buf, mid_buf, hi_buf)
    
    # Check that buffers were modified
    assert not np.allclose(low_buf, 0.0), "IMDCT low output should not be all zeros"
    assert not np.allclose(mid_buf, 0.0), "IMDCT mid output should not be all zeros"
    assert not np.allclose(hi_buf, 0.0), "IMDCT high output should not be all zeros"


def test_mdct_different_block_modes(mdct_instance):
    """Test MDCT with different block size modes."""
    
    # Test long windows
    bsm_long = BlockSizeMode(False, False, False)
    low_buf = np.random.randn(256).astype(np.float64) * 0.1
    mid_buf = np.random.randn(256).astype(np.float64) * 0.1  
    hi_buf = np.random.randn(512).astype(np.float64) * 0.1
    specs_long = np.zeros(512, dtype=np.float64)
    
    mdct_instance.mdct(specs_long, low_buf, mid_buf, hi_buf, bsm_long)
    assert not np.allclose(specs_long, 0.0)
    
    # Test short windows
    bsm_short = BlockSizeMode(True, True, True)
    specs_short = np.zeros(512, dtype=np.float64)
    
    mdct_instance.mdct(specs_short, low_buf, mid_buf, hi_buf, bsm_short)
    assert not np.allclose(specs_short, 0.0)
    
    # Outputs should be different for different window modes
    assert not np.allclose(specs_long, specs_short, atol=1e-10)


def test_mdct_impulse_response(mdct_instance):
    """Test MDCT response to impulse input."""
    bsm = BlockSizeMode(False, False, False)  # All long
    
    # Create impulse in low band
    low_buf = np.zeros(256, dtype=np.float64)
    low_buf[0] = 1.0  # Impulse
    mid_buf = np.zeros(256, dtype=np.float64)
    hi_buf = np.zeros(512, dtype=np.float64)
    
    specs = np.zeros(512, dtype=np.float64)
    mdct_instance.mdct(specs, low_buf, mid_buf, hi_buf, bsm)
    
    # For an impulse, MDCT coefficients should be non-zero and spread out
    assert not np.allclose(specs, 0.0)
    assert np.sum(np.abs(specs)) > 0.1  # Should have significant energy


def test_imdct_different_block_modes(mdct_instance):
    """Test IMDCT with different block size modes."""
    
    # Test long windows
    bsm_long = BlockSizeMode(False, False, False)
    specs = np.random.randn(512).astype(np.float64) * 0.1
    
    low_buf_long = np.zeros(256, dtype=np.float64)
    mid_buf_long = np.zeros(256, dtype=np.float64)
    hi_buf_long = np.zeros(512, dtype=np.float64)
    
    mdct_instance.imdct(specs, bsm_long, low_buf_long, mid_buf_long, hi_buf_long)
    assert not np.allclose(low_buf_long, 0.0)
    
    # Test short windows  
    bsm_short = BlockSizeMode(True, True, True)
    low_buf_short = np.zeros(256, dtype=np.float64)
    mid_buf_short = np.zeros(256, dtype=np.float64)
    hi_buf_short = np.zeros(512, dtype=np.float64)
    
    mdct_instance.imdct(specs, bsm_short, low_buf_short, mid_buf_short, hi_buf_short)
    assert not np.allclose(low_buf_short, 0.0)
    
    # Outputs should be different for different window modes
    assert not np.allclose(low_buf_long, low_buf_short, atol=1e-10)


def test_mdct_imdct_round_trip(mdct_instance):
    """Test basic MDCT -> IMDCT round trip functionality."""
    
    # Test with long windows
    bsm = BlockSizeMode(False, False, False)
    
    # Create input buffers with some energy
    low_buf_orig = np.random.randn(256).astype(np.float64) * 0.1
    mid_buf_orig = np.random.randn(256).astype(np.float64) * 0.1  
    hi_buf_orig = np.random.randn(512).astype(np.float64) * 0.1
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float64)
    mdct_instance.mdct(specs, low_buf_orig.copy(), mid_buf_orig.copy(), hi_buf_orig.copy(), bsm)
    
    # Inverse MDCT
    low_buf_recon = np.zeros(256, dtype=np.float64)
    mid_buf_recon = np.zeros(256, dtype=np.float64)
    hi_buf_recon = np.zeros(512, dtype=np.float64)
    
    mdct_instance.imdct(specs, bsm, low_buf_recon, mid_buf_recon, hi_buf_recon)
    
    # Check that reconstruction has similar energy (allowing for windowing effects)
    orig_energy = np.sum(low_buf_orig**2) + np.sum(mid_buf_orig**2) + np.sum(hi_buf_orig**2)
    recon_energy = np.sum(low_buf_recon**2) + np.sum(mid_buf_recon**2) + np.sum(hi_buf_recon**2)
    
    energy_ratio = recon_energy / orig_energy if orig_energy > 0 else 1.0
    assert 0.01 < energy_ratio < 100.0, f"Energy ratio {energy_ratio} outside acceptable range"
    
    # Check that reconstruction is not all zeros
    assert not np.allclose(low_buf_recon, 0.0), "Low band reconstruction is all zeros"
    assert not np.allclose(mid_buf_recon, 0.0), "Mid band reconstruction is all zeros"  
    assert not np.allclose(hi_buf_recon, 0.0), "High band reconstruction is all zeros"
    
    # Check for finite values
    assert np.all(np.isfinite(low_buf_recon)), "Low band contains NaN or infinity"
    assert np.all(np.isfinite(mid_buf_recon)), "Mid band contains NaN or infinity"
    assert np.all(np.isfinite(hi_buf_recon)), "High band contains NaN or infinity"


def test_mdct_swap_array_effect(mdct_instance):
    """Test that swap_array is applied correctly for mid and high bands."""
    bsm = BlockSizeMode(False, False, False)  # All long
    
    # Create input with some structure to detect swapping
    low_buf = np.zeros(256, dtype=np.float64)
    mid_buf = np.linspace(0, 1, 256).astype(np.float64)  # Structured input 
    hi_buf = np.zeros(512, dtype=np.float64)
    
    specs1 = np.zeros(512, dtype=np.float64)
    mdct_instance.mdct(specs1, low_buf, mid_buf, hi_buf, bsm)
    
    # Create another input with reversed mid band
    mid_buf_rev = mid_buf[::-1]
    specs2 = np.zeros(512, dtype=np.float64)
    mdct_instance.mdct(specs2, low_buf, mid_buf_rev, hi_buf, bsm)
    
    # The mid band coefficients should be different due to swap_array
    mid_coeffs1 = specs1[128:256]
    mid_coeffs2 = specs2[128:256]
    
    assert not np.allclose(mid_coeffs1, mid_coeffs2, atol=1e-10), "Mid band swap_array not working"
