"""
Implements Modified Discrete Cosine Transform (MDCT) and Inverse MDCT (IMDCT)
for ATRAC1, including adaptive windowing and Time-Domain Aliasing Cancellation (TDAC).
Uses extracted atracdenc MDCT implementation for maximum compatibility.
"""

import math
import numpy as np

from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.common.utils import swap_array
from pyatrac1.common.debug_logger import log_debug, debug_logger


# Extracted atracdenc MDCT implementation
def vector_fmul_window(dst: np.ndarray, src0: np.ndarray, src1: np.ndarray, win: np.ndarray, length: int):
    """
    Exact implementation of atracdenc vector_fmul_window with proper pointer arithmetic.
    Matches atracdenc lines 50-67.
    """
    # atracdenc pointer arithmetic: dst += len, win += len, src0 += len
    # This means we access elements using negative indexing from the offset position
    
    for i in range(length):
        j = length - 1 - i
        # atracdenc: i goes from -len to -1, j goes from len-1 to 0
        # Our i goes from 0 to len-1, so we map: actual_i = i - len = -len + i
        
        s0 = src0[i]        # src0[i] where i = 0 to len-1
        s1 = src1[j]        # src1[j] where j = len-1 to 0
        wi = win[i]         # win[i] where i = 0 to len-1
        wj = win[j]         # win[j] where j = len-1 to 0
        
        dst[i] = s0 * wj - s1 * wi     # dst[0] to dst[len-1]
        dst[length + j] = s0 * wi + s1 * wj  # dst[len] to dst[2*len-1]

def calc_sin_cos(n, scale):
    tmp = np.zeros(n // 2, dtype=np.float32)  # Use 32-bit to match atracdenc
    alpha = 2.0 * np.pi / (8.0 * n)  # Use np.pi for consistency
    omega = 2.0 * np.pi / n      # Use np.pi for consistency
    # Ensure scale is float32 for sqrt operation if it comes from an integer context
    # and to maintain precision similar to C++ float
    current_scale = np.float32(scale)
    current_scale = np.sqrt(current_scale / np.float32(n))

    for i in range(n // 4):
        # Ensure intermediate calculations are also float32
        angle_val = omega * np.float32(i) + alpha
        tmp[2 * i] = current_scale * np.cos(angle_val)
        tmp[2 * i + 1] = current_scale * np.sin(angle_val)

    return tmp

class NMDCTBase:
    def __init__(self, n, l, scale):
        self.N = n
        self.Scale = scale  # Store scale for debugging
        self.buf = np.zeros(int(l), dtype=np.float32)  # Use 32-bit to match atracdenc
        self.SinCos = calc_sin_cos(n, scale)

class MDCT(NMDCTBase):
    def __init__(self, n, scale=None):
        if not scale:
            scale = n
        super().__init__(n, n // 2, scale)

    def __call__(self, input_signal: np.ndarray): # Renamed for clarity, ensure it's float32
        # Ensure input is float32, as C++ version uses float*
        input_signal = input_signal.astype(np.float32, copy=False)

        n2 = self.N // 2
        n4 = self.N // 4
        n34 = 3 * n4
        # n54 = 5 * n4 # n54 is used in the second loop for IMDCT, not MDCT in C++

        # C++ version:
        # const float* cos_ptr = &SinCos[0];
        # const float* sin_ptr = &SinCos[1];
        # Access pattern: cos_ptr[n], sin_ptr[n] where n increments by 2.
        # This means it accesses SinCos[0], SinCos[1] for n=0
        # SinCos[2], SinCos[3] for n=2 etc.
        # Python SinCos is already [c0,s0,c1,s1,...]
        # So cos_values[idx] from Python (where idx = k/2) corresponds to SinCos[k] in C++
        # and sin_values[idx] from Python corresponds to SinCos[k+1] in C++

        # The FFTIn buffer in C++ is of size N/4 complex pairs (N/2 floats)
        # kiss_fft_cpx* FFTIn; (allocated for N/4 complex numbers)
        # xr = (float*)FFTIn;
        # xi = (float*)FFTIn + 1;
        # xr[n], xi[n] in C++ means FFTIn[n/2].r and FFTIn[n/2].i if FFTIn was complex array
        # However, it's more like xr are even indices, xi are odd indices of a float array view
        # The loop for n goes up to n2 (N/2), with step 2.
        # So, xr[n] and xi[n] are accessed for n = 0, 2, ..., (N/2 - 2)
        # This means FFTIn has N/4 real parts and N/4 imaginary parts.
        # Python `real` and `imag` arrays are of size N/4 (size = n2 // 2 = (N/2)//2 = N/4)

        fft_in_real = np.zeros(n4, dtype=np.float32) # Equivalent to xr in C++ (N/4 elements)
        fft_in_imag = np.zeros(n4, dtype=np.float32) # Equivalent to xi in C++ (N/4 elements)

        # First loop - pre-rotation stage
        # C++ loop: for (n = 0; n < n4; n += 2)
        # Python: for idx, k in enumerate(range(0, n4, 2)):
        # Here k is equivalent to C++ n. idx is k/2.
        pre_rotation_debug = []
        for n_loop_var in range(0, n4, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for fft_in_real, fft_in_imag
            
            # C++: r0 = in[n34 - 1 - n] + in[n34 + n];
            # C++: i0 = in[n4 + n] - in[n4 - 1 - n];
            r0 = input_signal[n34 - 1 - n_loop_var] + input_signal[n34 + n_loop_var]
            i0 = input_signal[n4 + n_loop_var] - input_signal[n4 - 1 - n_loop_var]

            # C++: c = cos[n]; s = sin[n]; (where cos is &SinCos[0], sin is &SinCos[1])
            # This means c = SinCos[n_loop_var*2 / 2 * 2] = SinCos[n_loop_var] (if SinCos was [c0,s0,c1,s1...])
            # Correct C++ access: c = SinCos[n_loop_var], s = SinCos[n_loop_var + 1]
            # Python's SinCos is already [c0,s0, c2,s2, ... ] (scaled) for indices 0,1,2,3...
            # Python's original: cos_values = self.SinCos[0::2], sin_values = self.SinCos[1::2]
            # c = cos_values[idx] = self.SinCos[2*idx] = self.SinCos[n_loop_var]
            # s = sin_values[idx] = self.SinCos[2*idx+1] = self.SinCos[n_loop_var+1]
            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]

            # C++: xr[n] = r0 * c + i0 * s; xi[n] = i0 * c - r0 * s;
            # Python: fft_in_real[idx_fft], fft_in_imag[idx_fft]
            fft_in_real[idx_fft] = r0 * c + i0 * s
            fft_in_imag[idx_fft] = i0 * c - r0 * s

            if n_loop_var < 8: # Match C++ debug condition
                pre_rotation_debug.extend([r0, i0, c, s, fft_in_real[idx_fft], fft_in_imag[idx_fft]])
        
        debug_logger.log_stage("MDCT_SCALE_CURRENT", "value", [float(self.Scale)], 
                             frame=0, band="DEBUG", algorithm="mdct_internal", operation="current_scale", transform_size=self.N)
        debug_logger.log_stage("MDCT_PRE_ROTATION", "values", pre_rotation_debug[:24], 
                             frame=0, band="DEBUG", algorithm="mdct_internal", operation="pre_rotation")

        # Second loop
        # C++ loop: for (; n < n2; n += 2) (n continues from n4)
        # Python: for idx, k in enumerate(range(n4, n2, 2), start=size // 2):
        # k corresponds to C++ n. idx is k/2.
        for n_loop_var in range(n4, n2, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for fft_in_real, fft_in_imag

            # C++: r0 = in[n34 - 1 - n] - in[n - n4];
            # C++: i0 = in[n4 + n]    + in[n54 - 1 - n]; (n54 used here)
            # Python's original n54 was commented out, let's define it for MDCT too.
            n54 = 5 * n4 # This was missing in Python MDCT, present in IMDCT and C++ MDCT
            r0 = input_signal[n34 - 1 - n_loop_var] - input_signal[n_loop_var - n4]
            i0 = input_signal[n4 + n_loop_var] + input_signal[n54 - 1 - n_loop_var]

            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]

            fft_in_real[idx_fft] = r0 * c + i0 * s
            fft_in_imag[idx_fft] = i0 * c - r0 * s

        fft_input_debug = []
        for i in range(min(8, len(fft_in_real))): # Log N/4 complex numbers, so 8 pairs = 16 floats
            fft_input_debug.extend([fft_in_real[i], fft_in_imag[i]])
        debug_logger.log_stage("MDCT_FFT_INPUT", "complex", fft_input_debug, # C++ logs min(16, n2) floats -> min(8, n4) complex pairs
                             frame=0, band="DEBUG", algorithm="mdct_internal", operation="fft_input")

        # Perform FFT on N/4 complex samples
        # C++ uses kiss_fft(FFTPlan, FFTIn, FFTOut) where FFTPlan is for N/4 points.
        complex_input = fft_in_real + 1j * fft_in_imag # Array of N/4 complex numbers
        fft_result = np.fft.fft(complex_input) # Output is also N/4 complex numbers

        # C++: xr = (float*)FFTOut; xi = (float*)FFTOut + 1;
        # real_fft and imag_fft are arrays of N/4 floats each.
        real_fft_out = fft_result.real.astype(np.float32)
        imag_fft_out = fft_result.imag.astype(np.float32)
        
        fft_output_debug = []
        for i in range(min(8, len(real_fft_out))): # Log N/4 complex numbers
            fft_output_debug.extend([real_fft_out[i], imag_fft_out[i]])
        debug_logger.log_stage("MDCT_FFT_OUTPUT", "complex", fft_output_debug, # C++ logs min(16, n2) floats
                             frame=0, band="DEBUG", algorithm="mdct_internal", operation="fft_output")

        # Output - post-rotation stage
        # C++ loop: for (n = 0; n < n2; n += 2)
        # Python: for idx, k in enumerate(range(0, n2, 2)):
        # k corresponds to C++ n. idx is k/2.
        # Buf is size N/2
        post_rotation_debug = []
        for n_loop_var in range(0, n2, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for real_fft_out, imag_fft_out (which are N/4 long)
                                      # and also for cos/sin access from SinCos table (using n_loop_var)

            # C++: r0 = xr[n]; i0 = xi[n];
            # This implies xr and xi are views into FFTOut, which has N/4 complex (N/2 float) values.
            # xr[n] means FFTOut[n/2].r, xi[n] means FFTOut[n/2].i
            r0 = real_fft_out[idx_fft]
            i0 = imag_fft_out[idx_fft]

            # C++: c = cos[n]; s = sin[n];
            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]
            
            # C++: Buf[n] = - r0 * c - i0 * s;
            # C++: Buf[n2 - 1 -n] = - r0 * s + i0 * c;
            # Python self.buf is N/2. C++ Buf is N/2.
            # n_loop_var is the index for self.buf here.
            self.buf[n_loop_var] = -r0 * c - i0 * s
            self.buf[n2 - 1 - n_loop_var] = -r0 * s + i0 * c

            if n_loop_var < 8: # Match C++ debug condition
                post_rotation_debug.extend([r0, i0, c, s, self.buf[n_loop_var], self.buf[n2 - 1 - n_loop_var]])
        
        debug_logger.log_stage("MDCT_POST_ROTATION", "values", post_rotation_debug[:24], 
                             frame=0, band="DEBUG", algorithm="mdct_internal", operation="post_rotation")

        return self.buf.astype(np.float32) # Ensure output is float32

class IMDCT(NMDCTBase):
    def __init__(self, n, scale=None):
        if scale is None:
            scale = n # Default scale for IMDCT in C++ is TN (which is N here)
        # C++ TMIDCT constructor: TMDCTBase(TN, scale/2)
        # Python: super().__init__(n, n, scale / 2) -> N=n, L=n, base_scale = scale/2
        # This seems correct.
        super().__init__(n, n, np.float32(scale) / 2.0) # L=N for IMDCT output, ensure float division

    def __call__(self, input_coeffs: np.ndarray): # Renamed for clarity, ensure it's float32
        # Ensure input is float32
        input_coeffs = input_coeffs.astype(np.float32, copy=False)

        n2 = self.N // 2
        n4 = self.N // 4
        n34 = 3 * n4
        n54 = 5 * n4 # Used in the second output loop

        # SinCos access pattern similar to MDCT
        # const float* cos = &SinCos[0];
        # const float* sin = &SinCos[1];

        # FFTIn buffer in C++ is N/4 complex pairs (N/2 floats)
        # FFTPlan is for N/4 points.
        fft_in_real = np.zeros(n4, dtype=np.float32) # N/4 elements
        fft_in_imag = np.zeros(n4, dtype=np.float32) # N/4 elements

        # Prepare input for FFT (Pre-IFFT rotation)
        # C++ loop: for (n = 0; n < n2; n += 2)
        # Python: for idx, k in enumerate(range(0, n2, 2)):
        # k corresponds to C++ n. idx is k/2.
        for n_loop_var in range(0, n2, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for fft_in_real, fft_in_imag

            # C++: r0 = in[n]; i0 = in[n2 - 1 - n];
            r0 = input_coeffs[n_loop_var]
            i0 = input_coeffs[n2 - 1 - n_loop_var]

            # C++: c = cos[n]; s = sin[n];
            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]

            # C++: xr[n] = -2.0 * (i0 * s + r0 * c);
            # C++: xi[n] = -2.0 * (i0 * c - r0 * s);
            # xr[n] means FFTIn[n/2].r, xi[n] means FFTIn[n/2].i
            fft_in_real[idx_fft] = -2.0 * (i0 * s + r0 * c)
            fft_in_imag[idx_fft] = -2.0 * (i0 * c - r0 * s)

        # Perform FFT (actually IFFT, but kiss_fft can be configured for inverse)
        # np.fft.fft is used. The C++ code uses kiss_fft with is_inverse_fft=false.
        # The surrounding scaling and conjugation might handle the direct/inverse transform differences.
        # For now, assume np.fft.fft is the target, and ensure logic matches.
        complex_input = fft_in_real + 1j * fft_in_imag # Array of N/4 complex numbers
        fft_result = np.fft.fft(complex_input) # Output is also N/4 complex numbers

        # C++: xr = (float*)FFTOut; xi = (float*)FFTOut + 1;
        real_fft_out = fft_result.real.astype(np.float32)
        imag_fft_out = fft_result.imag.astype(np.float32)

        # Output stage (Post-IFFT rotation and reconstruction)
        # Buf is size N for IMDCT
        # First C++ loop: for (n = 0; n < n4; n += 2)
        for n_loop_var in range(0, n4, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for real_fft_out, imag_fft_out

            # C++: r0 = xr[n]; i0 = xi[n];
            r0 = real_fft_out[idx_fft]
            i0 = imag_fft_out[idx_fft]

            # C++: c = cos[n]; s = sin[n];
            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]

            r1 = r0 * c + i0 * s
            i1 = r0 * s - i0 * c # Note: C++ has i1 = r0*s - i0*c. Python was i0*s - r0*c. This needs to match.

            # C++ assignments:
            # Buf[n34 - 1 - n] = r1;
            # Buf[n34 + n] = r1;
            # Buf[n4 + n] = i1;
            # Buf[n4 - 1 - n] = -i1;
            self.buf[n34 - 1 - n_loop_var] = r1
            self.buf[n34 + n_loop_var] = r1
            self.buf[n4 + n_loop_var] = i1
            self.buf[n4 - 1 - n_loop_var] = -i1

        # Second C++ loop: for (; n < n2; n += 2) (n continues from n4)
        for n_loop_var in range(n4, n2, 2): # n_loop_var corresponds to 'n' in C++
            idx_fft = n_loop_var // 2 # index for real_fft_out, imag_fft_out

            # C++: r0 = xr[n]; i0 = xi[n];
            r0 = real_fft_out[idx_fft]
            i0 = imag_fft_out[idx_fft]

            # C++: c = cos[n]; s = sin[n];
            c = self.SinCos[n_loop_var]
            s = self.SinCos[n_loop_var + 1]

            r1 = r0 * c + i0 * s
            i1 = r0 * s - i0 * c # Match C++

            # C++ assignments:
            # Buf[n34 - 1 - n] = r1;
            # Buf[n - n4] = -r1;
            # Buf[n4 + n] = i1;
            # Buf[n54 - 1 - n] = i1;
            self.buf[n34 - 1 - n_loop_var] = r1
            self.buf[n_loop_var - n4] = -r1       # Corrected index from Python's k-n4
            self.buf[n4 + n_loop_var] = i1
            self.buf[n54 - 1 - n_loop_var] = i1

        return self.buf.astype(np.float32) # Ensure output is float32

        return self.buf

class BlockSizeMode:
    """
    Represents the adaptive windowing mode for MDCT based on transient detection.
    Controls whether long (256/512) or short (64) MDCT windows are used per band.
    """

    def __init__(
        self, low_band_short: bool, mid_band_short: bool, high_band_short: bool
    ):
        self.low_band_short = low_band_short
        self.mid_band_short = mid_band_short
        self.high_band_short = high_band_short
        
        # Add log_count for atracdenc compatibility
        # BSM encoding: 0=short blocks, 2/3=long blocks  
        # LogCount calculation: Low/Mid = 2-BSM, High = 3-BSM
        # Short blocks: BSM=0 → LogCount=2 (Low/Mid) or 3 (High) → 4/8 MDCT blocks
        # Long blocks: BSM=2/3 → LogCount=0 → 1 MDCT block
        self.log_count = [
            2 if low_band_short else 0,   # Low: 2-0=2 (short) or 2-2=0 (long)
            2 if mid_band_short else 0,   # Mid: 2-0=2 (short) or 2-2=0 (long)
            3 if high_band_short else 0   # High: 3-0=3 (short) or 3-3=0 (long)
        ]

    @property
    def low_mdct_size(self) -> int:
        return 64 if self.low_band_short else 128

    @property
    def mid_mdct_size(self) -> int:
        return 64 if self.mid_band_short else 128

    @property
    def high_mdct_size(self) -> int:
        return 64 if self.high_band_short else 256


class Atrac1MDCT:
    """
    Handles the Modified Discrete Cosine Transform (MDCT) and its inverse (IMDCT)
    for the ATRAC1 codec, including adaptive windowing and TDAC.
    Uses extracted atracdenc implementation for exact compatibility.
    """
    
    # ATRAC1 constants (from atracdenc)
    MAX_BFUS = 52
    NUM_QMF = 3
    SOUND_UNIT_SIZE = 212
    NUM_SAMPLES = 512

    def __init__(self):
        """
        Initializes the Atrac1MDCT instance with atracdenc-compatible MDCT engines.
        """
        # Initialize atracdenc-compatible MDCT/IMDCT engines with exact scaling
        self.mdct512 = MDCT(512, 1)
        self.mdct256 = MDCT(256, 0.5)
        self.mdct64 = MDCT(64, 0.5)
        # atracdenc IMDCT scaling: TMIDCT(TN*2) -> TMDCTBase(TN, TN)
        self.imdct512 = IMDCT(512, 512 * 2)  # scale=1024 -> base_scale=512 (matches atracdenc)
        self.imdct256 = IMDCT(256, 256 * 2)  # scale=512 -> base_scale=256 (matches atracdenc)  
        self.imdct64 = IMDCT(64, 64 * 2)     # scale=128 -> base_scale=64 (matches atracdenc)
        
        # Persistent windowing buffers to match atracdenc (per-channel, with overlap)
        # Format: [channel][buffer] with extra 16 samples for windowing overlap
        self.pcm_buf_low = [np.zeros(256 + 16, dtype=np.float32), 
                           np.zeros(256 + 16, dtype=np.float32)]
        self.pcm_buf_mid = [np.zeros(256 + 16, dtype=np.float32), 
                           np.zeros(256 + 16, dtype=np.float32)]
        self.pcm_buf_hi = [np.zeros(512 + 16, dtype=np.float32), 
                          np.zeros(512 + 16, dtype=np.float32)]
        
        # Persistent MDCT tmp buffers for frame-to-frame windowing state
        # Format: [channel][band] - each band needs its own tmp buffer state
        self.tmp_buffers = [
            [np.zeros(512, dtype=np.float32) for _ in range(3)],  # Channel 0: [LOW, MID, HIGH]
            [np.zeros(512, dtype=np.float32) for _ in range(3)]   # Channel 1: [LOW, MID, HIGH]
        ]
        
        
        # Initialize atracdenc-compatible sine window and store as float32 numpy array
        self.SINE_WINDOW = np.array([
            np.sin((i + 0.5) * (np.pi / (2.0 * 32.0))) for i in range(32)
        ], dtype=np.float32)
    
    def initialize_windowing_state(self, channel: int = 0):
        """
        Initialize MDCT windowing state to match atracdenc exactly.
        atracdenc constructor explicitly zeros all buffers after initialization.
        """
        # atracdenc explicitly zeros all buffers in constructor, so we start with zeros
        # All overlap regions should be zero-initialized
        self.pcm_buf_low[channel][128:] = 0.0
        self.pcm_buf_mid[channel][128:] = 0.0
        self.pcm_buf_hi[channel][256:] = 0.0
        
        # Fixed 32-sample sine window matching atracdenc exactly
        self._atrac1_sine_window = np.array([
            np.sin((i + 0.5) * np.pi / 64.0) for i in range(32)
        ], dtype=np.float32)
    
    def swap_array(self, array):
        """Use NumPy slicing to reverse the array in place"""
        array[:] = array[::-1]


    def mdct(
        self, specs: np.ndarray, low: np.ndarray, mid: np.ndarray, hi: np.ndarray, 
        block_size_mode: BlockSizeMode, channel: int = 0, frame: int = 0
    ):
        """
        Performs the forward MDCT operation using atracdenc-compatible implementation.
        
        Args:
            specs: Output spectral coefficients array (modified in-place)
            low: Low band QMF samples
            mid: Mid band QMF samples  
            hi: High band QMF samples
            block_size_mode: BlockSizeMode determining window types
            channel: Channel index for logging
            frame: Frame index for logging
        """
        # ATRACDENC ALIGNMENT: Zero out tmp buffers at start of each mdct() call
        for b_idx in range(self.NUM_QMF):
            self.tmp_buffers[channel][b_idx].fill(0.0)
        
        # Copy new QMF data into persistent buffers to maintain windowing state
        # This ensures frame-to-frame state persistence like atracdenc
        
        # Log QMF input data before copying to persistent buffers
        log_debug("MDCT_QMF_INPUT_RAW", "samples", low[:128].tolist(),
                  channel=channel, frame=frame, band="LOW", algorithm="buffer_tracking", operation="qmf_input")
        log_debug("MDCT_QMF_INPUT_RAW", "samples", mid[:128].tolist(),
                  channel=channel, frame=frame, band="MID", algorithm="buffer_tracking", operation="qmf_input")
        log_debug("MDCT_QMF_INPUT_RAW", "samples", hi[:256].tolist(),
                  channel=channel, frame=frame, band="HIGH", algorithm="buffer_tracking", operation="qmf_input")
        
        self.pcm_buf_low[channel][:128] = low[:128]
        self.pcm_buf_mid[channel][:128] = mid[:128] 
        self.pcm_buf_hi[channel][:256] = hi[:256]
        
        # Log persistent buffer state after copying
        log_debug("MDCT_PERSISTENT_AFTER_COPY", "samples", self.pcm_buf_low[channel][:128].tolist(),
                  channel=channel, frame=frame, band="LOW", algorithm="buffer_tracking", operation="after_copy")
        log_debug("MDCT_PERSISTENT_AFTER_COPY", "samples", self.pcm_buf_mid[channel][:128].tolist(),
                  channel=channel, frame=frame, band="MID", algorithm="buffer_tracking", operation="after_copy")
        log_debug("MDCT_PERSISTENT_AFTER_COPY", "samples", self.pcm_buf_hi[channel][:256].tolist(),
                  channel=channel, frame=frame, band="HIGH", algorithm="buffer_tracking", operation="after_copy")
        
        # Work with persistent buffers (not input arrays) for windowing state
        persistent_low = self.pcm_buf_low[channel]
        persistent_mid = self.pcm_buf_mid[channel]
        persistent_hi = self.pcm_buf_hi[channel]
        
        # Log input QMF bands to match atracdenc format (before processing)
        debug_logger.log_stage("MDCT_INPUT", "samples", persistent_low[:128], frame=frame, band="LOW", 
                             algorithm="mdct", mdct_size=128 if not block_size_mode.low_band_short else 64,
                             window_type="long" if not block_size_mode.low_band_short else "short")
        debug_logger.log_stage("MDCT_INPUT", "samples", persistent_mid[:128], frame=frame, band="MID",
                             algorithm="mdct", mdct_size=128 if not block_size_mode.mid_band_short else 64, 
                             window_type="long" if not block_size_mode.mid_band_short else "short")
        debug_logger.log_stage("MDCT_INPUT", "samples", persistent_hi[:256], frame=frame, band="HIGH",
                             algorithm="mdct", mdct_size=256 if not block_size_mode.high_band_short else 64,
                             window_type="long" if not block_size_mode.high_band_short else "short")
        
        # Log input QMF bands (only actual data portion, not zero-padding)
        log_debug("MDCT_INPUT_LOW", "samples", persistent_low[:128].tolist(),
                  channel=channel, frame=frame, band="LOW", algorithm="mdct")
        log_debug("MDCT_INPUT_MID", "samples", persistent_mid[:128].tolist(), 
                  channel=channel, frame=frame, band="MID", algorithm="mdct")
        log_debug("MDCT_INPUT_HIGH", "samples", persistent_hi[:256].tolist(),
                  channel=channel, frame=frame, band="HIGH", algorithm="mdct")
        
        pos = 0
        for band in range(self.NUM_QMF):
            num_mdct_blocks = 1 << block_size_mode.log_count[band]
            # Use persistent buffers instead of input arrays for windowing state
            src_buf = persistent_low if band == 0 else persistent_mid if band == 1 else persistent_hi
            buf_sz = 256 if band == 2 else 128
            block_sz = buf_sz if num_mdct_blocks == 1 else 32
            # Original atracdenc win_start values (restore for now to maintain TDAC)
            win_start = 112 if num_mdct_blocks == 1 and band == 2 else 48 if num_mdct_blocks == 1 else 0
            multiple = 2.0 if num_mdct_blocks != 1 and band == 2 else 1.0
            # Use persistent tmp buffer to maintain frame-to-frame windowing state
            tmp = self.tmp_buffers[channel][band]
            block_pos = 0
            
            band_name = ["LOW", "MID", "HIGH"][band]
            
            # Log srcBuf assignment to match atracdenc
            log_debug("MDCT_SRCBUF_ASSIGNED", "samples", src_buf[:16].tolist(),
                      channel=channel, frame=frame, band=band_name, algorithm="buffer_tracking", operation="srcbuf_assignment")
            
            # Log block size mode parameters  
            log_debug("MDCT_BAND_PARAMS", "params", [num_mdct_blocks, buf_sz, block_sz, win_start, multiple],
                      channel=channel, frame=frame, band=band_name, algorithm="mdct",
                      num_mdct_blocks=num_mdct_blocks, buf_sz=buf_sz, block_sz=block_sz)
            
            # Note: MDCT_INPUT logging removed to avoid confusion with MDCT_INPUT_LOW/MID/HIGH
            # The original input data is already logged by MDCT_INPUT_LOW/MID/HIGH above
            
            for k in range(num_mdct_blocks):
                # Ensure indices are within bounds (persistent buffers have +16 for overlap)
                assert buf_sz + 32 <= len(src_buf), f"src_buf too small for band {band}, buf_sz {buf_sz}, src_buf len {len(src_buf)}"
                
                # Log pre-windowing state
                debug_logger.log_stage("MDCT_PRE_WINDOW", f"BLK{k}", src_buf[block_pos:block_pos + block_sz][:16], frame=frame, band=band_name)
                debug_logger.log_stage("MDCT_WINDOW_FUNC", f"BLK{k}", self.SINE_WINDOW[:16], frame=frame, band=band_name)
                
                tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]

                # Store original values for logging
                orig_values = src_buf[block_pos:block_pos + block_sz].copy()
                
                for i in range(32):
                    idx1 = buf_sz + i
                    idx2 = block_pos + block_sz - 32 + i
                    src_buf[idx1] = self.SINE_WINDOW[i] * src_buf[idx2]
                    src_buf[idx2] = self.SINE_WINDOW[31 - i] * src_buf[idx2]

                # Log windowing operation details
                debug_logger.log_stage("MDCT_WINDOW_APPLIED", f"BLK{k}", src_buf[block_pos:block_pos + block_sz][:16], frame=frame, band=band_name)
                debug_logger.log_stage("MDCT_WINDOW_DIFF", f"BLK{k}", 
                                     (src_buf[block_pos:block_pos + block_sz] - orig_values)[:16], frame=frame, band=band_name)

                tmp[win_start + 32:win_start + 32 + block_sz] = src_buf[block_pos:block_pos + block_sz]

                # Log final windowed input with detailed metadata
                debug_logger.log_stage("MDCT_WINDOWED_FINAL", f"BLK{k}", tmp[:64], frame=frame, band=band_name)
                # Log windowed input to match atracdenc format exactly  
                window_size = 64 if num_mdct_blocks != 1 else (512 if band_name == "HIGH" else 256)
                actual_mdct_size = block_sz if num_mdct_blocks != 1 else buf_sz
                debug_logger.log_stage("MDCT_WINDOWED", "samples", tmp[:window_size], 
                                     frame=frame, band=band_name, algorithm='"mdct"', mdct_size=actual_mdct_size, window_type='"sine"', block=k)

                # Log input to transform
                debug_logger.log_stage("MDCT_TRANSFORM_INPUT", f"BLK{k}", tmp[:64], frame=frame, band=band_name)
                
                # Select the appropriate MDCT function
                if num_mdct_blocks == 1:
                    mdct_engine = "mdct512" if band == 2 else "mdct256"
                    sp = self.mdct512(tmp) if band == 2 else self.mdct256(tmp)
                else:
                    mdct_engine = "mdct64"
                    sp = self.mdct64(tmp)

                # Log raw MDCT output with transform details
                debug_logger.log_stage("MDCT_RAW_DCT", f"BLK{k}", sp[:32], frame=frame, band=band_name)
                debug_logger.log_stage("MDCT_TRANSFORM_INFO", f"BLK{k}", [len(tmp), len(sp)], frame=frame, band=band_name, mdct_engine=mdct_engine)
                log_debug("MDCT_RAW_OUTPUT", "coeffs", sp.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="mdct", block=k, mdct_engine=mdct_engine)

                # Multiply by the scaling factor
                specs_slice = sp * multiple
                specs_len = len(sp)
                
                # Log scaling details before assignment
                log_debug("MDCT_PRE_ASSIGNMENT", "details", [pos, block_pos, specs_len, multiple],
                          channel=channel, frame=frame, band=band_name, algorithm="mdct", 
                          block=k, operation="pre_assignment")
                
                # Log buffer slice before assignment
                log_debug("MDCT_BUFFER_BEFORE", "coeffs", specs[block_pos + pos:block_pos + pos + specs_len][:16].tolist(),
                          channel=channel, frame=frame, band=band_name, algorithm="mdct", 
                          block=k, operation="buffer_before_assignment")
                
                specs[block_pos + pos:block_pos + pos + specs_len] = specs_slice
                
                # Log buffer slice after assignment
                log_debug("MDCT_BUFFER_AFTER", "coeffs", specs[block_pos + pos:block_pos + pos + specs_len][:16].tolist(),
                          channel=channel, frame=frame, band=band_name, algorithm="mdct", 
                          block=k, operation="buffer_after_assignment")
                
                # Log MDCT_BLOCK_OUTPUT to match atracdenc (full block size)
                debug_logger.log_stage("MDCT_BLOCK_OUTPUT", "coeffs", specs_slice, 
                                     frame=frame, band=band_name, algorithm="mdct", 
                                     block=k)

                # Log after scaling
                log_debug("MDCT_AFTER_SCALE", "coeffs", specs_slice.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="mdct", block=k, scale_factor=multiple)
                
                # Log MDCT_AFTER_LEVEL to match atracdenc
                debug_logger.log_stage("MDCT_AFTER_LEVEL", "coeffs", specs_slice, 
                                     frame=frame, band=band_name, algorithm="mdct", 
                                     block=k, level_applied=multiple)

                # Swap array if band is not zero
                if band:
                    self.swap_array(specs[block_pos + pos:block_pos + pos + specs_len])
                    
                    # Log after swap
                    log_debug("MDCT_AFTER_SWAP", "coeffs", 
                              specs[block_pos + pos:block_pos + pos + specs_len].tolist(),
                              channel=channel, frame=frame, band=band_name,
                              algorithm="mdct", block=k, operation="swap_array")

                block_pos += 32
            pos += buf_sz
            
            # Log MDCT_OUTPUT for entire band to match atracdenc
            debug_logger.log_stage("MDCT_OUTPUT", "coeffs", specs[pos-buf_sz:pos][:16], 
                                 frame=frame, band=band_name, algorithm="mdct", 
                                 band_complete=True, total_coeffs=buf_sz)
        
        # Log final spectral coefficients
        log_debug("MDCT_FINAL_OUTPUT", "coeffs", specs.tolist(),
                  channel=channel, frame=frame, algorithm="mdct", processing_stage="final")


    def imdct(
        self, specs: np.ndarray, mode, low: np.ndarray, mid: np.ndarray, hi: np.ndarray,
        channel: int = 0, frame: int = 0
    ):
        """
        Performs the inverse MDCT operation using atracdenc-compatible implementation.
        
        Args:
            specs: Input spectral coefficients array
            mode: BlockSizeMode determining window types
            low: Output low band QMF samples (modified in-place)
            mid: Output mid band QMF samples (modified in-place)
            hi: Output high band QMF samples (modified in-place)
            channel: Channel index for logging
            frame: Frame index for logging
        """
        # Log input spectral coefficients
        log_debug("IMDCT_INPUT_SPECS", "coeffs", specs.tolist(),
                  channel=channel, frame=frame, algorithm="imdct")
        
        pos = 0
        for band in range(self.NUM_QMF):
            num_mdct_blocks = 1 << mode.log_count[band]
            buf_sz = 256 if band == 2 else 128
            block_sz = buf_sz if num_mdct_blocks == 1 else 32
            # ATRACDENC ALIGNMENT: Use persistent buffers for TDAC/stateful operations
            dst_buf = self.pcm_buf_low[channel] if band == 0 else self.pcm_buf_mid[channel] if band == 1 else self.pcm_buf_hi[channel]
            # ATRACDENC ALIGNMENT: Use np.float32 for inv_buf
            max_block_size = buf_sz if num_mdct_blocks == 1 else 32
            inv_buf = np.zeros(2 * max_block_size * num_mdct_blocks, dtype=np.float32)
            prev_buf = dst_buf[buf_sz * 2 - 16:]  # Use persistent buffer overlap region
            start = 0
            
            band_name = ["LOW", "MID", "HIGH"][band]
            
            # Log IMDCT band parameters
            log_debug("IMDCT_BAND_PARAMS", "params", [num_mdct_blocks, buf_sz, block_sz],
                      channel=channel, frame=frame, band=band_name, algorithm="imdct",
                      num_mdct_blocks=num_mdct_blocks, buf_sz=buf_sz, block_sz=block_sz)
            
            for block in range(num_mdct_blocks):
                specs_len = block_sz
                
                # Log input coefficients for this block
                log_debug("IMDCT_BLOCK_INPUT", "coeffs", specs[pos:pos + specs_len].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block)
                
                # Swap array if band is not zero
                if band:
                    self.swap_array(specs[pos:pos + specs_len])
                    
                    # Log after swap
                    log_debug("IMDCT_AFTER_SWAP", "coeffs", specs[pos:pos + specs_len].tolist(),
                              channel=channel, frame=frame, band=band_name,
                              algorithm="imdct", block=block, operation="swap_array")

                # Select the appropriate IMDCT function
                if num_mdct_blocks != 1:
                    inv = self.imdct64(specs[pos:pos + specs_len])
                elif buf_sz == 128:
                    inv = self.imdct256(specs[pos:pos + buf_sz])
                else:
                    inv = self.imdct512(specs[pos:pos + buf_sz])

                # Log raw IMDCT output
                log_debug("IMDCT_RAW_OUTPUT", "samples", inv.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block, imdct_engine=f"imdct{len(inv)}")

                inv_len = len(inv)
                # atracdenc CRITICAL: Only store middle half of IMDCT output (lines 269-271)
                # for (size_t i = 0; i < (inv.size()/2); i++) {
                #     invBuf[start+i] = inv[i + inv.size()/4];
                # }
                # EXACT ATRACDENC MATCH: Extract inv[i + inv.size()/4] for i=0..inv.size()/2-1
                middle_start = inv_len // 4        # 64 (atracdenc: inv.size()/4)
                middle_length = inv_len // 2       # 128 (atracdenc: inv.size()/2)
                inv_buf[start:start + middle_length] = inv[middle_start:middle_start + middle_length]

                # Log windowing input buffers
                log_debug("IMDCT_WINDOW_INPUTS", "prev_buf", prev_buf.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block, operation="vector_fmul_window_prep")
                log_debug("IMDCT_WINDOW_INPUTS", "inv_buf_segment", inv_buf[start:start + inv_len // 2].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block, operation="vector_fmul_window_prep")

                # Ensure dst_buf has enough space
                dst_len_required = start + 2 * 16  # Since vector_fmul_window accesses up to index [2*len - 1]
                assert len(dst_buf) >= dst_len_required, "dst_buf too small"

                # CRITICAL: atracdenc uses pointer arithmetic - we need to handle indexing correctly
                # vector_fmul_window(dstBuf + start, prevBuf, &invBuf[start], &TAtrac1Data::SineWindow[0], 16);
                
                # Create properly sized temporary arrays for vector_fmul_window
                temp_dst = np.zeros(32, dtype=np.float32)  # 2 * 16 = 32
                temp_src1 = inv_buf[start:start + 16]
                
                # Pass the pre-converted SINE_WINDOW numpy array
                vector_fmul_window(
                    temp_dst,  # dst (will use indices 0 to 31 in Python version)
                    prev_buf,  # src0
                    temp_src1,  # src1
                    self.SINE_WINDOW, # Already a np.float32 array
                    16
                )
                
                # Copy result back to destination buffer
                dst_buf[start:start + 32] = temp_dst

                # Log windowing output
                log_debug("IMDCT_WINDOW_OUTPUT", "samples", dst_buf[start:start + 32].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block, operation="vector_fmul_window_output")

                # Update prev_buf for next iteration (atracdenc line 275)
                # prevBuf = &invBuf[start+16];
                prev_buf = inv_buf[start + 16:start + 16 + 16]  # 16 samples starting at start+16
                start += block_sz
                pos += block_sz

            if num_mdct_blocks == 1:
                length = 240 if band == 2 else 112
                # Copy middle non-overlapped portion from raw IMDCT output (atracdenc line 90)
                # memcpy(dstBuf + 32, &invBuf[16], ((band == 2) ? 240 : 112) * sizeof(float));
                dst_buf[32:32 + length] = inv_buf[16:16 + length]
                
                # Log long block copy
                log_debug("IMDCT_LONG_COPY", "samples", dst_buf[32:32 + length].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", operation="long_block_copy", length=length)

            # Store current tail for next frame (atracdenc lines 92-93)
            # for (size_t j = 0; j < 16; j++) { dstBuf[bufSz*2 - 16 + j] = invBuf[bufSz - 16 + j]; }
            for j in range(16):
                dst_buf[buf_sz * 2 - 16 + j] = inv_buf[buf_sz - 16 + j]
            
            # Log final output for this band
            log_debug("IMDCT_BAND_OUTPUT", "samples", dst_buf.tolist(),
                      channel=channel, frame=frame, band=band_name,
                      algorithm="imdct", processing_stage="final")
        
        # ATRACDENC ALIGNMENT: Copy from persistent buffers to output arguments at the end
        # Copy only the main data portion, not the overlap regions
        low[:] = self.pcm_buf_low[channel][:len(low)]
        mid[:] = self.pcm_buf_mid[channel][:len(mid)]
        hi[:] = self.pcm_buf_hi[channel][:len(hi)]
