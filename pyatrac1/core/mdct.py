"""
Implements Modified Discrete Cosine Transform (MDCT) and Inverse MDCT (IMDCT)
for ATRAC1, including adaptive windowing and Time-Domain Aliasing Cancellation (TDAC).
Uses extracted atracdenc MDCT implementation for maximum compatibility.
"""

import math
import numpy as np

from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.common.utils import swap_array
from pyatrac1.common.debug_logger import log_debug


# Extracted atracdenc MDCT implementation
def vector_fmul_window(dst: np.ndarray, src0: np.ndarray, src1: np.ndarray, win: np.ndarray, length: int):
    j = length - 1
    for i in range(-length, 0):
        s0 = src0[i]
        s1 = src1[j]
        wi = win[i]
        wj = win[j]
        dst[i] = s0 * wj - s1 * wi
        dst[j] = s0 * wi + s1 * wj
        j -= 1

def calc_sin_cos(n, scale):
    tmp = np.zeros(n // 2, dtype=np.float64)
    alpha = 2.0 * math.pi / (8.0 * n)
    omega = 2.0 * math.pi / n
    scale = np.sqrt(scale / n)

    for i in range(n // 4):
        tmp[2 * i] = scale * np.cos(omega * i + alpha)
        tmp[2 * i + 1] = scale * np.sin(omega * i + alpha)

    return tmp

class NMDCTBase:
    def __init__(self, n, l, scale):
        self.N = n
        self.buf = np.zeros(int(l), dtype=np.float64)
        self.SinCos = calc_sin_cos(n, scale)

class MDCT(NMDCTBase):
    def __init__(self, n, scale=None):
        if not scale:
            scale = n
        super().__init__(n, n // 2, scale)

    def __call__(self, input):
        n2 = self.N // 2
        n4 = self.N // 4
        n34 = 3 * n4
        n54 = 5 * n4
        cos_values = self.SinCos[0::2]
        sin_values = self.SinCos[1::2]
        size = n2 // 2
        real = np.zeros(size, dtype=np.float64)
        imag = np.zeros(size, dtype=np.float64)

        # First loop
        for idx, k in enumerate(range(0, n4, 2)):
            r0 = input[n34 - 1 - k] + input[n34 + k]
            i0 = input[n4 + k] - input[n4 - 1 - k]
            c = cos_values[idx]
            s = sin_values[idx]
            real[idx] = r0 * c + i0 * s
            imag[idx] = i0 * c - r0 * s

        # Second loop
        for idx, k in enumerate(range(n4, n2, 2), start=size // 2):
            r0 = input[n34 - 1 - k] - input[k - n4]
            i0 = input[n4 + k] + input[n54 - 1 - k]
            c = cos_values[idx]
            s = sin_values[idx]
            real[idx] = r0 * c + i0 * s
            imag[idx] = i0 * c - r0 * s

        # Perform FFT
        complex_input = real + 1j * imag
        fft_result = np.fft.fft(complex_input)
        real_fft = fft_result.real
        imag_fft = fft_result.imag

        # Output
        for idx, k in enumerate(range(0, n2, 2)):
            r0 = real_fft[idx]
            i0 = imag_fft[idx]
            c = cos_values[idx]
            s = sin_values[idx]
            self.buf[k] = -r0 * c - i0 * s
            self.buf[n2 - 1 - k] = -r0 * s + i0 * c

        return self.buf

class IMDCT(NMDCTBase):
    def __init__(self, n, scale=None):
        if scale is None:
            scale = n
        super().__init__(n, n, scale / 2)

    def __call__(self, input):
        n2 = self.N // 2
        n4 = self.N // 4
        n34 = 3 * n4
        n54 = 5 * n4

        cos_values = self.SinCos[0::2]
        sin_values = self.SinCos[1::2]

        size = n2 // 2
        real = np.zeros(size, dtype=np.float64)
        imag = np.zeros(size, dtype=np.float64)

        # Prepare input for FFT
        for idx, k in enumerate(range(0, n2, 2)):
            r0 = input[k]
            i0 = input[n2 - 1 - k]
            c = cos_values[idx]
            s = sin_values[idx]
            real[idx] = -2.0 * (i0 * s + r0 * c)
            imag[idx] = -2.0 * (i0 * c - r0 * s)

        # Perform FFT
        complex_input = real + 1j * imag
        fft_result = np.fft.fft(complex_input)
        real_fft = fft_result.real
        imag_fft = fft_result.imag

        # Output
        for idx, k in enumerate(range(0, n4, 2)):
            r0 = real_fft[idx]
            i0 = imag_fft[idx]
            c = cos_values[idx]
            s = sin_values[idx]
            r1 = r0 * c + i0 * s
            i1 = r0 * s - i0 * c
            self.buf[n34 - 1 - k] = r1
            self.buf[n34 + k] = r1
            self.buf[n4 + k] = i1
            self.buf[n4 - 1 - k] = -i1

        for idx, k in enumerate(range(n4, n2, 2), start=size // 2):
            r0 = real_fft[idx]
            i0 = imag_fft[idx]
            c = cos_values[idx]
            s = sin_values[idx]
            r1 = r0 * c + i0 * s
            i1 = r0 * s - i0 * c
            self.buf[n34 - 1 - k] = r1
            self.buf[k - n4] = -r1
            self.buf[n4 + k] = i1
            self.buf[n54 - 1 - k] = i1

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
        self.log_count = [
            1 if low_band_short else 0,
            1 if mid_band_short else 0, 
            1 if high_band_short else 0
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
        # Initialize atracdenc MDCT/IMDCT engines with exact scaling
        self.mdct512 = MDCT(512, 1)
        self.mdct256 = MDCT(256, 0.5)
        self.mdct64 = MDCT(64, 0.5)
        self.imdct512 = IMDCT(512, 512 * 2)
        self.imdct256 = IMDCT(256, 256 * 2)
        self.imdct64 = IMDCT(64, 64 * 2)
        
        # Initialize atracdenc-compatible sine window
        self.SINE_WINDOW = [0.0] * 32
        for i in range(32):
            self.SINE_WINDOW[i] = np.sin((i + 0.5) * (np.pi / (2.0 * 32.0)))
        
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
        # Log input QMF bands
        log_debug("MDCT_INPUT_LOW", "samples", low.tolist(),
                  channel=channel, frame=frame, band="LOW", algorithm="mdct")
        log_debug("MDCT_INPUT_MID", "samples", mid.tolist(), 
                  channel=channel, frame=frame, band="MID", algorithm="mdct")
        log_debug("MDCT_INPUT_HIGH", "samples", hi.tolist(),
                  channel=channel, frame=frame, band="HIGH", algorithm="mdct")
        
        pos = 0
        for band in range(self.NUM_QMF):
            num_mdct_blocks = 1 << block_size_mode.log_count[band]
            src_buf = low if band == 0 else mid if band == 1 else hi
            buf_sz = 256 if band == 2 else 128
            block_sz = buf_sz if num_mdct_blocks == 1 else 32
            win_start = 112 if num_mdct_blocks == 1 and band == 2 else 48 if num_mdct_blocks == 1 else 0
            multiple = 2.0 if num_mdct_blocks != 1 and band == 2 else 1.0
            tmp = np.zeros(512, dtype=np.float64)
            block_pos = 0
            
            band_name = ["LOW", "MID", "HIGH"][band]
            
            # Log block size mode parameters
            log_debug("MDCT_BAND_PARAMS", "params", [num_mdct_blocks, buf_sz, block_sz, win_start, multiple],
                      channel=channel, frame=frame, band=band_name, algorithm="mdct",
                      num_mdct_blocks=num_mdct_blocks, buf_sz=buf_sz, block_sz=block_sz)
            
            for k in range(num_mdct_blocks):
                # Ensure indices are within bounds
                assert buf_sz + 32 <= len(src_buf), f"src_buf too small for band {band}, buf_sz {buf_sz}"
                tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]

                for i in range(32):
                    idx1 = buf_sz + i
                    idx2 = block_pos + block_sz - 32 + i
                    src_buf[idx1] = self.SINE_WINDOW[i] * src_buf[idx2]
                    src_buf[idx2] = self.SINE_WINDOW[31 - i] * src_buf[idx2]

                tmp[win_start + 32:win_start + 32 + block_sz] = src_buf[block_pos:block_pos + block_sz]

                # Log windowed input
                log_debug("MDCT_WINDOWED", "samples", tmp.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="mdct", block=k, window_type="atracdenc_sine")

                # Select the appropriate MDCT function
                if num_mdct_blocks == 1:
                    sp = self.mdct512(tmp) if band == 2 else self.mdct256(tmp)
                else:
                    sp = self.mdct64(tmp)

                # Log raw MDCT output
                log_debug("MDCT_RAW_OUTPUT", "coeffs", sp.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="mdct", block=k, mdct_engine=f"mdct{len(sp)}")

                # Multiply by the scaling factor
                specs_slice = sp * multiple
                specs_len = len(sp)
                specs[block_pos + pos:block_pos + pos + specs_len] = specs_slice

                # Log after scaling
                log_debug("MDCT_AFTER_SCALE", "coeffs", specs_slice.tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="mdct", block=k, scale_factor=multiple)

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
            dst_buf = low if band == 0 else mid if band == 1 else hi
            inv_buf = np.zeros(512, dtype=np.float64)
            prev_buf = dst_buf[buf_sz * 2 - 16:]
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
                # Copy the middle half of the inverse MDCT output
                mid_inv = inv[inv_len // 4: inv_len // 4 + inv_len // 2]
                inv_buf[start:start + inv_len // 2] = mid_inv

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

                vector_fmul_window(
                    dst_buf[start:],  # dst
                    prev_buf,  # src0
                    inv_buf[start:],  # src1
                    np.array(self.SINE_WINDOW, dtype=np.float64),
                    16
                )

                # Log windowing output
                log_debug("IMDCT_WINDOW_OUTPUT", "samples", dst_buf[start:start + 32].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", block=block, operation="vector_fmul_window_output")

                # Update prev_buf for the next iteration
                prev_buf = inv_buf[start + 16:start + inv_len // 2]
                start += block_sz
                pos += block_sz

            if num_mdct_blocks == 1:
                length = 240 if band == 2 else 112
                dst_buf[32:32 + length] = inv_buf[16:16 + length]
                
                # Log long block copy
                log_debug("IMDCT_LONG_COPY", "samples", dst_buf[32:32 + length].tolist(),
                          channel=channel, frame=frame, band=band_name,
                          algorithm="imdct", operation="long_block_copy", length=length)

            # Copy the last 16 samples
            dst_buf[buf_sz * 2 - 16:buf_sz * 2] = inv_buf[buf_sz - 16:buf_sz]
            
            # Log final output for this band
            log_debug("IMDCT_BAND_OUTPUT", "samples", dst_buf.tolist(),
                      channel=channel, frame=frame, band=band_name,
                      algorithm="imdct", processing_stage="final")
