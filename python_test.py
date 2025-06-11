import numpy as np
from pyatrac1.core.qmf import Atrac1AnalysisFilterBank, Atrac1SynthesisFilterBank
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode # Import BlockSizeMode
from pyatrac1.core.codec_data import Atrac1CodecData

# Helper function to write a numpy array to a file
def write_array_to_file(data, filename):
    with open(filename, 'w') as f:
        if hasattr(data, '__iter__'):
            for val in data:
                f.write(f"{val:.8f}\n")
        else:
            f.write(f"{data:.8f}\n")
    print(f"Data written to {filename}")

def main():
    print("Starting Python test for ATRAC QMF and MDCT...")

    try:
        input_samples = np.loadtxt("cpp_input_samples.txt", dtype=np.float32)
    except FileNotFoundError:
        print("Error: cpp_input_samples.txt not found. Please run the C++ test first.")
        return

    num_samples = len(input_samples)
    print(f"Loaded {num_samples} input samples.")

    codec_data = Atrac1CodecData()

    # Define BlockSizeMode for short blocks across all bands
    # This aligns with the requirement "use BLOCK_SIZE_SHORT for all subbands"
    # by making all bands use their respective "short" mode.
    short_block_mode = BlockSizeMode(low_band_short=True, mid_band_short=True, high_band_short=True)
    print(f"Using MDCT BlockSizeMode: low_short={short_block_mode.low_band_short}, mid_short={short_block_mode.mid_band_short}, high_short={short_block_mode.high_band_short}")

    print("\n--- Performing Filter Bank Analysis (QMF) ---")
    analysis_qmf = Atrac1AnalysisFilterBank()

    qmf_frame_size = 512
    num_qmf_frames = num_samples // qmf_frame_size
    QMF_FRAME_OUT_LOW_MID = 128
    QMF_FRAME_OUT_HIGH = 256

    # Store QMF outputs per frame for MDCT processing
    qmf_frames_low = []
    qmf_frames_mid = []
    qmf_frames_high = []

    for i in range(num_qmf_frames):
        frame_data = input_samples[i*qmf_frame_size : (i+1)*qmf_frame_size]
        low, mid, high = analysis_qmf.analysis(frame_data.tolist(), frame_num=i)
        qmf_frames_low.append(np.array(low, dtype=np.float32))
        qmf_frames_mid.append(np.array(mid, dtype=np.float32))
        qmf_frames_high.append(np.array(high, dtype=np.float32))

    # For writing to file, concatenate all frames for each subband
    qmf_subband_0_concat = np.concatenate(qmf_frames_low)
    qmf_subband_1_concat = np.concatenate(qmf_frames_mid)
    qmf_subband_2_concat = np.concatenate(qmf_frames_high)

    print(f"QMF analysis complete.")
    print(f"Subband 0 (Low) total size: {len(qmf_subband_0_concat)}")
    print(f"Subband 1 (Mid) total size: {len(qmf_subband_1_concat)}")
    print(f"Subband 2 (High) total size: {len(qmf_subband_2_concat)}")

    write_array_to_file(qmf_subband_0_concat, "py_qmf_subband_0.txt")
    write_array_to_file(qmf_subband_1_concat, "py_qmf_subband_1.txt")
    write_array_to_file(qmf_subband_2_concat, "py_qmf_subband_2.txt")

    print("\n--- Performing MDCT ---")
    atrac_mdct = Atrac1MDCT()
    atrac_mdct.initialize_windowing_state(channel=0) # Initialize windowing state for channel 0

    # Total number of spectral coefficients per frame = 128 (low) + 128 (mid) + 256 (high) = 512
    # This is based on the buf_sz passed to mdct calls in pyatrac1 for long blocks.
    # For short blocks, the actual number of coefficients might differ based on BlockSizeMode logic.
    # The `specs` array in `Atrac1MDCT.mdct` is accumulated.
    # Low band MDCT (short=64coeffs, long=128coeffs)
    # Mid band MDCT (short=64coeffs, long=128coeffs)
    # High band MDCT (short=32coeffs, long=256coeffs)
    # If short_block_mode is True for all: 64+64+32 = 160 coefficients if BlockSizeMode is strictly followed for output length.
    # However, the `specs` array in atracdenc `TAtrac1::DoMdct` has fixed band offsets.
    # Let's assume `specs` array needs to be 512 (max possible size for a frame).
    # The `Atrac1MDCT.mdct` method in `pyatrac1` appears to fill this based on `pos += buf_sz`.
    # `buf_sz` refers to QMF output sizes (128, 128, 256). So total size is 512.

    mdct_coeffs_frames = []
    SPECS_FRAME_SIZE = QMF_FRAME_OUT_LOW_MID + QMF_FRAME_OUT_LOW_MID + QMF_FRAME_OUT_HIGH # 128+128+256 = 512

    for i in range(num_qmf_frames):
        frame_specs = np.zeros(SPECS_FRAME_SIZE, dtype=np.float32)
        # The Atrac1MDCT.mdct method modifies frame_specs in-place
        atrac_mdct.mdct(
            specs=frame_specs,
            low=qmf_frames_low[i],
            mid=qmf_frames_mid[i],
            hi=qmf_frames_high[i],
            block_size_mode=short_block_mode,
            channel=0,
            frame=i
        )
        mdct_coeffs_frames.append(frame_specs)

    mdct_coeffs_concat = np.concatenate(mdct_coeffs_frames)

    # Split concatenated MDCT coeffs for writing, assuming fixed sizes based on QMF output sizes
    # This matches how atracdenc structures its spectral data before quantization.
    # Offset for subband 1 is 128 (size of subband 0 coefficients if long block)
    # Offset for subband 2 is 128+128 = 256
    # This splitting is an interpretation, as `frame_specs` is a flat array.
    # For comparison with C++ which used BLOCK_SIZE_SHORT=128 for all, this part may need care.
    # The pyatrac1 mdct method uses internal block sizes (e.g. 64 for high-short).
    # The `specs` array is filled according to QMF band sizes (128, 128, 256).

    mdct_out_0 = []
    mdct_out_1 = []
    mdct_out_2 = []
    for frame_spec in mdct_coeffs_frames:
        mdct_out_0.extend(frame_spec[0 : QMF_FRAME_OUT_LOW_MID])
        mdct_out_1.extend(frame_spec[QMF_FRAME_OUT_LOW_MID : QMF_FRAME_OUT_LOW_MID+QMF_FRAME_OUT_LOW_MID])
        mdct_out_2.extend(frame_spec[QMF_FRAME_OUT_LOW_MID+QMF_FRAME_OUT_LOW_MID : SPECS_FRAME_SIZE])

    print("MDCT processing complete.")
    write_array_to_file(np.array(mdct_out_0, dtype=np.float32), "py_mdct_coeffs_0.txt")
    write_array_to_file(np.array(mdct_out_1, dtype=np.float32), "py_mdct_coeffs_1.txt")
    write_array_to_file(np.array(mdct_out_2, dtype=np.float32), "py_mdct_coeffs_2.txt")

    print("\n--- Performing IMDCT ---")
    imdct_frames_low = []
    imdct_frames_mid = []
    imdct_frames_high = []

    for i in range(num_qmf_frames):
        frame_specs = mdct_coeffs_frames[i]
        out_low_frame = np.zeros(QMF_FRAME_OUT_LOW_MID, dtype=np.float32)
        out_mid_frame = np.zeros(QMF_FRAME_OUT_LOW_MID, dtype=np.float32)
        out_high_frame = np.zeros(QMF_FRAME_OUT_HIGH, dtype=np.float32)

        atrac_mdct.imdct(
            specs=frame_specs,
            mode=short_block_mode,
            low=out_low_frame,
            mid=out_mid_frame,
            hi=out_high_frame,
            channel=0,
            frame=i
        )
        imdct_frames_low.append(out_low_frame)
        imdct_frames_mid.append(out_mid_frame)
        imdct_frames_high.append(out_high_frame)

    imdct_subband_0_concat = np.concatenate(imdct_frames_low)
    imdct_subband_1_concat = np.concatenate(imdct_frames_mid)
    imdct_subband_2_concat = np.concatenate(imdct_frames_high)

    print("IMDCT processing complete.")
    write_array_to_file(imdct_subband_0_concat, "py_imdct_output_0.txt")
    write_array_to_file(imdct_subband_1_concat, "py_imdct_output_1.txt")
    write_array_to_file(imdct_subband_2_concat, "py_imdct_output_2.txt")

    print("\n--- Performing Filter Bank Synthesis (IQMF) ---")
    synthesis_qmf = Atrac1SynthesisFilterBank()
    reconstructed_output_frames = []

    num_iqmf_frames = len(imdct_subband_0_concat) // QMF_FRAME_OUT_LOW_MID

    for i in range(num_iqmf_frames):
        low_frame_np = imdct_subband_0_concat[i*QMF_FRAME_OUT_LOW_MID : (i+1)*QMF_FRAME_OUT_LOW_MID]
        mid_frame_np = imdct_subband_1_concat[i*QMF_FRAME_OUT_LOW_MID : (i+1)*QMF_FRAME_OUT_LOW_MID]
        high_frame_np = imdct_subband_2_concat[i*QMF_FRAME_OUT_HIGH : (i+1)*QMF_FRAME_OUT_HIGH]

        reconstructed_frame = synthesis_qmf.synthesis(low_frame_np.tolist(),
                                                      mid_frame_np.tolist(),
                                                      high_frame_np.tolist())
        reconstructed_output_frames.extend(reconstructed_frame)

    reconstructed_output = np.array(reconstructed_output_frames, dtype=np.float32)

    print("QMF synthesis complete.")
    print(f"Reconstructed output size: {len(reconstructed_output)}")
    if len(reconstructed_output) != num_samples:
        print(f"Warning: Reconstructed output size ({len(reconstructed_output)}) does not match original input size ({num_samples})")

    write_array_to_file(reconstructed_output, "py_reconstructed_output.txt")

    print("\nPython test finished.")

if __name__ == "__main__":
    main()
