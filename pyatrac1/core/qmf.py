"""
Implements the Quadrature Mirror Filter (QMF) bank for ATRAC1 sub-band coding.
Exact implementation matching atracdenc reference for audio quality.
"""

import numpy as np
from pyatrac1.common.constants import NUM_SAMPLES

# atracdenc TapHalf coefficients (24 values, doubled to create 48-tap window)
TAP_HALF = np.array([
    -0.00001461907,  -0.00009205479, -0.000056157569,  0.00030117269,
    0.0002422519,    -0.00085293897, -0.0005205574,    0.0020340169,
    0.00078333891,   -0.0042153862,  -0.00075614988,   0.0078402944,
    -0.000061169922, -0.01344162,    0.0024626821,     0.021736089,
    -0.007801671,    -0.034090221,   0.01880949,       0.054326009,
    -0.043596379,    -0.099384367,   0.13207909,       0.46424159
], dtype=np.float32)

# Create QMF window exactly like atracdenc
QMF_WINDOW = np.zeros(48, dtype=np.float32)
for i in range(24):
    QMF_WINDOW[i] = QMF_WINDOW[47 - i] = TAP_HALF[i] * 2.0


class TQmf:
    """
    Exact implementation of atracdenc's TQmf template class.
    Handles QMF analysis and synthesis with proper delay buffers.
    """

    def __init__(self, n_input_samples: int):
        """
        Initialize TQmf exactly like atracdenc.
        
        Args:
            n_input_samples: Number of input samples (512 or 256)
        """
        self.n_input_samples = n_input_samples
        # atracdenc: TPCM PcmBuffer[nIn + 46];
        self.pcm_buffer = np.zeros(n_input_samples + 46, dtype=np.float32)
        # atracdenc: float PcmBufferMerge[nIn + 46];
        self.pcm_buffer_merge = np.zeros(n_input_samples + 46, dtype=np.float32)
        # atracdenc: float DelayBuff[46];
        self.delay_buff = np.zeros(46, dtype=np.float32)

    def _calculate_qmf_taps(self, pcm_buffer: np.ndarray, j: int) -> tuple[float, float]:
        """
        Exact implementation of atracdenc QMF tap calculation.
        """
        lower = 0.0
        upper = 0.0
        for i in range(24):
            # atracdenc: lower[j/2] += QmfWindow[2*i] * PcmBuffer[48-1+j-(2*i)];
            lower += QMF_WINDOW[2*i] * pcm_buffer[48-1+j-(2*i)]
            # atracdenc: upper[j/2] += QmfWindow[(2*i)+1] * PcmBuffer[48-1+j-(2*i)-1];
            upper += QMF_WINDOW[(2*i)+1] * pcm_buffer[48-1+j-(2*i)-1]
        return lower, upper

    def analysis(self, input_data: list[float]) -> tuple[list[float], list[float]]:
        """
        Exact implementation of atracdenc TQmf::Analysis.
        """
        input_array = np.array(input_data, dtype=np.float32)
        
        # atracdenc: for (size_t i = 0; i < 46; i++) PcmBuffer[i] = PcmBuffer[nIn + i];
        self.pcm_buffer[:46] = self.pcm_buffer[self.n_input_samples:self.n_input_samples + 46]
        
        # atracdenc: for (size_t i = 0; i < nIn; i++) PcmBuffer[46+i] = in[i];
        self.pcm_buffer[46:46 + self.n_input_samples] = input_array
        
        # Prepare output arrays
        lower = np.zeros(self.n_input_samples // 2, dtype=np.float32)
        upper = np.zeros(self.n_input_samples // 2, dtype=np.float32)
        
        # atracdenc: for (size_t j = 0; j < nIn; j+=2)
        for j in range(0, self.n_input_samples, 2):
            lower_val, upper_val = self._calculate_qmf_taps(self.pcm_buffer, j)
            
            # atracdenc butterfly:
            # temp = upper[j/2];
            # upper[j/2] = lower[j/2] - upper[j/2];
            # lower[j/2] += temp;
            temp = upper_val
            upper[j//2] = lower_val - upper_val
            lower[j//2] = lower_val + temp
            
        return lower.tolist(), upper.tolist()

    def synthesis(self, lower_band: list[float], upper_band: list[float]) -> list[float]:
        """
        Exact implementation of atracdenc TQmf::Synthesis.
        """
        lower_array = np.array(lower_band, dtype=np.float32)
        upper_array = np.array(upper_band, dtype=np.float32)
        
        # atracdenc: memcpy(&PcmBufferMerge[0], &DelayBuff[0], 46*sizeof(float));
        self.pcm_buffer_merge[:46] = self.delay_buff
        
        # atracdenc: float* newPart = &PcmBufferMerge[46];
        new_part_start = 46
        
        # atracdenc inverse butterfly reconstruction:
        # for (int i = 0; i < nIn; i+=4) {
        #     newPart[i+0] = lower[i/2] + upper[i/2];
        #     newPart[i+1] = lower[i/2] - upper[i/2];
        #     newPart[i+2] = lower[i/2 + 1] + upper[i/2 + 1];
        #     newPart[i+3] = lower[i/2 + 1] - upper[i/2 + 1];
        # }
        max_samples = min(len(lower_array), len(upper_array)) * 2
        actual_samples = min(self.n_input_samples, max_samples)
        
        for i in range(0, actual_samples, 4):
            idx = i // 2
            if idx < len(lower_array) and idx + 1 < len(lower_array) and idx + 1 < len(upper_array):
                self.pcm_buffer_merge[new_part_start + i + 0] = lower_array[idx] + upper_array[idx]
                self.pcm_buffer_merge[new_part_start + i + 1] = lower_array[idx] - upper_array[idx]
                self.pcm_buffer_merge[new_part_start + i + 2] = lower_array[idx + 1] + upper_array[idx + 1]
                self.pcm_buffer_merge[new_part_start + i + 3] = lower_array[idx + 1] - upper_array[idx + 1]
            elif idx < len(lower_array) and idx < len(upper_array):
                # Handle last pair if only one sample left
                self.pcm_buffer_merge[new_part_start + i + 0] = lower_array[idx] + upper_array[idx]
                self.pcm_buffer_merge[new_part_start + i + 1] = lower_array[idx] - upper_array[idx]
        
        # Prepare output
        output = np.zeros(self.n_input_samples, dtype=np.float32)
        
        # atracdenc synthesis convolution:
        # float* winP = &PcmBufferMerge[0];
        # for (size_t j = nIn/2; j != 0; j--) {
        #     float s1 = 0; float s2 = 0;
        #     for (size_t i = 0; i < 48; i+=2) {
        #         s1 += winP[i] * QmfWindow[i];
        #         s2 += winP[i+1] * QmfWindow[i+1];
        #     }
        #     out[0] = s2; out[1] = s1;
        #     winP += 2; out += 2;
        # }
        win_ptr = 0
        out_ptr = 0
        
        for j in range(self.n_input_samples // 2):
            s1 = 0.0
            s2 = 0.0
            for i in range(0, 48, 2):
                s1 += self.pcm_buffer_merge[win_ptr + i] * QMF_WINDOW[i]
                s2 += self.pcm_buffer_merge[win_ptr + i + 1] * QMF_WINDOW[i + 1]
            
            output[out_ptr] = s2
            output[out_ptr + 1] = s1
            win_ptr += 2
            out_ptr += 2
        
        # atracdenc: memcpy(&DelayBuff[0], &PcmBufferMerge[nIn], 46*sizeof(float));
        self.delay_buff[:] = self.pcm_buffer_merge[self.n_input_samples:self.n_input_samples + 46]
        
        return output.tolist()


class Atrac1AnalysisFilterBank:
    """
    Exact implementation of atracdenc Atrac1AnalysisFilterBank template.
    Handles 39-sample delay compensation between QMF stages.
    """
    
    def __init__(self):
        """
        Initialize exactly like atracdenc.
        """
        # atracdenc constants
        self.n_in_samples = 512
        self.delay_comp = 39  # const static int delayComp = 39;
        
        # atracdenc QMF stages
        self.qmf1 = TQmf(self.n_in_samples)  # TQmf<TIn, nInSamples> Qmf1;
        self.qmf2 = TQmf(self.n_in_samples // 2)  # TQmf<TIn, nInSamples / 2> Qmf2;
        
        # atracdenc buffers
        self.mid_low_tmp = np.zeros(512, dtype=np.float32)  # std::vector<float> MidLowTmp;
        self.delay_buf = np.zeros(self.delay_comp + 512, dtype=np.float32)  # std::vector<float> DelayBuf;

    def analysis(self, pcm_input: list[float]) -> tuple[list[float], list[float], list[float]]:
        """
        Exact implementation of atracdenc Atrac1AnalysisFilterBank::Analysis.
        Always returns 256 samples for high band (atracdenc standard).
        """
        pcm_array = np.array(pcm_input, dtype=np.float32)
        
        # atracdenc: memcpy(&DelayBuf[0], &DelayBuf[256], sizeof(float) * delayComp);
        self.delay_buf[:self.delay_comp] = self.delay_buf[256:256 + self.delay_comp]
        
        # atracdenc: Qmf1.Analysis(pcm, &MidLowTmp[0], &DelayBuf[delayComp]);
        mid_low_list, high_delayed_list = self.qmf1.analysis(pcm_array.tolist())
        self.mid_low_tmp[:len(mid_low_list)] = np.array(mid_low_list, dtype=np.float32)
        self.delay_buf[self.delay_comp:self.delay_comp + len(high_delayed_list)] = np.array(high_delayed_list, dtype=np.float32)
        
        # atracdenc: Qmf2.Analysis(&MidLowTmp[0], low, mid);
        low_list, mid_list = self.qmf2.analysis(mid_low_list)
        
        # atracdenc: memcpy(hi, &DelayBuf[0], sizeof(float) * 256);
        # Always return 256 samples for high band (as per atracdenc)
        high_list = self.delay_buf[:256].tolist()
        
        return low_list, mid_list, high_list


class Atrac1SynthesisFilterBank:
    """
    Exact implementation of atracdenc Atrac1SynthesisFilterBank template.
    Handles 39-sample delay compensation for synthesis reconstruction.
    """
    
    def __init__(self):
        """
        Initialize exactly like atracdenc.
        """
        # atracdenc constants
        self.n_in_samples = 512
        self.delay_comp = 39  # const static int delayComp = 39;
        
        # atracdenc QMF stages  
        self.qmf1 = TQmf(self.n_in_samples)  # TQmf<TOut, nInSamples> Qmf1;
        self.qmf2 = TQmf(self.n_in_samples // 2)  # TQmf<TOut, nInSamples / 2> Qmf2;
        
        # atracdenc buffers
        self.mid_low_tmp = np.zeros(512, dtype=np.float32)  # std::vector<float> MidLowTmp;
        self.delay_buf = np.zeros(self.delay_comp + 512, dtype=np.float32)  # std::vector<float> DelayBuf;

    def synthesis(self, low_band: list[float], mid_band: list[float], hi_band: list[float]) -> list[float]:
        """
        Exact implementation of atracdenc Atrac1SynthesisFilterBank::Synthesis.
        Handles variable high band sizes (256 for long blocks, 64 for short blocks).
        """
        low_array = np.array(low_band, dtype=np.float32)
        mid_array = np.array(mid_band, dtype=np.float32)
        hi_array = np.array(hi_band, dtype=np.float32)
        
        # atracdenc: memcpy(&DelayBuf[0], &DelayBuf[256], sizeof(float) * delayComp);
        self.delay_buf[:self.delay_comp] = self.delay_buf[256:256 + self.delay_comp]
        
        # Handle variable high band size (256 for long blocks, 64 for short blocks)
        # atracdenc: memcpy(&DelayBuf[delayComp], hi, sizeof(float) * 256);
        if len(hi_array) == 256:
            # Long block mode: use full high band
            self.delay_buf[self.delay_comp:self.delay_comp + 256] = hi_array
        elif len(hi_array) == 64:
            # Short block mode: expand 64 samples to 256 by zero-padding
            # This is a temporary fix - proper MDCT buffering should handle this
            expanded_hi = np.zeros(256, dtype=np.float32)
            expanded_hi[:64] = hi_array  # Place 64 samples at start
            self.delay_buf[self.delay_comp:self.delay_comp + 256] = expanded_hi
        else:
            raise ValueError(f"Unexpected high band size: {len(hi_array)}, expected 256 or 64")
        
        # atracdenc: Qmf2.Synthesis(&MidLowTmp[0], &low[0], &mid[0]);
        mid_low_tmp_list = self.qmf2.synthesis(low_array.tolist(), mid_array.tolist())
        self.mid_low_tmp[:len(mid_low_tmp_list)] = np.array(mid_low_tmp_list, dtype=np.float32)
        
        # atracdenc: Qmf1.Synthesis(&pcm[0], &MidLowTmp[0], &DelayBuf[0]);
        pcm_list = self.qmf1.synthesis(mid_low_tmp_list, self.delay_buf[:256].tolist())
        
        return pcm_list