/*
 * This file is part of AtracDEnc.
 *
 * AtracDEnc is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * AtracDEnc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with AtracDEnc; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#pragma once
#include <string.h>
#include <algorithm>
#include <cstdio>

#include "../config.h"
#include "../debug_logger.h"

template<class TPCM, int nIn>
class TQmf {
    static const float TapHalf[24];
    float QmfWindow[48];
    TPCM PcmBuffer[nIn + 46];
    float PcmBufferMerge[nIn + 46];
    float DelayBuff[46];
public:
    TQmf() {
        const int sz = sizeof(QmfWindow)/sizeof(QmfWindow[0]);

        for (size_t i = 0 ; i < sz/2; i++) {
            QmfWindow[i] = QmfWindow[ sz - 1 - i] = TapHalf[i] * 2.0;
        }
        for (size_t i = 0; i < sizeof(PcmBuffer)/sizeof(PcmBuffer[0]); i++) {
            PcmBuffer[i] = 0;
            PcmBufferMerge[i] = 0;
        }
    }

    void Analysis(TPCM* in, float* lower, float* upper, uint32_t debug_channel = 0, uint32_t debug_frame = 0, const char* stage = "") {
        // Log QMF input
        ATRAC_LOG_STAGE("QMF_INPUT", "SAMPLES", in, std::min(16, nIn), debug_channel, debug_frame, stage, stage=qmf_analysis);
        
        float temp;
        
        // Log buffer shift operation
        ATRAC_LOG_STAGE("QMF_BUFFER_SHIFT", "OLD_TO_NEW", PcmBuffer, std::min(8, 46), debug_channel, debug_frame, stage, operation=buffer_shift);
        
        for (size_t i = 0; i < 46; i++)
            PcmBuffer[i] = PcmBuffer[nIn + i];

        for (size_t i = 0; i < nIn; i++)
            PcmBuffer[46+i] = in[i];
            
        // Log filled buffer
        ATRAC_LOG_STAGE("QMF_BUFFER_FILLED", "TOTAL", PcmBuffer, std::min(64, nIn + 46), debug_channel, debug_frame, stage, operation=buffer_fill);

        for (size_t j = 0; j < nIn; j+=2) {
            lower[j/2] = upper[j/2] = 0.0;
            for (size_t i = 0; i < 24; i++)  {
                lower[j/2] += QmfWindow[2*i] * PcmBuffer[48-1+j-(2*i)];
                upper[j/2] += QmfWindow[(2*i)+1] * PcmBuffer[48-1+j-(2*i)-1];
            }
            
            // Log raw tap calculations for first few samples
            if (j < 8) {
                float tap_values[2] = {lower[j/2], upper[j/2]};
                char tap_label[32];
                snprintf(tap_label, sizeof(tap_label), "J%d", (int)j);
                ATRAC_LOG_STAGE("QMF_RAW_TAPS", tap_label, tap_values, 2, debug_channel, debug_frame, stage, tap_index=j);
            }
            
            temp = upper[j/2];
            upper[j/2] = lower[j/2] - upper[j/2];
            lower[j/2] += temp;
            
            // Log butterfly output for first few samples  
            if (j < 8) {
                float butterfly_values[2] = {lower[j/2], upper[j/2]};
                char butterfly_label[32];
                snprintf(butterfly_label, sizeof(butterfly_label), "J%d", (int)j);
                ATRAC_LOG_STAGE("QMF_BUTTERFLY", butterfly_label, butterfly_values, 2, debug_channel, debug_frame, stage, butterfly_index=j);
            }
        }
        
        // Log final QMF outputs
        ATRAC_LOG_STAGE("QMF_FINAL_LOWER", "SAMPLES", lower, std::min(16, nIn/2), debug_channel, debug_frame, stage, output=lower_band);
        ATRAC_LOG_STAGE("QMF_FINAL_UPPER", "SAMPLES", upper, std::min(16, nIn/2), debug_channel, debug_frame, stage, output=upper_band);
        
        // Calculate and log statistics
        float stats[4];
        float min_lower = lower[0], max_lower = lower[0];
        float min_upper = upper[0], max_upper = upper[0];
        for (int i = 0; i < nIn/2; i++) {
            if (lower[i] < min_lower) min_lower = lower[i];
            if (lower[i] > max_lower) max_lower = lower[i];
            if (upper[i] < min_upper) min_upper = upper[i];
            if (upper[i] > max_upper) max_upper = upper[i];
        }
        stats[0] = min_lower; stats[1] = max_lower; stats[2] = min_upper; stats[3] = max_upper;
        ATRAC_LOG_STAGE("QMF_STATS", "RANGE", stats, 4, debug_channel, debug_frame, stage, statistic=min_max_range);
    }

    void Synthesis(TPCM* out, float* lower, float* upper) {
        memcpy(&PcmBufferMerge[0], &DelayBuff[0], 46*sizeof(float));
        float* newPart = &PcmBufferMerge[46];
        for (int i = 0; i < nIn; i+=4) {
            newPart[i+0] = lower[i/2] + upper[i/2];
            newPart[i+1] = lower[i/2] - upper[i/2];
            newPart[i+2] = lower[i/2 + 1] + upper[i/2 + 1];
            newPart[i+3] = lower[i/2 + 1] - upper[i/2 + 1];
        }

        float* winP = &PcmBufferMerge[0];
        for (size_t j = nIn/2; j != 0; j--) {
            float s1 = 0;
            float s2 = 0;
            for (size_t i = 0; i < 48; i+=2) {
                s1 += winP[i] * QmfWindow[i];
                s2 += winP[i+1] * QmfWindow[i+1];
            }
            out[0] = s2;
            out[1] = s1;
            winP += 2;
            out += 2;
        }
        memcpy(&DelayBuff[0], &PcmBufferMerge[nIn], 46*sizeof(float));
    }
};

template<class TPCM, int nIn>
const float TQmf<TPCM, nIn>::TapHalf[24] = {
    -0.00001461907,  -0.00009205479, -0.000056157569,  0.00030117269,
    0.0002422519,    -0.00085293897, -0.0005205574,    0.0020340169,
    0.00078333891,   -0.0042153862,  -0.00075614988,   0.0078402944,
    -0.000061169922, -0.01344162,    0.0024626821,     0.021736089,
    -0.007801671,    -0.034090221,   0.01880949,       0.054326009,
    -0.043596379,    -0.099384367,   0.13207909,       0.46424159
};

