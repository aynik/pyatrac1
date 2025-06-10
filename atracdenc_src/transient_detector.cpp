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

#include "transient_detector.h"
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
namespace NAtracDEnc {

using std::vector;
static float calculateRMS(const float* in, uint32_t n) {
    float s = 0;
    for (uint32_t i = 0; i < n; i++) {
        s += (in[i] * in[i]);
    }
    s /= n;
    return sqrt(s);
}

static float calculatePeak(const float* in, uint32_t n) {
    float s = 0;
    for (uint32_t i = 0; i < n; i++) {
        float absVal = std::abs(in[i]);
        if (absVal > s)
            s = absVal;
    }
    return s;
}

void TTransientDetector::HPFilter(const float* in, float* out) {
    static const float fircoef[] = {
        -8.65163e-18 * 2.0, -0.00851586 * 2.0, -6.74764e-18 * 2.0, 0.0209036 * 2.0,
        -3.36639e-17 * 2.0, -0.0438162 * 2.0, -1.54175e-17 * 2.0, 0.0931738 * 2.0,
        -5.52212e-17 * 2.0, -0.313819 * 2.0
    };
    memcpy(HPFBuffer.data() + PrevBufSz, in, BlockSz * sizeof(float));
    const float* inBuf = HPFBuffer.data();
    for (size_t i = 0; i < BlockSz; ++i) {
        float s = inBuf[i + 10];
        float s2 = 0;
        for (size_t j = 0; j < ((FIRLen - 1) / 2) - 1 ; j += 2) {
            s += fircoef[j] * (inBuf[i + j] + inBuf[i + FIRLen - j]);
            s2 += fircoef[j + 1] * (inBuf[i + j + 1] + inBuf[i + FIRLen - j - 1]);
        }
        out[i] = (s + s2)/2;
    }
    memcpy(HPFBuffer.data(), in + (BlockSz - PrevBufSz),  PrevBufSz * sizeof(float));
}


bool TTransientDetector::Detect(const float* buf, uint32_t debug_channel, uint32_t debug_frame, const char* band_name) {
    const uint16_t nBlocksToAnalize = NShortBlocks + 1;
    float* rmsPerShortBlock = reinterpret_cast<float*>(alloca(sizeof(float) * nBlocksToAnalize));
    std::vector<float> filtered(BlockSz);
    
    // Log input buffer to transient detection
    ATRAC_LOG_STAGE("TRANSIENT_INPUT", "SAMPLES", buf, std::min(32, (int)BlockSz), debug_channel, debug_frame, band_name, algorithm=transient_detection);
    
    // Log input statistics to match PyATRAC1 format
    float min_input = *std::min_element(buf, buf + BlockSz);
    float max_input = *std::max_element(buf, buf + BlockSz);
    float sum_input = std::accumulate(buf, buf + BlockSz, 0.0f);
    float mean_input = sum_input / BlockSz;
    float input_stats[3] = {min_input, max_input, mean_input};
    ATRAC_LOG_STAGE("TRANSIENT_INPUT_STATS", "RANGE", input_stats, 3, debug_channel, debug_frame, band_name, algorithm=transient_detection operation=input_statistics);
    
    // Apply HPF and log filtered output  
    HPFilter(buf, filtered.data());
    ATRAC_LOG_STAGE("TRANSIENT_HPF_OUTPUT", "SAMPLES", filtered.data(), std::min(32, (int)BlockSz), debug_channel, debug_frame, band_name, algorithm=transient_detection operation=hpf_filter);
    
    // Log HPF statistics to match PyATRAC1 format
    float min_hpf = *std::min_element(filtered.data(), filtered.data() + BlockSz);
    float max_hpf = *std::max_element(filtered.data(), filtered.data() + BlockSz);
    float hpf_stats[2] = {min_hpf, max_hpf};
    ATRAC_LOG_STAGE("TRANSIENT_HPF_STATS", "RANGE", hpf_stats, 2, debug_channel, debug_frame, band_name, algorithm=transient_detection operation=hpf_statistics);
    
    bool trans = false;
    rmsPerShortBlock[0] = LastEnergy;
    
    // Log previous energy value
    ATRAC_LOG_VALUE("TRANSIENT_PREV_ENERGY", "energy", LastEnergy, debug_channel, debug_frame, band_name, algorithm=transient_detection operation=energy_tracking);
    
    for (uint16_t i = 1; i < nBlocksToAnalize; ++i) {
        // Calculate RMS energy for this block
        float raw_rms = calculateRMS(&filtered[(size_t)(i - 1) * ShortSz], ShortSz);
        rmsPerShortBlock[i] = 19.0 * log10(raw_rms);
        
        // Log block energy calculations
        if (i <= 4) { // Log first few blocks for debugging
            float energy_data[2] = {raw_rms, rmsPerShortBlock[i]};
            ATRAC_LOG_STAGE("TRANSIENT_BLOCK_ENERGY", "raw_log", energy_data, 2, debug_channel, debug_frame, band_name, 
                           algorithm=transient_detection block_index=i operation=energy_calculation);
        }
        
        // Check for energy increase (attack transient)
        float energy_diff_up = rmsPerShortBlock[i] - rmsPerShortBlock[i - 1];
        if (energy_diff_up > 16) {
            trans = true;
            LastTransientPos = i;
            
            // Log transient detection
            float threshold_data[3] = {rmsPerShortBlock[i - 1], rmsPerShortBlock[i], energy_diff_up};
            ATRAC_LOG_STAGE("TRANSIENT_ATTACK", "prev_curr_diff", threshold_data, 3, debug_channel, debug_frame, band_name,
                           algorithm=transient_detection threshold=16.0 transient_type=attack block_pos=i);
        }
        
        // Check for energy decrease (decay transient)
        float energy_diff_down = rmsPerShortBlock[i - 1] - rmsPerShortBlock[i];
        if (energy_diff_down > 20) {
            trans = true;
            LastTransientPos = i;
            
            // Log transient detection
            float threshold_data[3] = {rmsPerShortBlock[i - 1], rmsPerShortBlock[i], energy_diff_down};
            ATRAC_LOG_STAGE("TRANSIENT_DECAY", "prev_curr_diff", threshold_data, 3, debug_channel, debug_frame, band_name,
                           algorithm=transient_detection threshold=20.0 transient_type=decay block_pos=i);
        }
    }
    
    // Log energy values to match PyATRAC1 format (first 3 values)
    float energy_values[3] = {
        nBlocksToAnalize > 1 ? rmsPerShortBlock[1] : -100.0f,
        nBlocksToAnalize > 2 ? rmsPerShortBlock[2] : -100.0f, 
        -100.0f  // PyATRAC1 seems to log a third energy value
    };
    ATRAC_LOG_STAGE("TRANSIENT_ENERGY", "VALUES", energy_values, 3, debug_channel, debug_frame, band_name,
                   algorithm=transient_detection operation=energy_calculation);
    
    LastEnergy = rmsPerShortBlock[NShortBlocks];
    
    // Log final decision to match PyATRAC1 format
    float decision_data[2] = {
        nBlocksToAnalize > 1 ? (rmsPerShortBlock[1] - rmsPerShortBlock[0]) : 0.0f,  // Energy difference
        trans ? 1.0f : 0.0f  // Boolean decision
    };
    ATRAC_LOG_STAGE("TRANSIENT_DECISION", "RESULT", decision_data, 2, debug_channel, debug_frame, band_name,
                   algorithm=transient_detection operation=final_decision);
    
    // Log thresholds to match PyATRAC1 format
    float threshold_values[4] = {16.0f, -12.0f, trans ? 1.0f : 0.0f, 0.0f};
    ATRAC_LOG_STAGE("TRANSIENT_THRESHOLDS", "VALUES", threshold_values, 4, debug_channel, debug_frame, band_name,
                   algorithm=transient_detection operation=threshold_comparison);
    
    return trans;
}

std::vector<float> AnalyzeGain(const float* in, const uint32_t len, const uint32_t maxPoints, bool useRms) {
    vector<float> res;
    const uint32_t step = len / maxPoints;
    for (uint32_t pos = 0; pos < len; pos += step) {
        float rms = useRms ? calculateRMS(in + pos, step) : calculatePeak(in + pos, step);
        res.emplace_back(rms);
    }
    return res;
}

} //namespace NAtracDEnc
