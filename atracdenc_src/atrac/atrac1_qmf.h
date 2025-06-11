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

#include <vector> // Added to provide std::vector
#include "../qmf/qmf.h"
#include "../debug_logger.h"

namespace NAtracDEnc {

template<class TIn>
class Atrac1AnalysisFilterBank {
    const static int nInSamples = 512;
    const static int delayComp = 39;
    TQmf<TIn, nInSamples> Qmf1;
    TQmf<TIn, nInSamples / 2> Qmf2;
    std::vector<float> MidLowTmp;
    std::vector<float> DelayBuf;
public:
    Atrac1AnalysisFilterBank() {
        MidLowTmp.resize(512);
        DelayBuf.resize(delayComp + 512);
    }
    void Analysis(TIn* pcm, float* low, float* mid, float* hi, uint32_t debug_channel = 0, uint32_t debug_frame = 0) {
        // Log PCM input to analysis filter bank  
        ATRAC_LOG_STAGE("QMF_ANALYSIS_INPUT", "PCM", pcm, nInSamples, debug_channel, debug_frame, "FULL", algorithm=qmf_analysis);
        
        // Log delay buffer shift
        ATRAC_LOG_STAGE("QMF_DELAY_SHIFT", "OLD_NEW", &DelayBuf[0], delayComp, debug_channel, debug_frame, "FULL", operation=delay_shift);
        
        memcpy(&DelayBuf[0], &DelayBuf[256], sizeof(float) *  delayComp);
        
        // Call QMF stages with debug context
        Qmf1.Analysis(pcm, &MidLowTmp[0], &DelayBuf[delayComp], debug_channel, debug_frame, "QMF1");
        
        // Log intermediate outputs from QMF1
        ATRAC_LOG_STAGE("QMF1_MIDLOW_OUT", "SAMPLES", &MidLowTmp[0], nInSamples/2, debug_channel, debug_frame, "FULL", qmf_operation=qmf1_output);
        ATRAC_LOG_STAGE("QMF1_HIGH_DELAYED", "SAMPLES", &DelayBuf[delayComp], nInSamples/2, debug_channel, debug_frame, "FULL", qmf_operation=qmf1_delayed);
        
        Qmf2.Analysis(&MidLowTmp[0], low, mid, debug_channel, debug_frame, "QMF2");
        memcpy(hi, &DelayBuf[0], sizeof(float) * 256);

        // Log final outputs for all bands  
        ATRAC_LOG_STAGE("QMF_FINAL_LOW", "BAND", low, nInSamples/4, debug_channel, debug_frame, "FULL", qmf_operation=final_output);
        ATRAC_LOG_STAGE("QMF_FINAL_MID", "BAND", mid, nInSamples/4, debug_channel, debug_frame, "FULL", qmf_operation=final_output);
        ATRAC_LOG_STAGE("QMF_FINAL_HIGH", "BAND", hi, 256, debug_channel, debug_frame, "FULL", qmf_operation=final_output);
        
        // Log band energy statistics
        float energies[3];
        energies[0] = 0; for(int i = 0; i < nInSamples/4; i++) energies[0] += low[i] * low[i];
        energies[1] = 0; for(int i = 0; i < nInSamples/4; i++) energies[1] += mid[i] * mid[i];
        energies[2] = 0; for(int i = 0; i < 256; i++) energies[2] += hi[i] * hi[i];
        ATRAC_LOG_STAGE("QMF_BAND_STATS", "ENERGY", energies, 3, debug_channel, debug_frame, "FULL", statistic=band_energy);
    }
};
template<class TOut>
class Atrac1SynthesisFilterBank {
    const static int nInSamples = 512;
    const static int delayComp = 39;
    TQmf<TOut, nInSamples> Qmf1;
    TQmf<TOut, nInSamples / 2> Qmf2;
    std::vector<float> MidLowTmp;
    std::vector<float> DelayBuf;
public:
    Atrac1SynthesisFilterBank() {
        MidLowTmp.resize(512);
        DelayBuf.resize(delayComp + 512);
    }
    void Synthesis(TOut* pcm, float* low, float* mid, float* hi) {
        memcpy(&DelayBuf[0], &DelayBuf[256], sizeof(float) *  delayComp);
        memcpy(&DelayBuf[delayComp], hi, sizeof(float) * 256);
        Qmf2.Synthesis(&MidLowTmp[0], &low[0], &mid[0]);
        Qmf1.Synthesis(&pcm[0], &MidLowTmp[0], &DelayBuf[0]);
    }
};

} //namespace NAtracDEnc
