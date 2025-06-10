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

#include <vector>
#include <cstring>

#include "debug_logger.h"
#include "atrac1denc.h"
#include "bitstream/bitstream.h"
#include "atrac/atrac1.h"
#include "atrac/atrac1_dequantiser.h"
#include "atrac/atrac1_qmf.h"
#include "atrac/atrac1_bitalloc.h"
#include "atrac/atrac_psy_common.h"
#include "util.h"

namespace NAtracDEnc {
using namespace NBitStream;
using namespace NAtrac1;
using namespace NMDCT;
using std::vector;

TAtrac1Encoder::TAtrac1Encoder(TCompressedOutputPtr&& aea, TAtrac1EncodeSettings&& settings)
    : Aea(std::move(aea))
    , Settings(std::move(settings))
    , LoudnessCurve(CreateLoudnessCurve(TAtrac1Data::NumSamples))
{
}

TAtrac1Decoder::TAtrac1Decoder(TCompressedInputPtr&& aea)
    : Aea(std::move(aea))
{
}

static void vector_fmul_window(float *dst, const float *src0,
                                const float *src1, const float *win, int len)
{
    int i, j;

    dst  += len;
    win  += len;
    src0 += len;

    for (i = -len, j = len - 1; i < 0; i++, j--) {
        float s0 = src0[i];
        float s1 = src1[j];
        float wi = win[i];
        float wj = win[j];
        dst[i] = s0 * wj - s1 * wi;
        dst[j] = s0 * wi + s1 * wj;
    }
}

void TAtrac1MDCT::Mdct(float Specs[512], float* low, float* mid, float* hi, const TAtrac1Data::TBlockSizeMod& blockSize) {
    uint32_t pos = 0;
    
    // Log MDCT inputs for each band
    ATRAC_LOG_MDCT_INPUT(low, 128, debug_channel, debug_frame, "LOW", 
                         (blockSize.LogCount[0] == 0) ? 128 : 64, 
                         (blockSize.LogCount[0] == 0) ? "long" : "short");
    ATRAC_LOG_MDCT_INPUT(mid, 128, debug_channel, debug_frame, "MID", 
                         (blockSize.LogCount[1] == 0) ? 128 : 64,
                         (blockSize.LogCount[1] == 0) ? "long" : "short");
    ATRAC_LOG_MDCT_INPUT(hi, 256, debug_channel, debug_frame, "HIGH", 
                         (blockSize.LogCount[2] == 0) ? 256 : 64,
                         (blockSize.LogCount[2] == 0) ? "long" : "short");
    
    // Log individual band inputs to match PyATRAC1
    ATRAC_LOG_STAGE("MDCT_INPUT_LOW", "samples", low, 128, debug_channel, debug_frame, "LOW", algorithm="mdct");
    ATRAC_LOG_STAGE("MDCT_INPUT_MID", "samples", mid, 128, debug_channel, debug_frame, "MID", algorithm="mdct");
    ATRAC_LOG_STAGE("MDCT_INPUT_HIGH", "samples", hi, 256, debug_channel, debug_frame, "HIGH", algorithm="mdct");
    
    for (uint32_t band = 0; band < TAtrac1Data::NumQMF; band++) {
        const uint32_t numMdctBlocks = 1 << blockSize.LogCount[band];
        float* srcBuf = (band == 0) ? low : (band == 1) ? mid : hi;
        
        uint32_t bufSz = (band == 2) ? 256 : 128;
        const uint32_t blockSz = (numMdctBlocks == 1) ? bufSz : 32;
        uint32_t winStart = (numMdctBlocks == 1) ? ((band == 2) ? 112 : 48) : 0;
        //compensate level for 3rd band in case of short window
        const float multiple = (numMdctBlocks != 1 && band == 2) ? 2.0 : 1.0;
        vector<float> tmp(512);
        uint32_t blockPos = 0;
        
        // Log MDCT band parameters to match PyATRAC1
        const char* band_name = (band == 0) ? "LOW" : (band == 1) ? "MID" : "HIGH";
        
        // Log srcBuf assignment for buffer tracking
        ATRAC_LOG_STAGE("MDCT_SRCBUF_ASSIGNED", "samples", srcBuf, 16, debug_channel, debug_frame, band_name, 
                       algorithm="buffer_tracking", operation="srcbuf_assignment");
        float band_params[] = {(float)numMdctBlocks, (float)bufSz, (float)blockSz, (float)winStart, multiple};
        ATRAC_LOG_STAGE("MDCT_BAND_PARAMS", "params", band_params, 5, debug_channel, debug_frame, band_name, 
                       algorithm="mdct", num_mdct_blocks=numMdctBlocks, buf_sz=bufSz, block_sz=blockSz);

        for (size_t k = 0; k < numMdctBlocks; ++k) {
            // Log pre-windowing state to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_PRE_WINDOW", "samples", &srcBuf[blockPos], std::min(16, (int)blockSz), 
                           debug_channel, debug_frame, band_name, block=k);
            
            // Log window function to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_WINDOW_FUNC", "coeffs", TAtrac1Data::SineWindow, 16, 
                           debug_channel, debug_frame, band_name, block=k);
            
            // Store original values for difference calculation
            vector<float> orig_values(srcBuf + blockPos, srcBuf + blockPos + blockSz);
            
            // Log buffer state before reading overlap data
            ATRAC_LOG_STAGE("BUFFER_STATE_READ", "overlap", &srcBuf[bufSz], 32, 
                           debug_channel, debug_frame, band_name, 
                           algorithm="buffer_tracking", buffer_offset=bufSz, operation="read_overlap");
            
            memcpy(&tmp[winStart], &srcBuf[bufSz], 32 * sizeof(float));
            
            // Log what was copied to tmp buffer  
            ATRAC_LOG_STAGE("BUFFER_OVERLAP_COPIED", "data", &tmp[winStart], 32,
                           debug_channel, debug_frame, band_name,
                           algorithm="buffer_tracking", target_offset=winStart, operation="copy_overlap");
            for (size_t i = 0; i < 32; i++) {
                srcBuf[bufSz + i] = TAtrac1Data::SineWindow[i] * srcBuf[blockPos + blockSz - 32 + i];
                srcBuf[blockPos + blockSz - 32 + i] = TAtrac1Data::SineWindow[31 - i] * srcBuf[blockPos + blockSz - 32 + i];
            }
            
            // Log windowing operation details to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_WINDOW_APPLIED", "samples", &srcBuf[blockPos], std::min(16, (int)blockSz), 
                           debug_channel, debug_frame, band_name, block=k);
            
            // Log windowing difference to match PyATRAC1
            vector<float> window_diff(blockSz);
            for (size_t i = 0; i < blockSz && i < orig_values.size(); i++) {
                window_diff[i] = srcBuf[blockPos + i] - orig_values[i];
            }
            ATRAC_LOG_STAGE("MDCT_WINDOW_DIFF", "samples", window_diff.data(), std::min(16, (int)blockSz), 
                           debug_channel, debug_frame, band_name, block=k);
            
            memcpy(&tmp[winStart+32], &srcBuf[blockPos], blockSz * sizeof(float));
            
            // Log windowed input before MDCT
            int mdct_size = (numMdctBlocks == 1) ? ((band == 2) ? 256 : 128) : 64;
            int window_size = (numMdctBlocks == 1) ? ((band == 2) ? 512 : 256) : 64;
            
            // Ensure we don't read beyond tmp buffer bounds (size 512)
            int safe_window_size = std::min(window_size, (int)(tmp.size() - winStart));
            
            // Log final windowed state to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_WINDOWED_FINAL", "samples", &tmp[winStart], std::min(64, safe_window_size), 
                           debug_channel, debug_frame, band_name, block=k);
            
            ATRAC_LOG_STAGE("MDCT_WINDOWED", "samples", &tmp[winStart], safe_window_size, 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", mdct_size=mdct_size, window_type="sine", block=k);
            
            // Log MDCT input before transform to compare with PyATRAC1
            ATRAC_LOG_STAGE("MDCT_TRANSFORM_INPUT", "samples", &tmp[0], 64, 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", operation="pre_transform", block=k);
            
            const vector<float>&  sp = (numMdctBlocks == 1) ? ((band == 2) ? Mdct512(&tmp[0]) : Mdct256(&tmp[0])) : Mdct64(&tmp[0]);
            
            // Log transform info to match PyATRAC1
            float transform_info[] = {(float)tmp.size(), (float)sp.size()};
            ATRAC_LOG_STAGE("MDCT_TRANSFORM_INFO", "info", transform_info, 2, 
                           debug_channel, debug_frame, band_name, block=k, mdct_engine="mdct");
            
            // Log raw DCT coefficients
            ATRAC_LOG_STAGE("MDCT_RAW_DCT", "coeffs", sp.data(), sp.size(), 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", mdct_size=mdct_size, dct_type="IV", block=k);
            
            // Log raw output to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_RAW_OUTPUT", "coeffs", sp.data(), sp.size(), 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", block=k, mdct_engine="mdct");
            
            // Log scaling details before assignment to match PyATRAC1
            float assignment_details[] = {(float)pos, (float)blockPos, (float)sp.size(), multiple};
            ATRAC_LOG_STAGE("MDCT_PRE_ASSIGNMENT", "details", assignment_details, 4,
                           debug_channel, debug_frame, band_name, algorithm="mdct", 
                           block=k, operation="pre_assignment");
            
            // Log buffer slice before assignment
            ATRAC_LOG_STAGE("MDCT_BUFFER_BEFORE", "coeffs", &Specs[blockPos + pos], std::min(16, (int)sp.size()),
                           debug_channel, debug_frame, band_name, algorithm="mdct", 
                           block=k, operation="buffer_before_assignment");
            
            // Apply level compensation and store
            for (size_t i = 0; i < sp.size(); i++) {
                Specs[blockPos + pos + i] = sp[i] * multiple;
            }
            
            // Log buffer slice after assignment
            ATRAC_LOG_STAGE("MDCT_BUFFER_AFTER", "coeffs", &Specs[blockPos + pos], std::min(16, (int)sp.size()),
                           debug_channel, debug_frame, band_name, algorithm="mdct", 
                           block=k, operation="buffer_after_assignment");
            
            // Log after scaling to match PyATRAC1
            ATRAC_LOG_STAGE("MDCT_AFTER_SCALE", "coeffs", &Specs[blockPos + pos], sp.size(), 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", block=k, scale_factor=multiple);
            
            // Log after level compensation (if applied)
            if (multiple != 1.0) {
                ATRAC_LOG_STAGE("MDCT_AFTER_LEVEL", "coeffs", &Specs[blockPos + pos], sp.size(), 
                               debug_channel, debug_frame, band_name, 
                               algorithm="mdct", operation="level_compensation", multiplier=multiple, block=k);
            }
            
            // Apply SwapArray for mid and high bands
            if (band) {
                SwapArray(&Specs[blockPos + pos], sp.size());
                
                // Log after swap
                ATRAC_LOG_STAGE("MDCT_AFTER_SWAP", "coeffs", &Specs[blockPos + pos], sp.size(), 
                               debug_channel, debug_frame, band_name, 
                               algorithm="mdct", operation="swap_array", block=k);
            }
            
            // Log final output for this block
            ATRAC_LOG_STAGE("MDCT_BLOCK_OUTPUT", "coeffs", &Specs[blockPos + pos], sp.size(), 
                           debug_channel, debug_frame, band_name, 
                           algorithm="mdct", stage="block_final", block=k);

            blockPos += 32;
        }
        
        // Log MDCT_OUTPUT for entire band to match PyATRAC1
        ATRAC_LOG_STAGE("MDCT_OUTPUT", "coeffs", &Specs[pos], std::min(16, (int)bufSz), 
                       debug_channel, debug_frame, band_name, 
                       algorithm="mdct", band_complete=true, total_coeffs=bufSz);
        
        pos += bufSz;
    }
    
    // Log combined MDCT outputs
    ATRAC_LOG_MDCT_OUTPUT(Specs, 512, debug_channel, debug_frame, "COMBINED", 
                          512, "combined");
}
void TAtrac1MDCT::IMdct(float Specs[512], const TAtrac1Data::TBlockSizeMod& mode, float* low, float* mid, float* hi) {
    uint32_t pos = 0;
    for (size_t band = 0; band < TAtrac1Data::NumQMF; band++) {
        const uint32_t numMdctBlocks = 1 << mode.LogCount[band];
        const uint32_t bufSz = (band == 2) ? 256 : 128;
        const uint32_t blockSz = (numMdctBlocks == 1) ? bufSz : 32;
        uint32_t start = 0;

        float* dstBuf = (band == 0) ? low : (band == 1) ? mid : hi;

        vector<float> invBuf(512);
        float* prevBuf = &dstBuf[bufSz * 2  - 16];
        for (uint32_t block = 0; block < numMdctBlocks; block++) {
            if (band) {
                SwapArray(&Specs[pos], blockSz);
            }
            vector<float> inv = (numMdctBlocks != 1) ? Midct64(&Specs[pos]) : (bufSz == 128) ? Midct256(&Specs[pos]) : Midct512(&Specs[pos]);
            for (size_t i = 0; i < (inv.size()/2); i++) {
                invBuf[start+i] = inv[i + inv.size()/4];
            }

            vector_fmul_window(dstBuf + start, prevBuf, &invBuf[start], &TAtrac1Data::SineWindow[0], 16);

            prevBuf = &invBuf[start+16];
            start += blockSz;
            pos += blockSz;
        }
        if (numMdctBlocks == 1)
            memcpy(dstBuf + 32, &invBuf[16], ((band == 2) ? 240 : 112) * sizeof(float));

        // Log buffer state before writing overlap data
        const char* band_name = (band == 0) ? "LOW" : (band == 1) ? "MID" : "HIGH";
        ATRAC_LOG_STAGE("BUFFER_STATE_BEFORE_WRITE", "overlap", &dstBuf[bufSz*2 - 16], 16, 
                       0, 0, band_name,  // TODO: Need access to debug context in IMDCT
                       algorithm="buffer_tracking", buffer_offset=bufSz*2-16, operation="before_write_overlap");
        
        for (size_t j = 0; j < 16; j++) {
            dstBuf[bufSz*2 - 16  + j] = invBuf[bufSz - 16 + j];
        }
        
        // Log buffer state after writing overlap data  
        ATRAC_LOG_STAGE("BUFFER_STATE_AFTER_WRITE", "overlap", &dstBuf[bufSz*2 - 16], 16,
                       0, 0, band_name,
                       algorithm="buffer_tracking", buffer_offset=bufSz*2-16, operation="after_write_overlap");
        
        // Log what was written from invBuf
        ATRAC_LOG_STAGE("BUFFER_OVERLAP_SOURCE", "data", &invBuf[bufSz - 16], 16,
                       0, 0, band_name,
                       algorithm="buffer_tracking", source_offset=bufSz-16, operation="overlap_source_data");
    }
}

TPCMEngine::TProcessLambda TAtrac1Decoder::GetLambda() {
    return [this](float* data, const TPCMEngine::ProcessMeta& /*meta*/) {
        float sum[512];
        const uint32_t srcChannels = Aea->GetChannelNum();
        for (uint32_t channel = 0; channel < srcChannels; channel++) {
            std::unique_ptr<ICompressedIO::TFrame> frame(Aea->ReadFrame());

            TBitStream bitstream(frame->Get(), frame->Size());

            TAtrac1Data::TBlockSizeMod mode(&bitstream);
            TAtrac1Dequantiser dequantiser;
            vector<float> specs;
            specs.resize(512);;
            dequantiser.Dequant(&bitstream, mode, &specs[0]);

            IMdct(&specs[0], mode, &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
            SynthesisFilterBank[channel].Synthesis(&sum[0], &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0]);
            for (size_t i = 0; i < TAtrac1Data::NumSamples; ++i) {
                if (sum[i] > PcmValueMax)
                    sum[i] = PcmValueMax;
                if (sum[i] < PcmValueMin)
                    sum[i] = PcmValueMin;

                data[i * srcChannels + channel] = sum[i];
            }
        }
        return TPCMEngine::EProcessResult::PROCESSED;
    };
}


TPCMEngine::TProcessLambda TAtrac1Encoder::GetLambda() {
    const uint32_t srcChannels = Aea->GetChannelNum();

    BitAllocs.reserve(srcChannels);
    for (uint32_t ch = 0; ch < srcChannels; ch++) {
        BitAllocs.emplace_back(new TAtrac1SimpleBitAlloc(Aea.get(), Settings.GetBfuIdxConst(), Settings.GetFastBfuNumSearch()));
    }

    struct TChannelData {
        TChannelData()
            : Specs(TAtrac1Data::NumSamples)
            , Loudness(0.0)
        {}

        vector<float> Specs;
        float Loudness;
    };

    using TData = vector<TChannelData>;
    auto buf = std::make_shared<TData>(srcChannels);
    
    // Frame counter for debug logging
    static uint32_t frame_counter = 0;

    return [this, srcChannels, buf](float* data, const TPCMEngine::ProcessMeta& /*meta*/) {
        TAtrac1Data::TBlockSizeMod blockSz[2];

        // Log buffer state at start of frame for tracking  
        ATRAC_LOG_STAGE("BUFFER_INITIAL_LOW", "state", &PcmBufLow[0][128], 32, 
                       0, frame_counter, "LOW",
                       algorithm="buffer_tracking", buffer_type="PcmBufLow", operation="frame_start_state");
        ATRAC_LOG_STAGE("BUFFER_INITIAL_MID", "state", &PcmBufMid[0][128], 32,
                       0, frame_counter, "MID", 
                       algorithm="buffer_tracking", buffer_type="PcmBufMid", operation="frame_start_state");
        ATRAC_LOG_STAGE("BUFFER_INITIAL_HIGH", "state", &PcmBufHi[0][256], 32,
                       0, frame_counter, "HIGH",
                       algorithm="buffer_tracking", buffer_type="PcmBufHi", operation="frame_start_state");

        uint32_t windowMasks[2] = {0};
        for (uint32_t channel = 0; channel < srcChannels; channel++) {
            float src[TAtrac1Data::NumSamples];
            for (size_t i = 0; i < TAtrac1Data::NumSamples; ++i) {
                src[i] = data[i * srcChannels + channel];
            }
            
            // Log input PCM samples
            ATRAC_LOG_PCM_INPUT(src, TAtrac1Data::NumSamples, channel, frame_counter);

            AnalysisFilterBank[channel].Analysis(&src[0], &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0], channel, frame_counter);
            
            // Log QMF analysis outputs
            ATRAC_LOG_QMF_OUTPUT(PcmBufLow[channel], 128, channel, frame_counter, "LOW", "low");
            ATRAC_LOG_QMF_OUTPUT(PcmBufMid[channel], 128, channel, frame_counter, "MID", "mid");
            ATRAC_LOG_QMF_OUTPUT(PcmBufHi[channel], 256, channel, frame_counter, "HIGH", "high");

            uint32_t& windowMask = windowMasks[channel];
            if (Settings.GetWindowMode() == TAtrac1EncodeSettings::EWindowMode::EWM_AUTO) {
                windowMask |= (uint32_t)TransientDetectors.GetDetector(channel, 0).Detect(&PcmBufLow[channel][0], channel, frame_counter, "LOW");

                const vector<float>& invMid = InvertSpectr<128>(&PcmBufMid[channel][0]);
                windowMask |= (uint32_t)TransientDetectors.GetDetector(channel, 1).Detect(&invMid[0], channel, frame_counter, "MID") << 1;

                const vector<float>& invHi = InvertSpectr<256>(&PcmBufHi[channel][0]);
                windowMask |= (uint32_t)TransientDetectors.GetDetector(channel, 2).Detect(&invHi[0], channel, frame_counter, "HIGH") << 2;

                //std::cout << "trans: " << windowMask << std::endl;
            } else {
                //no transient detection, use given mask
                windowMask = Settings.GetWindowMask();
            }

            blockSz[channel]  = TAtrac1Data::TBlockSizeMod(windowMask & 0x1, windowMask & 0x2, windowMask & 0x4); //low, mid, hi
            
            // Log transient detection results and BSM values
            bool transients[3] = {
                static_cast<bool>(windowMask & 0x1),
                static_cast<bool>(windowMask & 0x2), 
                static_cast<bool>(windowMask & 0x4)
            };
            ATRAC_LOG_TRANSIENT_DETECT(transients, 3, channel, frame_counter);
            
            // Convert to atracdenc BSM format (0=short, 2/3=long)
            float bsm_values[3] = {
                (windowMask & 0x1) ? 0.0f : 2.0f,  // low
                (windowMask & 0x2) ? 0.0f : 2.0f,  // mid
                (windowMask & 0x4) ? 0.0f : 3.0f   // high
            };
            ATRAC_LOG_BSM_VALUES(bsm_values, 3, channel, frame_counter);

            auto& specs = (*buf)[channel].Specs;
            
            // Set debug context for MDCT logging
            SetDebugContext(channel, frame_counter);

            Mdct(&specs[0], &PcmBufLow[channel][0], &PcmBufMid[channel][0], &PcmBufHi[channel][0], blockSz[channel]);

            float l = 0.0;
            for (size_t i = 0; i < specs.size(); i++) {
                float e = specs[i] * specs[i];
                l += e * LoudnessCurve[i];
            }
            (*buf)[channel].Loudness = l;
        }

        if (srcChannels == 2 && windowMasks[0] == 0 && windowMasks[1] == 0) {
            Loudness = TrackLoudness(Loudness, (*buf)[0].Loudness, (*buf)[1].Loudness);
        } else if (windowMasks[0] == 0) {
            Loudness = TrackLoudness(Loudness, (*buf)[0].Loudness);
        }

        for (uint32_t channel = 0; channel < srcChannels; channel++) {
            // Log final spectral data before bit allocation
            ATRAC_LOG_STAGE("SPECTRAL_COMBINED", "coeffs", (*buf)[channel].Specs.data(), 512, 
                           channel, frame_counter, "", algorithm=spectrum_combination);
            
            BitAllocs[channel]->Write(Scaler.ScaleFrame((*buf)[channel].Specs, blockSz[channel]), blockSz[channel], Loudness / LoudFactor);
        }
        
        // Increment frame counter for next call
        frame_counter++;

        return TPCMEngine::EProcessResult::PROCESSED;
    };
}

} //namespace NAtracDEnc
