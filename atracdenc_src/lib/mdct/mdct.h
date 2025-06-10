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

#include "config.h"
#include <lib/fft/kissfft_impl/kiss_fft.h>
#include <vector>
#include <type_traits>
#include "../../debug_logger.h"  // For granular MDCT algorithm debugging

namespace NMDCT {

static_assert(sizeof(kiss_fft_scalar) == sizeof(float), "size of fft_scalar is not equal to size of float");

class TMDCTBase {
protected:
    const size_t N;
    const float Scale;  // Store scale for debugging
    const std::vector<float> SinCos;
    kiss_fft_cpx*   FFTIn;
    kiss_fft_cpx*   FFTOut;
    kiss_fft_cfg    FFTPlan;
    TMDCTBase(size_t n, float scale);
    virtual ~TMDCTBase();
};


template<size_t TN, typename TIO = float>
class TMDCT : public TMDCTBase {
    std::vector<TIO> Buf;
public:
    TMDCT(float scale = 1.0)
        : TMDCTBase(TN, scale)
        , Buf(TN/2)
    {
        // Log scale parameter for debugging
        ATRAC_LOG_STAGE("MDCT_SCALE_PARAM", "value", &scale, 1,
                       0, 0, "DEBUG", algorithm="mdct_init", operation="scale_param", mdct_size=TN);
    }
    const std::vector<TIO>& operator()(const TIO* in) {

        const size_t n2 = N >> 1;
        const size_t n4 = N >> 2;
        const size_t n34 = 3 * n4;
        const size_t n54 = 5 * n4;
        const float* cos = &SinCos[0];
        const float* sin = &SinCos[1];

        float  *xr, *xi, r0, i0;
        float  c, s;
        size_t n;

        xr = (float*)FFTIn;
        xi = (float*)FFTIn + 1;
        
        // Log first 4 pre-rotation calculations for debugging
        std::vector<float> pre_rotation_debug;
        
        for (n = 0; n < n4; n += 2) {
            r0 = in[n34 - 1 - n] + in[n34 + n];
            i0 = in[n4 + n] - in[n4 - 1 - n];

            c = cos[n];
            s = sin[n];

            xr[n] = r0 * c + i0 * s;
            xi[n] = i0 * c - r0 * s;
            
            // Collect debug data for first few samples
            if (n < 8) {
                pre_rotation_debug.push_back(r0);
                pre_rotation_debug.push_back(i0);
                pre_rotation_debug.push_back(c);
                pre_rotation_debug.push_back(s);
                pre_rotation_debug.push_back(xr[n]);
                pre_rotation_debug.push_back(xi[n]);
            }
        }
        
        // Log scale parameter for this transform
        ATRAC_LOG_STAGE("MDCT_SCALE_CURRENT", "value", &Scale, 1,
                       0, 0, "DEBUG", algorithm="mdct_internal", operation="current_scale", transform_size=TN);
        
        // Log pre-rotation stage
        ATRAC_LOG_STAGE("MDCT_PRE_ROTATION", "values", pre_rotation_debug.data(), std::min(24, (int)pre_rotation_debug.size()),
                       0, 0, "DEBUG", algorithm="mdct_internal", operation="pre_rotation");

        for (; n < n2; n += 2) {
            r0 = in[n34 - 1 - n] - in[n - n4];
            i0 = in[n4 + n]    + in[n54 - 1 - n];

            c = cos[n];
            s = sin[n];

            xr[n] = r0 * c + i0 * s;
            xi[n] = i0 * c - r0 * s;
        }

        // Log FFT input
        ATRAC_LOG_STAGE("MDCT_FFT_INPUT", "complex", (float*)FFTIn, std::min(16, (int)n2),
                       0, 0, "DEBUG", algorithm="mdct_internal", operation="fft_input");

        kiss_fft(FFTPlan, FFTIn, FFTOut);
        
        // Log FFT output
        ATRAC_LOG_STAGE("MDCT_FFT_OUTPUT", "complex", (float*)FFTOut, std::min(16, (int)n2),
                       0, 0, "DEBUG", algorithm="mdct_internal", operation="fft_output");

        xr = (float*)FFTOut;
        xi = (float*)FFTOut + 1;
        
        // Log first 4 post-rotation calculations for debugging
        std::vector<float> post_rotation_debug;
        
        for (n = 0; n < n2; n += 2) {
            r0 = xr[n];
            i0 = xi[n];

            c = cos[n];
            s = sin[n];

            Buf[n] = - r0 * c - i0 * s;
            Buf[n2 - 1 -n] = - r0 * s + i0 * c;
            
            // Collect debug data for first few samples
            if (n < 8) {
                post_rotation_debug.push_back(r0);
                post_rotation_debug.push_back(i0);
                post_rotation_debug.push_back(c);
                post_rotation_debug.push_back(s);
                post_rotation_debug.push_back(Buf[n]);
                post_rotation_debug.push_back(Buf[n2 - 1 - n]);
            }
        }
        
        // Log post-rotation stage
        ATRAC_LOG_STAGE("MDCT_POST_ROTATION", "values", post_rotation_debug.data(), std::min(24, (int)post_rotation_debug.size()),
                       0, 0, "DEBUG", algorithm="mdct_internal", operation="post_rotation");

        return Buf;
    }
};

template<size_t TN, typename TIO = float>
class TMIDCT : public TMDCTBase {
    std::vector<TIO> Buf;
public:
    TMIDCT(float scale = TN)
        : TMDCTBase(TN, scale/2)
        , Buf(TN)
    {
        // Log scale parameters for debugging
        float original_scale = scale;
        float adjusted_scale = scale/2;
        ATRAC_LOG_STAGE("IMDCT_SCALE_PARAM", "values", &original_scale, 1,
                       0, 0, "DEBUG", algorithm="imdct_init", operation="original_scale", imdct_size=TN);
        ATRAC_LOG_STAGE("IMDCT_SCALE_ADJUSTED", "values", &adjusted_scale, 1,
                       0, 0, "DEBUG", algorithm="imdct_init", operation="adjusted_scale", imdct_size=TN);
    }
    const std::vector<TIO>& operator()(const TIO* in) {

        const size_t n2 = N >> 1;
        const size_t n4 = N >> 2;
        const size_t n34 = 3 * n4;
        const size_t n54 = 5 * n4;
        const float* cos = &SinCos[0];
        const float* sin = &SinCos[1];

        float *xr, *xi, r0, i0, r1, i1;
        float c, s;
        size_t n;

        xr = (float*)FFTIn;
        xi = (float*)FFTIn + 1;

        for (n = 0; n < n2; n += 2) {
            r0 = in[n];
            i0 = in[n2 - 1 - n];

            c = cos[n];
            s = sin[n];

            xr[n] = -2.0 * (i0 * s + r0 * c);
            xi[n] = -2.0 * (i0 * c - r0 * s);
        }

        kiss_fft(FFTPlan, FFTIn, FFTOut);

        xr = (float*)FFTOut;
        xi = (float*)FFTOut + 1;

        for (n = 0; n < n4; n += 2) {
            r0 = xr[n];
            i0 = xi[n];

            c = cos[n];
            s = sin[n];

            r1 = r0 * c + i0 * s;
            i1 = r0 * s - i0 * c;

            Buf[n34 - 1 - n] = r1;
            Buf[n34 + n] = r1;
            Buf[n4 + n] = i1;
            Buf[n4 - 1 - n] = -i1;
        }

        for (; n < n2; n += 2) {
            r0 = xr[n];
            i0 = xi[n];

            c = cos[n];
            s = sin[n];

            r1 = r0 * c + i0 * s;
            i1 = r0 * s - i0 * c;

            Buf[n34 - 1 - n] = r1;
            Buf[n - n4] = -r1;
            Buf[n4 + n] = i1;
            Buf[n54 - 1 - n] = i1;
        }
        return Buf;
    }
};

} //namespace NMDCT
