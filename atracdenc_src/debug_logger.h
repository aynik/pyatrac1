/*
 * Debug logging system for atracdenc signal processing analysis.
 * Provides comprehensive metadata and source tracking for cross-comparison with PyATRAC1.
 * 
 * This file is part of AtracDEnc.
 */

#ifndef DEBUG_LOGGER_H
#define DEBUG_LOGGER_H

#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>

// Debug logging control - can be disabled by defining ATRACDENC_NO_DEBUG_LOG
#ifndef ATRACDENC_NO_DEBUG_LOG
    #define ATRACDENC_DEBUG_ENABLED 1
#else
    #define ATRACDENC_DEBUG_ENABLED 0
#endif

#if ATRACDENC_DEBUG_ENABLED

// Get current timestamp with microsecond precision
inline std::string get_debug_timestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    
    // Convert to ISO format
    auto seconds = timestamp / 1000000;
    auto microseconds = timestamp % 1000000;
    auto time_t_seconds = static_cast<time_t>(seconds);
    
    std::tm* tm_info = std::localtime(&time_t_seconds);
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%dT%H:%M:%S.");
    oss << std::setfill('0') << std::setw(6) << microseconds;
    
    return oss.str();
}

// Get basename of file path for cleaner logging
inline const char* get_basename(const char* path) {
    const char* last_slash = strrchr(path, '/');
    if (last_slash) {
        return last_slash + 1;
    }
    last_slash = strrchr(path, '\\');
    if (last_slash) {
        return last_slash + 1;
    }
    return path;
}

// Main logging macro for arrays/vectors with comprehensive metadata
#define ATRAC_LOG_STAGE(stage, data_type, values_ptr, size, channel, frame, band, ...) \
    do { \
        int size_int = (int)(size); \
        if (size_int > 0) { \
            const float* values = (const float*)(values_ptr); \
            double min_val = *std::min_element(values, values + size_int); \
            double max_val = *std::max_element(values, values + size_int); \
            double sum_val = std::accumulate(values, values + size_int, 0.0); \
            double mean_val = sum_val / size_int; \
            int nonzero = std::count_if(values, values + size_int, [](float x) { return x != 0.0f; }); \
            \
            printf("[%s][ATRACDENC][%s:%d][%s][CH%d][FR%03d]%s%s%s %s: %s=", \
                   get_debug_timestamp().c_str(), get_basename(__FILE__), __LINE__, __func__, \
                   channel, frame, (strlen(band) > 0 ? "[BAND_" : ""), (strlen(band) > 0 ? band : ""), \
                   (strlen(band) > 0 ? "]" : ""), stage, data_type); \
            \
            /* Print first 5 and last 5 values if array is large */ \
            if (size_int <= 10) { \
                printf("["); \
                for(int i = 0; i < size_int; i++) { \
                    printf("%.6f%s", (double)values[i], (i < size_int-1) ? "," : ""); \
                } \
                printf("]"); \
            } else { \
                printf("["); \
                for(int i = 0; i < 5; i++) printf("%.6f,", (double)values[i]); \
                printf("..."); \
                for(int i = size_int-5; i < size_int; i++) printf("%.6f%s", (double)values[i], (i < size_int-1) ? "," : ""); \
                printf("]"); \
            } \
            \
            printf(" |META: size=%d range=[%.6f,%.6f] sum=%.6f mean=%.6f nonzero=%d", \
                   size_int, min_val, max_val, sum_val, mean_val, nonzero); \
            printf(" |SRC: " #__VA_ARGS__ "\n"); \
        } \
    } while(0)

// Specialized logging macro for single values
#define ATRAC_LOG_VALUE(stage, data_type, value, channel, frame, band, ...) \
    do { \
        printf("[%s][ATRACDENC][%s:%d][%s][CH%d][FR%03d]%s%s %s: %s=%.6f", \
               get_debug_timestamp().c_str(), get_basename(__FILE__), __LINE__, __func__, \
               channel, frame, (strlen(band) > 0 ? "[BAND_" : ""), (strlen(band) > 0 ? band : ""), \
               stage, data_type, (double)value); \
        printf(" |META: size=1 range=[%.6f,%.6f] sum=%.6f mean=%.6f nonzero=%d", \
               (double)value, (double)value, (double)value, (double)value, (value != 0.0 ? 1 : 0)); \
        printf(" |SRC: " #__VA_ARGS__ "\n"); \
    } while(0)

// Logging macro for bitstream data (hex format)
#define ATRAC_LOG_BITSTREAM(stage, data, size_bytes, channel, frame, ...) \
    do { \
        printf("[%s][ATRACDENC][%s:%d][%s][CH%d][FR%03d] %s: hex=", \
               get_debug_timestamp().c_str(), get_basename(__FILE__), __LINE__, __func__, \
               channel, frame, stage); \
        for(int i = 0; i < size_bytes; i++) { \
            printf("%02x", ((unsigned char*)data)[i]); \
        } \
        printf(" |META: size=%d bytes |SRC: " #__VA_ARGS__ "\n", size_bytes); \
    } while(0)

// Logging macro for boolean arrays/decisions
#define ATRAC_LOG_BOOL_ARRAY(stage, data_type, values, size, channel, frame, ...) \
    do { \
        printf("[%s][ATRACDENC][%s:%d][%s][CH%d][FR%03d] %s: %s=[", \
               get_debug_timestamp().c_str(), get_basename(__FILE__), __LINE__, __func__, \
               channel, frame, stage, data_type); \
        for(int i = 0; i < size; i++) { \
            printf("%.6f%s", (values[i] ? 1.0 : 0.0), (i < size-1) ? "," : ""); \
        } \
        int true_count = 0; \
        for(int i = 0; i < size; i++) if(values[i]) true_count++; \
        printf("] |META: size=%d range=[0.000000,1.000000] sum=%.6f mean=%.6f nonzero=%d", \
               size, (double)true_count, (double)true_count/size, true_count); \
        printf(" |SRC: " #__VA_ARGS__ "\n"); \
    } while(0)

// Convenience macros for common scenarios
#define ATRAC_LOG_PCM_INPUT(samples, size, channel, frame) \
    ATRAC_LOG_STAGE("PCM_INPUT", "samples", samples, size, channel, frame, "", algorithm=encode_frame)

#define ATRAC_LOG_QMF_OUTPUT(samples, size, channel, frame, band_name, qmf_band_name) \
    ATRAC_LOG_STAGE("QMF_OUTPUT", "samples", samples, size, channel, frame, band_name, algorithm=qmf_analysis qmf_band=qmf_band_name)

#define ATRAC_LOG_MDCT_INPUT(samples, size, channel, frame, band_name, mdct_size, window_type) \
    ATRAC_LOG_STAGE("MDCT_INPUT", "samples", samples, size, channel, frame, band_name, algorithm=mdct window_type=window_type mdct_size=mdct_size)

#define ATRAC_LOG_MDCT_OUTPUT(coeffs, size, channel, frame, band_name, mdct_size, window_type) \
    ATRAC_LOG_STAGE("MDCT_OUTPUT", "coeffs", coeffs, size, channel, frame, band_name, algorithm=mdct window_type=window_type mdct_size=mdct_size)

#define ATRAC_LOG_TRANSIENT_DETECT(decisions, size, channel, frame) \
    ATRAC_LOG_BOOL_ARRAY("TRANSIENT_DETECT", "decision", decisions, size, channel, frame, algorithm=transient_detection)

#define ATRAC_LOG_BSM_VALUES(bsm_values, size, channel, frame) \
    ATRAC_LOG_STAGE("BSM_VALUES", "bsm", bsm_values, size, channel, frame, "", algorithm=block_size_mode)

#define ATRAC_LOG_BIT_ALLOC(word_lengths, size, channel, frame, bits_used) \
    ATRAC_LOG_STAGE("BIT_ALLOC", "word_lengths", word_lengths, size, channel, frame, "", algorithm=bit_allocation mantissa_bits_used=bits_used)

#define ATRAC_LOG_QUANTIZE(mantissas, size, channel, frame, bfu_idx, word_length) \
    ATRAC_LOG_STAGE("QUANTIZE", "mantissas", mantissas, size, channel, frame, "", algorithm=quantization bfu_idx=bfu_idx word_length=word_length)

#define ATRAC_LOG_BITSTREAM_OUTPUT(data, size_bytes, channel, frame) \
    ATRAC_LOG_BITSTREAM("BITSTREAM_OUTPUT", data, size_bytes, channel, frame, algorithm=bitstream_writer)

#else // ATRACDENC_DEBUG_ENABLED == 0

// All logging macros become no-ops when debugging is disabled
#define ATRAC_LOG_STAGE(stage, data_type, values, size, channel, frame, band, ...) do {} while(0)
#define ATRAC_LOG_VALUE(stage, data_type, value, channel, frame, band, ...) do {} while(0)
#define ATRAC_LOG_BITSTREAM(stage, data, size_bytes, channel, frame, ...) do {} while(0)
#define ATRAC_LOG_BOOL_ARRAY(stage, data_type, values, size, channel, frame, ...) do {} while(0)
#define ATRAC_LOG_PCM_INPUT(samples, size, channel, frame) do {} while(0)
#define ATRAC_LOG_QMF_OUTPUT(samples, size, channel, frame, band_name, qmf_band_name) do {} while(0)
#define ATRAC_LOG_MDCT_INPUT(samples, size, channel, frame, band_name, mdct_size, window_type) do {} while(0)
#define ATRAC_LOG_MDCT_OUTPUT(coeffs, size, channel, frame, band_name, mdct_size, window_type) do {} while(0)
#define ATRAC_LOG_TRANSIENT_DETECT(decisions, size, channel, frame) do {} while(0)
#define ATRAC_LOG_BSM_VALUES(bsm_values, size, channel, frame) do {} while(0)
#define ATRAC_LOG_BIT_ALLOC(word_lengths, size, channel, frame, bits_used) do {} while(0)
#define ATRAC_LOG_QUANTIZE(mantissas, size, channel, frame, bfu_idx, word_length) do {} while(0)
#define ATRAC_LOG_BITSTREAM_OUTPUT(data, size_bytes, channel, frame) do {} while(0)

#endif // ATRACDENC_DEBUG_ENABLED

#endif // DEBUG_LOGGER_H