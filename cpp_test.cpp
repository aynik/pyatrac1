#include "atracdenc_src/atrac/atrac1_qmf.h"
#include "atracdenc_src/lib/mdct/mdct.h"
// #include "atracdenc_src/atrac/atrac1.h" // BLOCK_SIZE_SHORT is not defined here
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath> // For std::sin
#include <cstring> // For memcpy
#include <numeric> // For std::iota (debugging)
#include <algorithm> // For std::copy

// Define block sizes here as they are not provided by the library headers
// Aligned with pyatrac1 short block mode: Low/Mid=128, High=64 for MDCT N
const size_t BLOCK_SIZE_MDCT_LOW_MID = 128;
const size_t BLOCK_SIZE_MDCT_HIGH = 64;
// const size_t BLOCK_SIZE_LONG = 256; // Defined for completeness, not used in this test

// Helper function to write a vector of floats to a file
void write_vector_to_file(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    outfile << std::fixed << std::setprecision(8);
    for (const auto& val : data) {
        outfile << val << std::endl;
    }
    outfile.close();
    std::cout << "Data written to " << filename << std::endl;
}

int main() {
    std::cout << "Starting C++ test for ATRAC QMF and MDCT (aligned MDCT sizes)..." << std::endl;

    // 1. Define a sample input
    const int num_samples = 1024;
    std::vector<float> input_samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        input_samples[i] = std::sin(2.0f * M_PI * static_cast<float>(i) / static_cast<float>(num_samples / 4));
    }
    std::cout << "Generated " << num_samples << " input samples." << std::endl;
    // write_vector_to_file(input_samples, "cpp_input_samples.txt"); // Already generated

    // 3. Perform Filter Bank Analysis
    std::cout << "\n--- Performing Filter Bank Analysis (QMF) ---" << std::endl;
    NAtracDEnc::Atrac1AnalysisFilterBank<float> analysis_qmf;

    const int QMF_INPUT_BLOCK_SIZE = 512;
    const int QMF_OUTPUT_LOW_MID_SIZE = QMF_INPUT_BLOCK_SIZE / 4; // 128
    const int QMF_OUTPUT_HIGH_SIZE = QMF_INPUT_BLOCK_SIZE / 2;    // 256

    std::vector<float> qmf_subband_0, qmf_subband_1, qmf_subband_2;
    qmf_subband_0.reserve(num_samples / QMF_INPUT_BLOCK_SIZE * QMF_OUTPUT_LOW_MID_SIZE);
    qmf_subband_1.reserve(num_samples / QMF_INPUT_BLOCK_SIZE * QMF_OUTPUT_LOW_MID_SIZE);
    qmf_subband_2.reserve(num_samples / QMF_INPUT_BLOCK_SIZE * QMF_OUTPUT_HIGH_SIZE);

    std::vector<float> temp_low(QMF_OUTPUT_LOW_MID_SIZE);
    std::vector<float> temp_mid(QMF_OUTPUT_LOW_MID_SIZE);
    std::vector<float> temp_high(QMF_OUTPUT_HIGH_SIZE);

    for (int i = 0; i < num_samples; i += QMF_INPUT_BLOCK_SIZE) {
        if (i + QMF_INPUT_BLOCK_SIZE > num_samples) {
            std::cerr << "Warning: Not enough remaining samples for a full QMF block. Skipping partial block." << std::endl;
            break;
        }
        // std::cout << "Processing QMF Analysis block starting at sample " << i << std::endl;
        analysis_qmf.Analysis(&input_samples[i], temp_low.data(), temp_mid.data(), temp_high.data());

        qmf_subband_0.insert(qmf_subband_0.end(), temp_low.begin(), temp_low.end());
        qmf_subband_1.insert(qmf_subband_1.end(), temp_mid.begin(), temp_mid.end());
        qmf_subband_2.insert(qmf_subband_2.end(), temp_high.begin(), temp_high.end());
    }

    std::cout << "QMF analysis complete." << std::endl;
    // write_vector_to_file(qmf_subband_0, "cpp_qmf_subband_0.txt"); // Already generated
    // write_vector_to_file(qmf_subband_1, "cpp_qmf_subband_1.txt");
    // write_vector_to_file(qmf_subband_2, "cpp_qmf_subband_2.txt");

    // 4. Perform MDCT
    std::cout << "\n--- Performing MDCT ---" << std::endl;
    std::cout << "Using MDCT block sizes: Low/Mid=" << BLOCK_SIZE_MDCT_LOW_MID << ", High=" << BLOCK_SIZE_MDCT_HIGH << std::endl;

    if (QMF_OUTPUT_LOW_MID_SIZE % BLOCK_SIZE_MDCT_LOW_MID != 0 ) { // 128 % 128 == 0
         std::cerr << "Error: QMF Low/Mid output size (" << QMF_OUTPUT_LOW_MID_SIZE
                   << ") not perfectly divisible by MDCT block size " << BLOCK_SIZE_MDCT_LOW_MID << std::endl;
         return 1;
    }
    // QMF_OUTPUT_HIGH_SIZE is 256. BLOCK_SIZE_MDCT_HIGH is 64. 256 % 64 == 0.
    if (QMF_OUTPUT_HIGH_SIZE % BLOCK_SIZE_MDCT_HIGH != 0 ) {
         std::cerr << "Error: QMF High output size (" << QMF_OUTPUT_HIGH_SIZE
                   << ") not perfectly divisible by MDCT block size " << BLOCK_SIZE_MDCT_HIGH << std::endl;
         return 1;
    }

    NMDCT::TMDCT<BLOCK_SIZE_MDCT_LOW_MID, float> mdct_0;
    NMDCT::TMDCT<BLOCK_SIZE_MDCT_LOW_MID, float> mdct_1;
    NMDCT::TMDCT<BLOCK_SIZE_MDCT_HIGH, float> mdct_2;

    std::vector<float> mdct_coeffs_0, mdct_coeffs_1, mdct_coeffs_2;
    // MDCT output N coefficients for N input samples (TMDCT in this lib seems to mean N/2 output for N input)
    // The N in TMDCT<N, float> is the transform size. Output is N coeffs.
    // For TMDCT, input is N points, output is N/2 points.
    // For TMIDCT, input is N/2 points, output is N points.
    // The current NMDCT::TMDCT and TMIDCT in this codebase are N input -> N output.
    // This was clarified by `mdct_coeffs_X.resize(qmf_subband_X.size())` and it worked.
    // So, if N=128, output is 128 coeffs. If N=64, output is 64 coeffs.
    // The number of coefficients per QMF frame part will change for subband 2.
    // QMF subband 0 size: 256. MDCT N=128. Num blocks = 256/128 = 2. Total coeffs = 2 * 128 = 256.
    // QMF subband 1 size: 256. MDCT N=128. Num blocks = 256/128 = 2. Total coeffs = 2 * 128 = 256.
    // QMF subband 2 size: 512. MDCT N=64. Num blocks = 512/64 = 8. Total coeffs = 8 * 64 = 512.
    // So total lengths of mdct_coeffs vectors remain the same as qmf_subband vectors.

    mdct_coeffs_0.resize(qmf_subband_0.size());
    mdct_coeffs_1.resize(qmf_subband_1.size());
    mdct_coeffs_2.resize(qmf_subband_2.size());

    for (size_t i = 0; i < qmf_subband_0.size(); i += BLOCK_SIZE_MDCT_LOW_MID) {
        const auto& coeffs_block = mdct_0(&qmf_subband_0[i]); // Returns std::vector of size N (template param)
        std::copy(coeffs_block.begin(), coeffs_block.end(), &mdct_coeffs_0[i]);
    }
    for (size_t i = 0; i < qmf_subband_1.size(); i += BLOCK_SIZE_MDCT_LOW_MID) {
        const auto& coeffs_block = mdct_1(&qmf_subband_1[i]);
        std::copy(coeffs_block.begin(), coeffs_block.end(), &mdct_coeffs_1[i]);
    }
    for (size_t i = 0; i < qmf_subband_2.size(); i += BLOCK_SIZE_MDCT_HIGH) { // Use new block size for high band
        const auto& coeffs_block = mdct_2(&qmf_subband_2[i]);
        std::copy(coeffs_block.begin(), coeffs_block.end(), &mdct_coeffs_2[i]);
    }

    std::cout << "MDCT processing complete." << std::endl;
    write_vector_to_file(mdct_coeffs_0, "cpp_mdct_coeffs_0.txt");
    write_vector_to_file(mdct_coeffs_1, "cpp_mdct_coeffs_1.txt");
    write_vector_to_file(mdct_coeffs_2, "cpp_mdct_coeffs_2.txt");

    // 5. Perform IMDCT
    std::cout << "\n--- Performing IMDCT ---" << std::endl;
    NMDCT::TMIDCT<BLOCK_SIZE_MDCT_LOW_MID, float> imdct_0;
    NMDCT::TMIDCT<BLOCK_SIZE_MDCT_LOW_MID, float> imdct_1;
    NMDCT::TMIDCT<BLOCK_SIZE_MDCT_HIGH, float> imdct_2; // Use new block size for high band

    std::vector<float> imdct_output_0, imdct_output_1, imdct_output_2;
    imdct_output_0.resize(mdct_coeffs_0.size());
    imdct_output_1.resize(mdct_coeffs_1.size());
    imdct_output_2.resize(mdct_coeffs_2.size());

    for (size_t i = 0; i < mdct_coeffs_0.size(); i += BLOCK_SIZE_MDCT_LOW_MID) {
        const auto& block_out = imdct_0(&mdct_coeffs_0[i]);
        std::copy(block_out.begin(), block_out.end(), &imdct_output_0[i]);
    }
    for (size_t i = 0; i < mdct_coeffs_1.size(); i += BLOCK_SIZE_MDCT_LOW_MID) {
        const auto& block_out = imdct_1(&mdct_coeffs_1[i]);
        std::copy(block_out.begin(), block_out.end(), &imdct_output_1[i]);
    }
    for (size_t i = 0; i < mdct_coeffs_2.size(); i += BLOCK_SIZE_MDCT_HIGH) { // Use new block size
        const auto& block_out = imdct_2(&mdct_coeffs_2[i]);
        std::copy(block_out.begin(), block_out.end(), &imdct_output_2[i]);
    }
    std::cout << "IMDCT processing complete." << std::endl;
    write_vector_to_file(imdct_output_0, "cpp_imdct_output_0.txt");
    write_vector_to_file(imdct_output_1, "cpp_imdct_output_1.txt");
    write_vector_to_file(imdct_output_2, "cpp_imdct_output_2.txt");

    // 6. Perform Filter Bank Synthesis
    std::cout << "\n--- Performing Filter Bank Synthesis (IQMF) ---" << std::endl;
    NAtracDEnc::Atrac1SynthesisFilterBank<float> synthesis_qmf;

    std::vector<float> reconstructed_output;
    reconstructed_output.reserve(num_samples);

    std::vector<float> current_low_block(QMF_OUTPUT_LOW_MID_SIZE);
    std::vector<float> current_mid_block(QMF_OUTPUT_LOW_MID_SIZE);
    std::vector<float> current_high_block(QMF_OUTPUT_HIGH_SIZE);
    std::vector<float> temp_pcm_out_block(QMF_INPUT_BLOCK_SIZE);

    size_t low_idx = 0;
    size_t mid_idx = 0;
    size_t high_idx = 0;

    for (int block = 0; block < (num_samples / QMF_INPUT_BLOCK_SIZE); ++block) {
        // std::cout << "Processing QMF Synthesis block " << block << std::endl;
        if (low_idx + QMF_OUTPUT_LOW_MID_SIZE > imdct_output_0.size() ||
            mid_idx + QMF_OUTPUT_LOW_MID_SIZE > imdct_output_1.size() ||
            high_idx + QMF_OUTPUT_HIGH_SIZE > imdct_output_2.size()) {
            std::cerr << "Error: Not enough data in subbands for QMF synthesis block " << block << std::endl; // Corrected std::endl
            break;
        }

        memcpy(current_low_block.data(), &imdct_output_0[low_idx], QMF_OUTPUT_LOW_MID_SIZE * sizeof(float));
        memcpy(current_mid_block.data(), &imdct_output_1[mid_idx], QMF_OUTPUT_LOW_MID_SIZE * sizeof(float));
        memcpy(current_high_block.data(), &imdct_output_2[high_idx], QMF_OUTPUT_HIGH_SIZE * sizeof(float));

        synthesis_qmf.Synthesis(temp_pcm_out_block.data(), current_low_block.data(), current_mid_block.data(), current_high_block.data());

        reconstructed_output.insert(reconstructed_output.end(), temp_pcm_out_block.begin(), temp_pcm_out_block.end());

        low_idx += QMF_OUTPUT_LOW_MID_SIZE;
        mid_idx += QMF_OUTPUT_LOW_MID_SIZE;
        high_idx += QMF_OUTPUT_HIGH_SIZE;
    }

    std::cout << "QMF synthesis complete." << std::endl;
    // std::cout << "Reconstructed output size: " << reconstructed_output.size() << std::endl;
    // if (reconstructed_output.size() != num_samples) {
    //      std::cout << "Warning: Reconstructed output size (" << reconstructed_output.size()
    //                << ") does not match original input size (" << num_samples << ")" << std::endl;
    // }
    write_vector_to_file(reconstructed_output, "cpp_reconstructed_output.txt");

    std::cout << "\nC++ test finished." << std::endl;

    return 0;
}
