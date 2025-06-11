// Compile with: g++ imdct_test.cpp atracdenc_src/atrac1denc.cpp atracdenc_src/lib/mdct/mdct.cpp atracdenc_src/atrac/atrac1.cpp -I./ -Iatracdenc_src/ -std=c++11 -DATRAC_ENABLE_DEBUG_LOGGING -o imdct_test_executable

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // For std::fixed, std::setprecision (if needed for printing floats)
#include <numeric> // For std::iota (if needed)
#include <algorithm> // For std::copy

// AtracDEnc headers
#include "atracdenc_src/debug_logger.h" // For SetDebugContext (should be first for global settings if any)
#include "atracdenc_src/atrac1denc.h" // For TAtrac1MDCT
#include "atracdenc_src/atrac/atrac1.h"    // For TAtrac1Data::TBlockSizeMod


// Global static persistent buffers (for one channel, zero-initialized by default for static storage duration)
static float PcmBufLow[256];
static float PcmBufMid[256];
static float PcmBufHi[512];

// Global static MDCT object (for one channel)
static NAtracDEnc::TAtrac1MDCT Mdc;

// Helper function to read float values from a text file
std::vector<float> load_floats_from_file(const std::string& filename) {
    std::vector<float> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data; // Return empty vector on error
    }
    float value;
    while (infile >> value) {
        data.push_back(value);
    }
    if (infile.bad()) {
        std::cerr << "Error: Reading from file " << filename << " failed." << std::endl;
        // Potentially clear data or handle error differently
    }
    infile.close();
    return data;
}

int main() {
    // Load coefficients from files
    std::vector<float> coeffs_ch0_low = load_floats_from_file("cpp_mdct_coeffs_0.txt");
    std::vector<float> coeffs_ch0_mid = load_floats_from_file("cpp_mdct_coeffs_1.txt");
    std::vector<float> coeffs_ch0_high = load_floats_from_file("cpp_mdct_coeffs_2.txt");

    if (coeffs_ch0_low.empty() || coeffs_ch0_mid.empty() || coeffs_ch0_high.empty()) {
        std::cerr << "Error: Failed to load one or more coefficient files. Ensure files exist and are readable." << std::endl;
        return 1;
    }

    const int num_frames_to_process = 2;
    const int channel = 0;

    // Specs array for MDCT coefficients
    float Specs[512];

    // Block size mode: low_band_short=false, mid_band_short=false, high_band_short=true
    // This corresponds to TBlockSizeMod(0,0,1) if 1 means short.
    // From atrac1.h: TBlockSizeMod(bool lowShort, bool midShort, bool hiShort)
    // From subtask: (low_band_short=false, mid_band_short=false, high_band_short=true)
    // This means LogCount should be [0,0,3] if short is 0 and long is 2/3 for BSM
    // However, TBlockSizeMod constructor takes booleans directly:
    // TBlockSizeMod mode(false, false, true);
    // This will set LogCount based on its internal logic:
    // LogCount[0] = lowShort ? 2 : 0;  -> 0
    // LogCount[1] = midShort ? 2 : 0;  -> 0
    // LogCount[2] = hiShort ? 3 : 0;   -> 3
    // So mode.LogCount will be [0, 0, 3]
    // The subtask description said: NAtrac1::TAtrac1Data::TBlockSizeMod mode(0, 0, 2);
    // This seems to be direct LogCount values, not constructor args.
    // Let's re-check TBlockSizeMod definition.
    // It has a constructor `TBlockSizeMod(BitStreamReader* br);` and `TBlockSizeMod(uint32_t low, uint32_t mid, uint32_t hi);`
    // The latter takes BSM values (0=short, 2/3=long).
    // If we want LogCount = [0,0,2], this means:
    // Low: long (BSM=2) -> LogCount = 0
    // Mid: long (BSM=2) -> LogCount = 0
    // High: short (BSM=0) -> LogCount = 3 (not 2)
    // If BSM for high is 0 (short), LogCount[2] becomes 3.
    // If BSM for high is 3 (long), LogCount[2] becomes 0.
    // The prompt says: "mode(0,0,2)" for LogCount = [0,0,2]. This seems to be a direct setting of LogCount members.
    // Let's assume the TBlockSizeMod constructor that takes 3 integers is for BSM values.
    // If we want LogCount = [0,0,2]:
    // Low: Long block => BSM_Low = 2
    // Mid: Long block => BSM_Mid = 2
    // High: LogCount[2] = 2 means numMdctBlocks = 1 << 2 = 4. This is for short blocks.
    // For short blocks, BSM_Hi = 0. This would make LogCount[2] = 3.
    // There seems to be a mismatch in how TBlockSizeMod is being specified vs. its constructors.
    // Let's use the boolean constructor to match "low_band_short=false, mid_band_short=false, high_band_short=true"
    NAtracDEnc::NAtrac1::TAtrac1Data::TBlockSizeMod mode(false, false, true); // low=long, mid=long, hi=short
    // This will result in LogCount = [0, 0, 3]

    // If the intention was truly LogCount = [0,0,2], that means High band has 4 MDCT blocks (1 << 2).
    // This corresponds to high_band_short = true.
    // The TBlockSizeMod calculates LogCount as:
    // LogCount[0] = lowShort ? 2 : 0;
    // LogCount[1] = midShort ? 2 : 0;
    // LogCount[2] = hiShort ? 3 : 0;  <-- This is the key. If hiShort is true, LogCount[2] is 3.
    // So, to get LogCount[2]=2 is not directly possible with this constructor logic for hi band.
    // The C++ code for TAtrac1MDCT::Mdct uses `1 << blockSize.LogCount[band]`.
    // The prompt's `mode(0,0,2)` might be a typo and intended LogCount `[0,0,3]` for hi-short, or it implies a different way to set mode.
    // Given the previous subtask context, (0,0,2) for BSM values (long, long, long for hi with special meaning for 2) might be intended.
    // If `NAtrac1::TAtrac1Data::TBlockSizeMod mode(0, 0, 2);` means BSM values:
    // BSM_Low = 0 (short) -> LogCount[0] = 2
    // BSM_Mid = 0 (short) -> LogCount[1] = 2
    // BSM_High = 2 (long, but 2 is not typical for high, usually 3 for long)
    // Let's stick to the boolean constructor as it's less ambiguous based on "low_band_short=X" phrasing.
    // The prompt said: "NAtrac1::TAtrac1Data::TBlockSizeMod mode(0, 0, 2); (This means low_band_short=false, mid_band_short=false, high_band_short=true, resulting in LogCount = [0, 0, 2])"
    // This statement is contradictory: if high_band_short=true, LogCount[2] becomes 3.
    // If LogCount = [0,0,2] is the absolute requirement, we might need to manually set mode.LogCount if possible,
    // or find a BSM combination that produces it.
    // (false, false, true) => LogCount = [0,0,3]
    // To get LogCount[2]=2: This is not possible with current TBlockSizeMod boolean constructor for high band.
    // Let's assume the text "resulting in LogCount = [0,0,2]" is the ground truth for LogCount values.
    // And TAtrac1Data::TBlockSizeMod mode(0,0,2) was a hint for BSM values.
    // BSM_Low = 0 (short) => LogCount[0] = 2
    // BSM_Mid = 0 (short) => LogCount[1] = 2
    // BSM_High = 2 (special long?)
    // If we use the BSM constructor: `NAtrac1::TAtrac1Data::TBlockSizeMod bsm_mode(0,0,2);`
    // This will interpret 0,0,2 as BSMs.
    // LogCount[0] = (LowMaskBfu[0] == bsm_low) ? 0 : 2; (from atrac1.cpp, where LowMaskBfu[0] is 2) -> (2 == 0) ? 0 : 2 -> LogCount[0] = 2
    // LogCount[1] = (LowMaskBfu[1] == bsm_mid) ? 0 : 2; (LowMaskBfu[1] is 2) -> (2 == 0) ? 0 : 2 -> LogCount[1] = 2
    // LogCount[2] = (LowMaskBfu[2] == bsm_hi) ? 0 : 3; (LowMaskBfu[2] is 3) -> (3 == 2) ? 0 : 3 -> LogCount[2] = 3
    // So `NAtrac1::TAtrac1Data::TBlockSizeMod bsm_mode(0,0,2);` yields LogCount = [2,2,3]. This is not [0,0,2].

    // Given the confusion, let's manually create a mode object and set its LogCount if that's what the test requires.
    // However, TBlockSizeMod members are typically const after construction.
    // The simplest interpretation matching "low_band_short=false, mid_band_short=false, high_band_short=true" is:
    // NAtrac1::TAtrac1Data::TBlockSizeMod mode_from_bools(false, false, true); // LogCount = [0,0,3]
    // If the test *must* have LogCount=[0,0,2], the TBlockSizeMod definition or requirement is tricky.
    // Let's assume the "resulting in LogCount = [0,0,2]" is the primary goal.
    // This implies: Low=Long (num_blocks=1), Mid=Long (num_blocks=1), High=4 blocks (num_blocks = 1 << 2).
    // This means low_short=false, mid_short=false, high_short=true (for 4/8 blocks).
    // But high_short=true gives LogCount[2]=3 (8 blocks).
    // There is no way to get LogCount[2]=2 (4 blocks for high band) with the current TBlockSizeMod boolean constructor.
    // It seems the prompt has an inconsistency for TBlockSizeMod.
    // I will proceed with `NAtracDEnc::NAtrac1::TAtrac1Data::TBlockSizeMod mode(false, false, true);` which gives LogCount [0,0,3]
    // as this matches the textual description of band shortness, and any deviation for LogCount[2]=2 would require a change to TBlockSizeMod
    // or a more direct way to set LogCount, which isn't standard.
    // If "LogCount = [0,0,2]" is a hard requirement for the test, this test harness won't achieve that specific LogCount[2] value.
    // For now, let's use the boolean description:
    NAtracDEnc::NAtrac1::TAtrac1Data::TBlockSizeMod current_mode(false, false, true); // LogCount will be [0,0,3]

    std::cout << "Using LogCount: [" << current_mode.LogCount[0] << ", "
              << current_mode.LogCount[1] << ", " << current_mode.LogCount[2] << "]" << std::endl;

    for (int current_frame = 0; current_frame < num_frames_to_process; ++current_frame) {
        SetDebugContext(channel, current_frame); // Removed NAtracDEnc::
        std::cout << "Processing frame: " << current_frame << " for channel: " << channel << std::endl;

        // Calculate offsets
        size_t offset_low_mid = static_cast<size_t>(current_frame) * 128;
        size_t offset_high = static_cast<size_t>(current_frame) * 256;

        // Check if enough data exists
        if (offset_low_mid + 128 > coeffs_ch0_low.size() ||
            offset_low_mid + 128 > coeffs_ch0_mid.size() ||
            offset_high + 256 > coeffs_ch0_high.size()) {
            std::cerr << "Error: Not enough coefficient data for frame " << current_frame << std::endl;
            return 1;
        }

        // Prepare Specs array for the current frame
        std::copy(coeffs_ch0_low.begin() + offset_low_mid, coeffs_ch0_low.begin() + offset_low_mid + 128, &Specs[0]);
        std::copy(coeffs_ch0_mid.begin() + offset_low_mid, coeffs_ch0_mid.begin() + offset_low_mid + 128, &Specs[128]);
        std::copy(coeffs_ch0_high.begin() + offset_high, coeffs_ch0_high.begin() + offset_high + 256, &Specs[256]);

        // Call IMDCT
        Mdc.IMdct(Specs, current_mode, PcmBufLow, PcmBufMid, PcmBufHi);
    }

    std::cout << "C++ IMDCT test harness finished successfully." << std::endl;
    return 0;
}
