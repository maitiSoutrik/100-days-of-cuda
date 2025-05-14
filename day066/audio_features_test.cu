#include "audio_features.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <string>
#include <cmath> // For fabs

// Test fixture for audio feature tests
class AudioFeaturesTest : public ::testing::Test {
protected:
    // You can define helper functions or member variables here
    // that are common to multiple tests.

    // Example: Helper to create a simple sine wave
    std::vector<float> createSineWave(int num_samples, float frequency, int sample_rate, float amplitude = 0.5f) {
        std::vector<float> samples(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            samples[i] = amplitude * sinf(2.0f * M_PI * frequency * i / sample_rate);
        }
        return samples;
    }
};

// Test case for WAV file reading
TEST_F(AudioFeaturesTest, WavReadTest_Basic) {
    // Create a dummy WAV file for testing or use a known small one.
    // For now, this test is more of a placeholder.
    // To properly test readWavFile, we'd need a known WAV file and check its properties.
    // Or, write a simple WAV file and then read it back.

    // Let's assume we have a way to get the path to a test audio file.
    // For now, we'll just check if the function can be called.
    // This test will likely fail or be problematic without a real input file setup for tests.
    std::vector<float> audio_samples;
    int sample_rate;
    short num_channels, bits_per_sample;

    // This path is relative to where the test executable might be run from (e.g., build/day066/)
    // The actual file `test_audio.wav` would need to be created and placed appropriately.
    // bool success = readWavFile("../test_data/simple_sine.wav", audio_samples, sample_rate, num_channels, bits_per_sample);
    // ASSERT_TRUE(success) << "Failed to read test WAV file.";
    // EXPECT_EQ(sample_rate, 44100); // Example assertion
    // EXPECT_EQ(num_channels, 1);
    // EXPECT_FALSE(audio_samples.empty());
    SUCCEED() << "WavReadTest_Basic placeholder. Needs a test WAV file.";
}

// Test case for Mel filterbank creation
TEST_F(AudioFeaturesTest, MelFilterbankCreation) {
    int sample_rate = 44100;
    int n_fft = 2048;
    int n_mels = 40;
    float f_min = 0.0f;
    float f_max = sample_rate / 2.0f;

    std::vector<std::vector<float>> mel_filters = createMelFilterbank(n_mels, n_fft, sample_rate, f_min, f_max);

    ASSERT_EQ(mel_filters.size(), n_mels);
    if (n_mels > 0) {
        ASSERT_EQ(mel_filters[0].size(), n_fft / 2 + 1);
    }
    // Add more specific checks:
    // - Sum of each filter (should be > 0)
    // - Triangular shape (values increase then decrease)
    // - Correct frequency coverage
    SUCCEED() << "MelFilterbankCreation basic checks passed.";
}

// Test case for DCT matrix creation
TEST_F(AudioFeaturesTest, DctMatrixCreation) {
    int n_coeffs = 13;
    int n_filters = 40; // e.g. n_mels

    std::vector<std::vector<float>> dct_matrix = createDctMatrix(n_coeffs, n_filters);
    ASSERT_EQ(dct_matrix.size(), n_coeffs);
    if (n_coeffs > 0) {
        ASSERT_EQ(dct_matrix[0].size(), n_filters);
    }
    // Add more specific checks if possible (e.g., orthogonality for a square matrix, known values)
    SUCCEED() << "DctMatrixCreation basic checks passed.";
}


// Test case for the GpuMfccExtractor (end-to-end, very basic)
TEST_F(AudioFeaturesTest, GpuMfccExtractor_SimpleRun) {
    int sample_rate = 44100;
    int n_window = 1102;
    int n_hop = 441;
    int n_fft = 2048;
    int n_mels = 40;
    int n_mfcc = 13;

    // Create a simple test signal (e.g., 1 second of sine wave)
    std::vector<float> test_audio = createSineWave(sample_rate * 1, 440.0f, sample_rate); // 1 sec, 440Hz tone

    GpuMfccExtractor extractor(n_window, n_hop, n_fft, n_mels, n_mfcc, sample_rate, false); // preemphasis=false

    std::vector<float> mfcc_output;
    int num_frames_out = 0;
    int num_coeffs_out = 0;

    bool success = extractor.computeMfccs(test_audio, mfcc_output, num_frames_out, num_coeffs_out);
    ASSERT_TRUE(success) << "GpuMfccExtractor::computeMfccs failed.";

    // Expected number of frames: 1 + (total_samples - n_window) / n_hop
    int expected_num_frames = 0;
    if (test_audio.size() >= static_cast<size_t>(n_window)) {
     expected_num_frames = 1 + (test_audio.size() - n_window) / n_hop;
    }


    ASSERT_EQ(num_frames_out, expected_num_frames);
    ASSERT_EQ(num_coeffs_out, n_mfcc);
    ASSERT_EQ(mfcc_output.size(), static_cast<size_t>(expected_num_frames * n_mfcc));

    // Note: The actual values will be mostly zero due to unimplemented kernels.
    // This test primarily checks if the pipeline runs without crashing and produces output of correct dimensions.
    // For actual value checking, reference MFCCs (e.g., from Librosa) would be needed.
    std::cout << "GpuMfccExtractor_SimpleRun produced " << num_frames_out << " frames and " << num_coeffs_out << " coeffs." << std::endl;
    if (num_frames_out > 0 && num_coeffs_out > 0) {
        std::cout << "First MFCC value: " << mfcc_output[0] << std::endl;
    }
}

// It would be beneficial to have a test that compares output with a known library like Librosa.
// This requires:
// 1. A reference audio file.
// 2. Python script to generate MFCCs using Librosa with exact same parameters.
// 3. Loading those reference MFCCs into the C++ test and comparing.
// This is outside the scope of initial setup but important for validation.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
