#include "audio_features.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <fstream> // For std::ofstream
#include <iomanip> // For std::fixed, std::setprecision
#include <chrono>  // For timing

// Helper to print MFCCs (optional)
void printMfccs(const std::vector<float>& mfccs, int num_frames, int num_coeffs) {
    std::cout << "MFCCs (" << num_frames << " frames x " << num_coeffs << " coeffs):\n";
    for (int i = 0; i < num_frames; ++i) {
        std::cout << "Frame " << std::setw(3) << i << ": ";
        for (int j = 0; j < num_coeffs; ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << mfccs[i * num_coeffs + j] << " ";
        }
        std::cout << std::endl;
        if (i >= 10 && num_frames > 20) { // Print first 10 and last one if many frames
            if (i == 10 && num_frames > 11) std::cout << "...\n";
            if (i < num_frames -1 ) continue;
            else if (i == num_frames -1) {
                 std::cout << "Frame " << std::setw(3) << i << ": ";
                 for (int j = 0; j < num_coeffs; ++j) {
                    std::cout << std::fixed << std::setprecision(4) << std::setw(10) << mfccs[i * num_coeffs + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

// Helper to save MFCCs to a file
bool saveMfccsToFile(const std::string& filename, const std::vector<float>& mfccs, int num_frames, int num_coeffs) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing MFCCs: " << filename << std::endl;
        return false;
    }
    outfile << "# MFCC Data: " << num_frames << " frames, " << num_coeffs << " coefficients per frame.\n";
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < num_coeffs; ++j) {
            outfile << mfccs[i * num_coeffs + j] << (j == num_coeffs - 1 ? "" : "\t");
        }
        outfile << "\n";
    }
    outfile.close();
    std::cout << "MFCCs saved to " << filename << std::endl;
    return true;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_wav_file> [output_mfcc_file.txt]" << std::endl;
        return 1;
    }

    std::string input_wav_path = argv[1];
    std::string output_mfcc_path = "output_mfccs.txt";
    if (argc >= 3) {
        output_mfcc_path = argv[2];
    }

    std::vector<float> audio_samples;
    int sample_rate;
    short num_channels_read;
    short bits_per_sample_read;

    std::cout << "Reading WAV file: " << input_wav_path << std::endl;
    if (!readWavFile(input_wav_path, audio_samples, sample_rate, num_channels_read, bits_per_sample_read)) {
        std::cerr << "Failed to read WAV file." << std::endl;
        return 1;
    }

    std::cout << "WAV file read successfully:" << std::endl;
    std::cout << "  Sample Rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << num_channels_read << std::endl;
    std::cout << "  Bits per Sample: " << bits_per_sample_read << std::endl;
    std::cout << "  Total Samples: " << audio_samples.size() << std::endl;
    std::cout << "  Duration: " << static_cast<float>(audio_samples.size()) / sample_rate << " seconds" << std::endl;

    if (num_channels_read != 1) {
        std::cerr << "Error: This program currently only supports mono WAV files for MFCC extraction." << std::endl;
        // Note: readWavFile also has a check, but this is an additional one in main.
        // The readWavFile implementation provided processes only the first channel if stereo,
        // but the .clinerules and plan focus on mono.
        // return 1; // Strict check
    }


    // --- MFCC Parameters (based on our plan for 44.1kHz audio) ---
    // These should ideally match what GpuMfccExtractor expects or be passed to its constructor.
    // The GpuMfccExtractor constructor in .cuh uses these values.
    int n_window = static_cast<int>(WINDOW_DURATION_S * sample_rate); // e.g., 0.025 * 44100 = 1102.5 -> 1102
    int n_hop = static_cast<int>(HOP_DURATION_S * sample_rate);       // e.g., 0.010 * 44100 = 441
    
    // Calculate N_FFT as the next power of 2 >= n_window
    int n_fft = 1;
    while(n_fft < n_window) {
        n_fft *= 2;
    }
    if (n_window > 0 && n_fft == 1 && n_window > 1) n_fft = 2; // handles n_window = 1 edge case if n_fft starts at 1
    else if (n_window == 0) n_fft = 0; // Or handle error

    if (n_fft == 0 && n_window > 0) { // If n_window was large and caused overflow, or n_window was 0
         n_fft = 2048; // Fallback, or error
         std::cerr << "Warning: Could not determine N_FFT dynamically, defaulting to " << n_fft << std::endl;
    }
     if (n_window == 0) {
        std::cerr << "Error: n_window is 0, cannot proceed." << std::endl;
        return 1;
    }


    int n_mels = DEFAULT_N_MELS;
    int n_mfcc = DEFAULT_N_MFCC; // Number of coefficients to keep (e.g., 13, often excluding 0th)

    std::cout << "\nMFCC Parameters:" << std::endl;
    std::cout << "  N_Window: " << n_window << " samples (" << WINDOW_DURATION_S * 1000 << " ms)" << std::endl;
    std::cout << "  N_Hop: " << n_hop << " samples (" << HOP_DURATION_S * 1000 << " ms)" << std::endl;
    std::cout << "  N_FFT: " << n_fft << std::endl;
    std::cout << "  N_Mels: " << n_mels << std::endl;
    std::cout << "  N_MFCC (coeffs to keep): " << n_mfcc << std::endl;


    GpuMfccExtractor extractor(n_window, n_hop, n_fft, n_mels, n_mfcc, sample_rate,
                               true, PRE_EMPHASIS_ALPHA); // apply_preemphasis = true

    std::vector<float> mfcc_output;
    int num_frames_out = 0;
    int num_coeffs_out = 0;

    std::cout << "\nStarting MFCC computation on GPU..." << std::endl;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    bool success = extractor.computeMfccs(audio_samples, mfcc_output, num_frames_out, num_coeffs_out);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (success) {
        std::cout << "MFCC computation successful." << std::endl;
        std::cout << "  Time taken for GPU MFCC extraction: " << milliseconds << " ms" << std::endl;
        std::cout << "  Output: " << num_frames_out << " frames x " << num_coeffs_out << " coefficients." << std::endl;
        
        // printMfccs(mfcc_output, num_frames_out, num_coeffs_out); // Optional: print some to console
        saveMfccsToFile(output_mfcc_path, mfcc_output, num_frames_out, num_coeffs_out);

    } else {
        std::cerr << "MFCC computation failed." << std::endl;
        return 1;
    }

    return 0;
}
