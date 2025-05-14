#ifndef AUDIO_FEATURES_CUH
#define AUDIO_FEATURES_CUH

#include <vector>
#include <string>
#include <cmath> // For M_PI, log10, cos, sin, etc.
#include <cufft.h> // For cuFFT types

// --- Error Checking Macros (from .clinerules, to be defined if not globally available) ---
// Example:
// #define CHECK_CUDA_ERROR(err) \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     }

// #define CHECK_CUFFT_ERROR(err) \
//     if (err != CUFFT_SUCCESS) { \
//         fprintf(stderr, "cuFFT error in %s at line %d: %s\n", __FILE__, __LINE__, _cudaGetErrorEnum(err)); \
//         exit(EXIT_FAILURE); \
//     }
// // Helper function for cuFFT errors
// static const char *_cudaGetErrorEnum(cufftResult error) {
//     switch (error) {
//         case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
//         case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
//         // ... add other CUFFT_ERROR cases
//         default: return "<unknown cufft error>";
//     }
// }


// --- Constants for Audio Processing ---
// These will be derived from the WAV file and user parameters
// For now, placeholders based on our discussion for audio-sample-cuda-challenge.wav
constexpr int    TARGET_SAMPLE_RATE = 44100; // Hz
constexpr float  WINDOW_DURATION_S  = 0.025f; // seconds
constexpr float  HOP_DURATION_S     = 0.010f; // seconds

// Derived parameters (will be calculated based on actual sample rate)
// Example for 44100 Hz:
// N_WINDOW = 0.025 * 44100 = 1102.5 -> 1102 samples
// N_HOP    = 0.010 * 44100 = 441 samples
// N_FFT    = next_power_of_2(N_WINDOW) -> 2048 for N_WINDOW=1102

constexpr int    DEFAULT_N_MELS     = 40;
constexpr int    DEFAULT_N_MFCC     = 13; // Number of coefficients to keep (e.g., 1-13)
constexpr float  PRE_EMPHASIS_ALPHA = 0.97f;
constexpr float  MEL_FMIN           = 0.0f;   // Hz
// MEL_FMAX will be sample_rate / 2.0f

// --- WAV File Handling ---
struct WavHeader {
    char riff_header[4]; // "RIFF"
    int wav_size;
    char wave_header[4]; // "WAVE"
    char fmt_header[4];  // "fmt "
    int fmt_chunk_size;
    short audio_format;
    short num_channels;
    int sample_rate;
    int byte_rate;
    short block_align;
    short bits_per_sample;
    // Potentially "LIST" chunk or "data" chunk next
    // We will search for "data"
    char data_header[4]; // "data"
    int data_size;
};

/**
 * @brief Reads a WAV file.
 *
 * Supports mono, 16-bit PCM WAV files. Audio samples are normalized to [-1.0, 1.0].
 *
 * @param filename Path to the WAV file.
 * @param audio_samples Output vector to store normalized audio samples.
 * @param sample_rate Output to store the sample rate of the audio file.
 * @param num_channels Output to store the number of channels.
 * @return true if successful, false otherwise.
 */
bool readWavFile(const std::string& filename, std::vector<float>& audio_samples,
                 int& sample_rate, short& num_channels, short& bits_per_sample);


// --- Mel Filterbank ---
/**
 * @brief Creates a Mel filterbank matrix.
 *
 * @param n_mels Number of Mel filters.
 * @param n_fft FFT size.
 * @param sample_rate Sample rate of the audio.
 * @param f_min Minimum frequency for the filterbank.
 * @param f_max Maximum frequency for the filterbank.
 * @return A 2D vector representing the Mel filterbank (n_mels x (n_fft/2 + 1)).
 */
std::vector<std::vector<float>> createMelFilterbank(
    int n_mels, int n_fft, int sample_rate, float f_min, float f_max);

// --- DCT Matrix ---
/**
 * @brief Creates a DCT-II matrix.
 *
 * @param n_filters Number of input filters (e.g., N_MELS).
 * @param n_coeffs Number of output coefficients (e.g., N_MFCC or N_MELS if keeping all).
 * @return A 2D vector representing the DCT matrix (n_coeffs x n_filters).
 */
std::vector<std::vector<float>> createDctMatrix(int n_coeffs, int n_filters);


// --- Main MFCC Extraction Orchestration ---
class GpuMfccExtractor {
public:
    GpuMfccExtractor(int n_window, int n_hop, int n_fft, int n_mels, int n_mfcc,
                     int sample_rate, bool apply_preemphasis = true, float preemphasis_alpha = PRE_EMPHASIS_ALPHA);
    ~GpuMfccExtractor();

    /**
     * @brief Computes MFCCs from raw audio samples on the GPU.
     *
     * @param h_audio_samples Host vector of audio samples (normalized floats).
     * @param h_mfcc_output Host vector to store the computed MFCCs (flattened: num_frames x n_mfcc).
     * @return true if successful, false otherwise.
     */
    bool computeMfccs(const std::vector<float>& h_audio_samples, std::vector<float>& h_mfcc_output, int& out_num_frames, int& out_num_coeffs);

private:
    // Parameters
    int n_window_;
    int n_hop_;
    int n_fft_;
    int n_mels_;
    int n_mfcc_to_keep_; // Number of MFCC coefficients to actually store (e.g., 13)
    int sample_rate_;
    bool apply_preemphasis_;
    float preemphasis_alpha_;
    int num_fft_bins_; // n_fft_ / 2 + 1

    // Host data
    std::vector<std::vector<float>> h_mel_filterbank_;
    std::vector<std::vector<float>> h_dct_matrix_; // If using custom DCT

    // Device data pointers
    float* d_audio_samples_ = nullptr;
    float* d_preemphasized_samples_ = nullptr; // Optional
    float* d_framed_windowed_audio_ = nullptr; // Input to FFT (real)
    cufftComplex* d_fft_output_ = nullptr;     // Output of FFT (complex)
    float* d_power_spectrum_ = nullptr;        // Power spectrum (real)
    float* d_mel_energies_ = nullptr;          // Mel energies (real)
    float* d_log_mel_energies_ = nullptr;      // Log Mel energies (real)
    float* d_mfcc_temp_ = nullptr;             // Output of DCT (real, all N_MELS coeffs)
    float* d_final_mfccs_ = nullptr;           // Final MFCCs (num_frames * n_mfcc_to_keep_)

    float* d_mel_filterbank_flat_ = nullptr;
    float* d_dct_matrix_flat_ = nullptr; // If using custom DCT

    // cuFFT plans
    cufftHandle fft_plan_r2c_;
    cufftHandle dct_plan_r2r_; // For cuFFT DCT

    // Internal state
    bool initialized_ = false;
    size_t max_audio_samples_ = 0; // Max samples this instance can handle without realloc
    int max_frames_ = 0;           // Max frames this instance can handle

    void cleanup();
    bool allocateDeviceMemory(size_t num_samples, int num_frames);
    bool initializePlans(int num_frames);
};


// --- CUDA Kernels (Prototypes) ---
// These will be defined in audio_features.cu

// Kernel for pre-emphasis (optional)
__global__ void preemphasis_kernel(const float* raw_audio, float* preemphasized_audio, int num_samples, float alpha);

// Kernel for framing, windowing (e.g., Hamming)
__global__ void frame_and_window_kernel(const float* audio_in, float* frames_out,
                                        int num_total_samples, int n_window, int n_hop, int n_fft, int num_frames);

// Kernel for power spectrum calculation (magnitude squared of FFT output)
__global__ void power_spectrum_kernel(const cufftComplex* fft_output, float* power_spectrum,
                                      int num_frames, int num_fft_bins); // num_fft_bins = n_fft/2 + 1

// Kernel for Mel filterbank application (matrix multiplication)
__global__ void mel_filterbank_kernel(const float* power_spectrum, const float* mel_filterbank_matrix,
                                      float* mel_energies,
                                      int num_frames, int num_fft_bins, int n_mels);

// Kernel for taking log of Mel energies
__global__ void log_energies_kernel(const float* mel_energies, float* log_mel_energies,
                                    int num_frames, int n_mels, float epsilon = 1e-6f);

// Kernel for custom DCT (matrix multiplication) - if not using cuFFT's DCT
__global__ void dct_kernel(const float* log_mel_energies, const float* dct_matrix,
                           float* mfcc_output_full, // Output with n_mels coefficients
                           int num_frames, int n_mels, int n_dct_coeffs_in_matrix);

// Kernel to select final N_MFCC coefficients
__global__ void select_mfcc_coeffs_kernel(const float* mfcc_input_full, float* mfcc_output_final,
                                          int num_frames, int n_mels_input, int n_mfcc_to_keep, bool exclude_zeroth);


#endif // AUDIO_FEATURES_CUH
