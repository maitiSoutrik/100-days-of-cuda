#include "audio_features.cuh"
#include <iostream>
#include <fstream>
#include <stdexcept> // For std::runtime_error
#include <algorithm> // For std::min, std::max
#include <cstring>   // For strncmp, memcpy

// Define M_PI if not already defined (e.g., on Windows with MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Error Checking Macros (Implementations) ---
// These should align with your project's common error handling, e.g., from a utility header.
// For now, basic implementations.
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* exit(EXIT_FAILURE); */ \
        /* Consider returning false or throwing an exception in class methods */ \
    }

// Helper function for cuFFT errors (as defined in .cuh comments)
static const char *_cudaGetErrorEnum(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "<unknown cufft error>";
    }
}

#define CHECK_CUFFT_ERROR(err) \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error in %s at line %d: %s\n", __FILE__, __LINE__, _cudaGetErrorEnum(err)); \
        /* exit(EXIT_FAILURE); */ \
        /* Consider returning false or throwing an exception in class methods */ \
    }


// --- WAV File Handling Implementation ---
bool readWavFile(const std::string& filename, std::vector<float>& audio_samples,
                 int& sample_rate_out, short& num_channels_out, short& bits_per_sample_out) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open WAV file: " << filename << std::endl;
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader) - 8); // Read up to bits_per_sample

    // Check RIFF and WAVE headers
    if (strncmp(header.riff_header, "RIFF", 4) != 0 || strncmp(header.wave_header, "WAVE", 4) != 0) {
        std::cerr << "Error: Invalid WAV file format (RIFF/WAVE headers)." << std::endl;
        return false;
    }
    // Check fmt header
    if (strncmp(header.fmt_header, "fmt ", 4) != 0) {
        std::cerr << "Error: Invalid WAV file format (fmt header)." << std::endl;
        return false;
    }

    // Read the rest of the fmt chunk if it's larger than minimal (16 bytes)
    if (header.fmt_chunk_size > 16) {
        file.seekg(header.fmt_chunk_size - 16, std::ios::cur);
    }
    
    // Search for "data" chunk
    char chunk_header[4];
    int chunk_size;
    bool data_chunk_found = false;
    while(file.read(chunk_header, 4)) {
        file.read(reinterpret_cast<char*>(&chunk_size), sizeof(int));
        if (strncmp(chunk_header, "data", 4) == 0) {
            strncpy(header.data_header, chunk_header, 4);
            header.data_size = chunk_size;
            data_chunk_found = true;
            break;
        }
        file.seekg(chunk_size, std::ios::cur); // Skip this chunk
        if (file.eof() || file.fail()) {
             std::cerr << "Error: Reached EOF or stream error before finding 'data' chunk." << std::endl;
             return false;
        }
    }

    if (!data_chunk_found) {
        std::cerr << "Error: 'data' chunk not found in WAV file." << std::endl;
        return false;
    }
    
    // Validate audio format (PCM), channels (mono), bits per sample (16)
    if (header.audio_format != 1) {
        std::cerr << "Error: Only PCM audio_format=1 is supported. Got: " << header.audio_format << std::endl;
        return false;
    }
    // For this project, we'll simplify and enforce mono, 16-bit as per ffprobe output.
    // Can be extended later.
    if (header.num_channels != 1) {
        std::cerr << "Warning: Expected mono audio (1 channel), got " << header.num_channels
                  << ". Processing as mono by taking the first channel if stereo, or failing." << std::endl;
        // For now, let's strictly require mono as per the plan.
        // If stereo, one could average or take one channel.
        // return false; // Or handle stereo to mono conversion
    }
     if (header.bits_per_sample != 16) {
        std::cerr << "Error: Expected 16 bits_per_sample, got " << header.bits_per_sample << std::endl;
        return false;
    }

    sample_rate_out = header.sample_rate;
    num_channels_out = header.num_channels; // Store actual, even if we only process one
    bits_per_sample_out = header.bits_per_sample;

    int num_samples = header.data_size / (header.num_channels * (header.bits_per_sample / 8));
    audio_samples.resize(num_samples);

    std::vector<short> sample_buffer(num_samples * header.num_channels);
    file.read(reinterpret_cast<char*>(sample_buffer.data()), header.data_size);

    if (file.gcount() != header.data_size) {
        std::cerr << "Error: Could not read all audio data. Expected " << header.data_size << " bytes, got " << file.gcount() << std::endl;
        return false;
    }

    // Normalize and convert to float (taking first channel if stereo for simplicity, though we check for mono)
    for (int i = 0; i < num_samples; ++i) {
        // If truly stereo and we decided to process, this would be:
        // audio_samples[i] = static_cast<float>(sample_buffer[i * header.num_channels]) / 32768.0f;
        // For mono:
        audio_samples[i] = static_cast<float>(sample_buffer[i]) / 32768.0f; // Max value for 16-bit signed int
    }

    return true;
}

// --- Mel Filterbank Implementation ---
// Helper to convert Hz to Mel
static inline float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}
// Helper to convert Mel to Hz
static inline float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> createMelFilterbank(
    int n_mels, int n_fft, int sample_rate, float f_min, float f_max_hz) {

    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));

    float mel_low = hzToMel(f_min);
    float mel_high = hzToMel(f_max_hz);

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }

    std::vector<float> hz_points(n_mels + 2);
    std::vector<int> fft_bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = melToHz(mel_points[i]);
        fft_bin_points[i] = static_cast<int>(floor((n_fft + 1) * hz_points[i] / sample_rate));
    }

    for (int m = 0; m < n_mels; ++m) {
        int f_m_minus_1 = fft_bin_points[m];     // left
        int f_m = fft_bin_points[m + 1];         // center
        int f_m_plus_1 = fft_bin_points[m + 2];  // right

        for (int k = f_m_minus_1; k < f_m; ++k) {
            if (k < (n_fft / 2 + 1)) { // Ensure k is within bounds
                 filterbank[m][k] = (k - f_m_minus_1) / static_cast<float>(f_m - f_m_minus_1);
            }
        }
        for (int k = f_m; k < f_m_plus_1; ++k) {
             if (k < (n_fft / 2 + 1)) { // Ensure k is within bounds
                filterbank[m][k] = (f_m_plus_1 - k) / static_cast<float>(f_m_plus_1 - f_m);
            }
        }
    }
    return filterbank;
}

// --- DCT Matrix Implementation ---
std::vector<std::vector<float>> createDctMatrix(int n_coeffs, int n_filters) {
    std::vector<std::vector<float>> dct_matrix(n_coeffs, std::vector<float>(n_filters));
    float scale = sqrtf(2.0f / n_filters);
    float scale0 = sqrtf(1.0f / n_filters); // For k=0, but typically we use type II where k starts from 0

    for (int k = 0; k < n_coeffs; ++k) {
        for (int n = 0; n < n_filters; ++n) {
            float val = cosf(M_PI / n_filters * (n + 0.5f) * k);
            dct_matrix[k][n] = (k == 0 ? scale0 : scale) * val;
            // Librosa uses an orthonormal DCT, which might have slightly different scaling.
            // For Type-II DCT:
            // dct_matrix[k][n] = cosf(M_PI * k * (2.0f * n + 1.0f) / (2.0f * n_filters));
            // if (k == 0) dct_matrix[k][n] *= (1.0f / sqrtf(n_filters));
            // else dct_matrix[k][n] *= sqrtf(2.0f / n_filters);
            // The formula used above (cos(PI/N * (n+0.5) * k)) is common for DCT-II.
            // Let's stick to a common DCT-II formulation.
            // The scaling factor can be applied later or incorporated.
            // For MFCC, often the first coefficient (k=0) is scaled by 1/sqrt(N) and others by sqrt(2/N).
            // Let's use the common definition: C_k = sum_{n=0}^{N-1} x_n cos(pi*k*(2n+1)/(2N))
            // and then apply normalization if needed.
            // The one from Wikipedia for DCT-II:
            dct_matrix[k][n] = cosf(M_PI * (n + 0.5f) * k / n_filters);
        }
    }
     // Orthonormal scaling (often applied)
    for (int n = 0; n < n_filters; ++n) {
        dct_matrix[0][n] *= (1.0f / sqrtf(static_cast<float>(n_filters)));
    }
    for (int k = 1; k < n_coeffs; ++k) {
        for (int n = 0; n < n_filters; ++n) {
            dct_matrix[k][n] *= sqrtf(2.0f / n_filters);
        }
    }
    return dct_matrix;
}


// --- GpuMfccExtractor Class Implementation ---
GpuMfccExtractor::GpuMfccExtractor(int n_window, int n_hop, int n_fft, int n_mels, int n_mfcc,
                                   int sample_rate, bool apply_preemphasis, float preemphasis_alpha)
    : n_window_(n_window), n_hop_(n_hop), n_fft_(n_fft), n_mels_(n_mels), n_mfcc_to_keep_(n_mfcc),
      sample_rate_(sample_rate), apply_preemphasis_(apply_preemphasis), preemphasis_alpha_(preemphasis_alpha),
      fft_plan_r2c_(0), dct_plan_r2r_(0), initialized_(false) {

    num_fft_bins_ = n_fft_ / 2 + 1;

    // Generate Mel filterbank on host
    float f_max = static_cast<float>(sample_rate_) / 2.0f;
    h_mel_filterbank_ = createMelFilterbank(n_mels_, n_fft_, sample_rate_, MEL_FMIN, f_max);

    // Generate DCT matrix on host (if using custom DCT, or for reference)
    // For cuFFT DCT, this matrix isn't directly used by cuFFT but good for understanding/testing
    h_dct_matrix_ = createDctMatrix(n_mels_, n_mels_); // DCT from N_MELS to N_MELS, then we pick N_MFCC

    // Further initialization (memory allocation, plan creation) will be done in computeMfccs
    // or a separate init method if we want to pre-allocate for a max size.
    // For now, let's defer to first call of computeMfccs or a dedicated init.
}

GpuMfccExtractor::~GpuMfccExtractor() {
    cleanup();
}

void GpuMfccExtractor::cleanup() {
    if (d_audio_samples_) cudaFree(d_audio_samples_);
    if (d_preemphasized_samples_) cudaFree(d_preemphasized_samples_);
    if (d_framed_windowed_audio_) cudaFree(d_framed_windowed_audio_);
    if (d_fft_output_) cudaFree(d_fft_output_);
    if (d_power_spectrum_) cudaFree(d_power_spectrum_);
    if (d_mel_energies_) cudaFree(d_mel_energies_);
    if (d_log_mel_energies_) cudaFree(d_log_mel_energies_);
    if (d_mfcc_temp_) cudaFree(d_mfcc_temp_);
    if (d_final_mfccs_) cudaFree(d_final_mfccs_);
    if (d_mel_filterbank_flat_) cudaFree(d_mel_filterbank_flat_);
    if (d_dct_matrix_flat_) cudaFree(d_dct_matrix_flat_);

    d_audio_samples_ = d_preemphasized_samples_ = d_framed_windowed_audio_ = nullptr;
    d_fft_output_ = nullptr;
    d_power_spectrum_ = d_mel_energies_ = d_log_mel_energies_ = d_mfcc_temp_ = d_final_mfccs_ = nullptr;
    d_mel_filterbank_flat_ = d_dct_matrix_flat_ = nullptr;

    if (fft_plan_r2c_) cufftDestroy(fft_plan_r2c_);
    if (dct_plan_r2r_) cufftDestroy(dct_plan_r2r_);
    fft_plan_r2c_ = 0;
    dct_plan_r2r_ = 0;
    initialized_ = false;
}


bool GpuMfccExtractor::allocateDeviceMemory(size_t num_audio_samples, int num_frames) {
    // Basic check: if already allocated and sufficient, skip.
    // This is a simplified check; a more robust one would check each buffer.
    if (initialized_ && num_audio_samples <= max_audio_samples_ && num_frames <= max_frames_) {
       // return true; // Assuming memory is still valid. For safety, could re-allocate or add more checks.
    }
    
    cleanup(); // Clear previous allocations and plans before reallocating

    max_audio_samples_ = num_audio_samples;
    max_frames_ = num_frames;

    cudaError_t err;

    // 1. Raw/Preemphasized audio samples
    err = cudaMalloc(&d_audio_samples_, num_audio_samples * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    if (apply_preemphasis_) {
        err = cudaMalloc(&d_preemphasized_samples_, num_audio_samples * sizeof(float));
        CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    }

    // 2. Framed and windowed audio (input to FFT)
    err = cudaMalloc(&d_framed_windowed_audio_, num_frames * n_fft_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // 3. FFT output (complex)
    err = cudaMalloc(&d_fft_output_, num_frames * num_fft_bins_ * sizeof(cufftComplex));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // 4. Power spectrum
    err = cudaMalloc(&d_power_spectrum_, num_frames * num_fft_bins_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // 5. Mel energies
    err = cudaMalloc(&d_mel_energies_, num_frames * n_mels_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // 6. Log Mel energies
    err = cudaMalloc(&d_log_mel_energies_, num_frames * n_mels_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    
    // 7. MFCC temporary (output of DCT, N_MELS coefficients)
    err = cudaMalloc(&d_mfcc_temp_, num_frames * n_mels_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // 8. Final MFCCs (N_MFCC_TO_KEEP coefficients)
    err = cudaMalloc(&d_final_mfccs_, num_frames * n_mfcc_to_keep_ * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // --- Allocate and copy matrices to device ---
    // Mel filterbank (flattened)
    std::vector<float> mel_flat(n_mels_ * num_fft_bins_);
    for(int i=0; i < n_mels_; ++i) {
        memcpy(mel_flat.data() + i * num_fft_bins_, h_mel_filterbank_[i].data(), num_fft_bins_ * sizeof(float));
    }
    err = cudaMalloc(&d_mel_filterbank_flat_, mel_flat.size() * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    err = cudaMemcpy(d_mel_filterbank_flat_, mel_flat.data(), mel_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;

    // DCT matrix (flattened, if using custom DCT kernel)
    // For cuFFT DCT, this is not strictly needed on device unless for custom kernel.
    // Let's assume we might use a custom DCT kernel or want it for reference.
    // DCT matrix is n_mels_ x n_mels_ (output coeffs x input filters)
    std::vector<float> dct_flat(n_mels_ * n_mels_);
    for(int i=0; i < n_mels_; ++i) { // h_dct_matrix_ is n_mels_ (coeffs) x n_mels_ (filters)
        memcpy(dct_flat.data() + i * n_mels_, h_dct_matrix_[i].data(), n_mels_ * sizeof(float));
    }
    err = cudaMalloc(&d_dct_matrix_flat_, dct_flat.size() * sizeof(float));
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    err = cudaMemcpy(d_dct_matrix_flat_, dct_flat.data(), dct_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err); if (err != cudaSuccess) return false;
    
    return true;
}

bool GpuMfccExtractor::initializePlans(int num_frames) {
    cufftResult cufft_err;

    // R2C FFT plan
    // int n_fft_single = n_fft_; // Size of 1D FFT
    // cufft_err = cufftPlan1d(&fft_plan_r2c_, n_fft_single, CUFFT_R2C, num_frames);
    // For batched FFTs:
    int rank = 1; // 1D FFTs
    int n[] = {n_fft_}; // Size of each FFT
    int idist = n_fft_; // Distance between consecutive input datasets
    int odist = num_fft_bins_; // Distance between consecutive output datasets
    int istride = 1, ostride = 1; // Stride within each dimension (for 1D, this is 1)
    
    cufft_err = cufftPlanMany(&fft_plan_r2c_, rank, n,
                              nullptr, istride, idist, // input format (embed=nullptr for basic)
                              nullptr, ostride, odist, // output format
                              CUFFT_R2C, num_frames);
    CHECK_CUFFT_ERROR(cufft_err);
    if (cufft_err != CUFFT_SUCCESS) return false;

    // R2R DCT plan (using cuFFT for DCT-II)
    // Input to DCT is Log Mel Energies (num_frames x n_mels_)
    // Output is MFCCs (num_frames x n_mels_)
    int n_dct[] = {n_mels_};
    int idist_dct = n_mels_;
    int odist_dct = n_mels_;

    cufft_err = cufftPlanMany(&dct_plan_r2r_, rank, n_dct,
                              nullptr, istride, idist_dct,
                              nullptr, ostride, odist_dct,
                              CUFFT_R2R, num_frames); // CUFFT_R2R with appropriate kind for DCT-II
                                                     // For DCT-II, kind is not explicitly set here,
                                                     // but cufftExecR2R with real input/output implies it.
                                                     // More specific DCT types might need cufftXtMakePlan(),
                                                     // but basic R2R often suffices for DCT-II like transforms.
                                                     // Let's assume this is sufficient for now.
                                                     // Libs like TorchAudio use cuFFT's R2R for DCT.
    CHECK_CUFFT_ERROR(cufft_err);
    if (cufft_err != CUFFT_SUCCESS) return false;
    
    return true;
}


bool GpuMfccExtractor::computeMfccs(const std::vector<float>& h_audio_samples, std::vector<float>& h_mfcc_output, int& out_num_frames, int& out_num_coeffs) {
    if (h_audio_samples.empty()) {
        std::cerr << "Error: Input audio samples vector is empty." << std::endl;
        return false;
    }

    size_t num_total_samples = h_audio_samples.size();
    if (num_total_samples < static_cast<size_t>(n_window_)) {
        std::cerr << "Error: Not enough audio samples (" << num_total_samples << ") for a single window (" << n_window_ << ")." << std::endl;
        return false;
    }

    // Calculate number of frames
    // num_frames = 1 + (num_total_samples - n_window_) / n_hop_ if num_total_samples >= n_window_ else 0
    int num_frames = 0;
    if (num_total_samples >= static_cast<size_t>(n_window_)) {
        num_frames = 1 + static_cast<int>((num_total_samples - n_window_) / n_hop_);
    }
     if (num_frames <= 0) {
        std::cerr << "Error: Calculated number of frames is zero or negative." << std::endl;
        return false;
    }
    out_num_frames = num_frames;
    out_num_coeffs = n_mfcc_to_keep_;


    // Allocate memory and initialize plans if not done or if size changed significantly
    // This simplified logic reallocates/reinitializes if num_frames is different than max_frames_
    // A more robust approach would check if current allocations are sufficient.
    if (!initialized_ || num_frames > max_frames_ || num_total_samples > max_audio_samples_) {
        if (!allocateDeviceMemory(num_total_samples, num_frames)) {
            std::cerr << "Error: Failed to allocate device memory." << std::endl;
            return false;
        }
        if (!initializePlans(num_frames)) {
            std::cerr << "Error: Failed to initialize cuFFT plans." << std::endl;
            return false;
        }
        initialized_ = true;
    }


    cudaError_t cuda_err;
    cufftResult cufft_err;

    // --- 1. Copy audio to device ---
    cuda_err = cudaMemcpy(d_audio_samples_, h_audio_samples.data(), num_total_samples * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;

    const float* current_audio_ptr = d_audio_samples_;

    // --- 2. Pre-emphasis (Optional) ---
    if (apply_preemphasis_) {
        dim3 block_pre(256);
        dim3 grid_pre((num_total_samples + block_pre.x - 1) / block_pre.x);
        preemphasis_kernel<<<grid_pre, block_pre>>>(d_audio_samples_, d_preemphasized_samples_, num_total_samples, preemphasis_alpha_);
        cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;
        current_audio_ptr = d_preemphasized_samples_;
    }

    // --- 3. Framing and Windowing ---
    // Each block processes one frame. Threads per block should be enough to cover n_window or n_fft efficiently.
    // Let's use a block size that's a power of 2, e.g., 128 or 256.
    // If n_window is large (e.g. > 1024), a larger block size might be better. For n_window=1102, 256 or 512 could work.
    // Max threads per block is 1024. Max shared memory per block is typically 48KB.
    // n_window_ * sizeof(float) = 1102 * 4 = 4408 bytes (around 4.3KB), well within limits.
    dim3 threads_per_block_fw(256); // Threads in a block for frame_and_window_kernel
    dim3 num_blocks_fw(num_frames);    // One block per frame
    size_t shared_mem_size = n_window_ * sizeof(float);

    frame_and_window_kernel<<<num_blocks_fw, threads_per_block_fw, shared_mem_size>>>(
        current_audio_ptr, d_framed_windowed_audio_,
        num_total_samples, n_window_, n_hop_, n_fft_, num_frames
    );
    cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;


    // --- 4. FFT ---
    cufft_err = cufftExecR2C(fft_plan_r2c_, reinterpret_cast<cufftReal*>(d_framed_windowed_audio_), d_fft_output_);
    CHECK_CUFFT_ERROR(cufft_err); if(cufft_err != CUFFT_SUCCESS) return false;

    // --- 5. Power Spectrum ---
    dim3 block_ps((num_fft_bins_ + 31) / 32 * 32); // Ensure blockDim.x is multiple of 32, up to 1024
    if (block_ps.x == 0) block_ps.x = 32; // Handle num_fft_bins_ < 32
    if (block_ps.x > 1024) block_ps.x = 1024;
    dim3 grid_ps(num_frames); // Each block for a frame, threads compute bins for that frame
                               // Or, a 1D grid over all elements:
                               // dim3 grid_ps_flat((num_frames * num_fft_bins_ + 255) / 256);
                               // dim3 block_ps_flat(256);
                               // power_spectrum_kernel<<<grid_ps_flat, block_ps_flat>>>(d_fft_output_, d_power_spectrum_, num_frames, num_fft_bins_);
    // Let's use one block per frame, threads iterate over bins if num_fft_bins > blockDim.x
    // Or better, a 2D grid/block or a flattened 1D grid.
    // For simplicity with current kernel:
    int total_power_spectrum_elements = num_frames * num_fft_bins_;
    dim3 block_ps_flat(256);
    dim3 grid_ps_flat((total_power_spectrum_elements + block_ps_flat.x - 1) / block_ps_flat.x);
    power_spectrum_kernel<<<grid_ps_flat, block_ps_flat>>>(d_fft_output_, d_power_spectrum_, num_frames, num_fft_bins_);
    cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;


    // --- 6. Mel Filterbank Application ---
    // Each block for a frame, each thread for a Mel filter. BlockDim.x = n_mels_
    dim3 block_mel(n_mels_); // Threads per block = number of mel filters
    dim3 grid_mel(num_frames);  // Number of blocks = number of frames
    if (n_mels_ > 0 && n_mels_ <=1024) { // Check if n_mels_ is a valid block dimension
        mel_filterbank_kernel<<<grid_mel, block_mel>>>(
            d_power_spectrum_, d_mel_filterbank_flat_, d_mel_energies_,
            num_frames, num_fft_bins_, n_mels_
        );
        cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;
    } else {
         std::cerr << "Warning: Mel filterbank kernel not launched due to invalid n_mels for blockDim: " << n_mels_ << std::endl;
         cudaMemset(d_mel_energies_, 0, num_frames * n_mels_ * sizeof(float)); // Keep pipeline going
    }


    // --- 7. Log Mel Energies ---
    int total_log_mel_elements = num_frames * n_mels_;
    dim3 block_log_mel(256);
    dim3 grid_log_mel((total_log_mel_elements + block_log_mel.x -1) / block_log_mel.x);
    if (n_mels_ > 0) {
        log_energies_kernel<<<grid_log_mel, block_log_mel>>>(d_mel_energies_, d_log_mel_energies_, num_frames, n_mels_);
        cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;
    } else {
        cudaMemset(d_log_mel_energies_, 0, num_frames * n_mels_ * sizeof(float));
    }


    // --- 8. DCT ---
    // Input: d_log_mel_energies_ (num_frames x n_mels_)
    // Output: d_mfcc_temp_      (num_frames x n_mels_)
    cufft_err = cufftExecR2R(dct_plan_r2r_, reinterpret_cast<cufftReal*>(d_log_mel_energies_), reinterpret_cast<cufftReal*>(d_mfcc_temp_));
    CHECK_CUFFT_ERROR(cufft_err); if(cufft_err != CUFFT_SUCCESS) return false;
    // If using custom DCT kernel:
    // dct_kernel<<<(num_frames * n_mels_ + 255) / 256, 256>>>(d_log_mel_energies_, d_dct_matrix_flat_, d_mfcc_temp_, num_frames, n_mels_, n_mels_);
    // cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;


    // --- 9. Select N_MFCC coefficients ---
    if (n_mfcc_to_keep_ > 0 && n_mfcc_to_keep_ <= n_mels_) {
        // Each block for a frame, each thread for an output MFCC coefficient
        dim3 block_select(n_mfcc_to_keep_);
        dim3 grid_select(num_frames);
        if (n_mfcc_to_keep_ <= 1024) { // Check blockDim validity
             select_mfcc_coeffs_kernel<<<grid_select, block_select>>>(
                d_mfcc_temp_, d_final_mfccs_,
                num_frames, n_mels_, n_mfcc_to_keep_, true /* exclude_zeroth=true, common for MFCCs */
            );
            cuda_err = cudaGetLastError(); CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;
        } else {
            std::cerr << "Warning: Select MFCC coeffs kernel not launched due to invalid n_mfcc_to_keep for blockDim: " << n_mfcc_to_keep_ << std::endl;
            cudaMemset(d_final_mfccs_, 0, num_frames * n_mfcc_to_keep_ * sizeof(float));
        }
    } else if (n_mfcc_to_keep_ > n_mels_) {
        std::cerr << "Error: n_mfcc_to_keep (" << n_mfcc_to_keep_ << ") > n_mels (" << n_mels_ << ")" << std::endl;
        return false;
    } else { // n_mfcc_to_keep_ is 0
         std::cerr << "Warning: n_mfcc_to_keep is 0. No MFCCs will be output." << std::endl;
         // h_mfcc_output will be empty, which is handled later.
    }


    // --- 10. Copy final MFCCs to host ---
    h_mfcc_output.resize(num_frames * n_mfcc_to_keep_);
    if (!h_mfcc_output.empty()) { // only copy if size > 0
        cuda_err = cudaMemcpy(h_mfcc_output.data(), d_final_mfccs_, num_frames * n_mfcc_to_keep_ * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(cuda_err); if(cuda_err != cudaSuccess) return false;
    }
    
    return true;
}


// --- CUDA Kernel Implementations (Stubs for now, to be filled in) ---

__global__ void preemphasis_kernel(const float* raw_audio, float* preemphasized_audio, int num_samples, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        if (idx == 0) {
            preemphasized_audio[idx] = raw_audio[idx]; // Or raw_audio[idx] * (1-alpha) or some other convention
        } else {
            preemphasized_audio[idx] = raw_audio[idx] - alpha * raw_audio[idx - 1];
        }
    }
}

__global__ void frame_and_window_kernel(const float* audio_in, float* frames_out,
                                        int num_total_samples, int n_window, int n_hop, int n_fft, int num_frames) {
    // Each block processes one frame
    int frame_idx = blockIdx.x;
    if (frame_idx >= num_frames) return;

    // Shared memory for one window of audio samples
    extern __shared__ float s_window_data[]; // Size n_window

    // Calculate start sample for this frame
    int frame_start_sample = frame_idx * n_hop;

    // Load samples into shared memory and apply Hamming window
    // Threads in block cooperate to load
    for (int i = threadIdx.x; i < n_window; i += blockDim.x) {
        if (frame_start_sample + i < num_total_samples) {
            s_window_data[i] = audio_in[frame_start_sample + i];
        } else {
            s_window_data[i] = 0.0f; // Zero padding if frame goes beyond audio length
        }
        // Apply Hamming window: w[n] = 0.54 - 0.46 * cos(2 * PI * n / (N_WINDOW - 1))
        s_window_data[i] *= (0.54f - 0.46f * cosf(2.0f * M_PI * i / (n_window - 1.0f)));
    }
    __syncthreads();

    // Write windowed samples to global memory, zero-padding up to n_fft
    // Threads in block cooperate to write
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        if (i < n_window) {
            frames_out[frame_idx * n_fft + i] = s_window_data[i];
        } else {
            frames_out[frame_idx * n_fft + i] = 0.0f; // Zero padding
        }
    }
}


__global__ void power_spectrum_kernel(const cufftComplex* fft_output, float* power_spectrum,
                                      int num_frames, int num_fft_bins) {
    // num_fft_bins = n_fft / 2 + 1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bins_to_compute = num_frames * num_fft_bins;

    if (idx < total_bins_to_compute) {
        cufftComplex val = fft_output[idx];
        power_spectrum[idx] = val.x * val.x + val.y * val.y; // Magnitude squared
    }
}

__global__ void mel_filterbank_kernel(const float* power_spectrum, const float* mel_filterbank_matrix_flat,
                                      float* mel_energies,
                                      int num_frames, int num_fft_bins, int n_mels) {
    // Each block could compute Mel energies for one frame.
    // Each thread in a block could compute one Mel energy for that frame.
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x; // Thread computes m-th mel energy for frame_idx

    if (frame_idx >= num_frames || mel_idx >= n_mels) return;

    float current_mel_energy = 0.0f;
    // power_spectrum for this frame starts at frame_idx * num_fft_bins
    // mel_filterbank_matrix_flat for this mel filter starts at mel_idx * num_fft_bins
    const float* frame_power_spectrum = power_spectrum + frame_idx * num_fft_bins;
    const float* filter_coeffs = mel_filterbank_matrix_flat + mel_idx * num_fft_bins;

    for (int k = 0; k < num_fft_bins; ++k) {
        current_mel_energy += frame_power_spectrum[k] * filter_coeffs[k];
    }
    mel_energies[frame_idx * n_mels + mel_idx] = current_mel_energy;
}


__global__ void log_energies_kernel(const float* mel_energies, float* log_mel_energies,
                                    int num_frames, int n_mels, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_frames * n_mels;

    if (idx < total_elements) {
        float energy = mel_energies[idx];
        log_mel_energies[idx] = logf(energy + epsilon);
    }
}

// Custom DCT kernel (if not using cuFFT's R2R for DCT)
__global__ void dct_kernel(const float* log_mel_energies, const float* dct_matrix_flat,
                           float* mfcc_output_full,
                           int num_frames, int n_mels, int n_dct_coeffs_in_matrix) {
    // n_dct_coeffs_in_matrix is likely n_mels if matrix is N_MELS x N_MELS
    // Each block for a frame, each thread for an MFCC coefficient (output)
    int frame_idx = blockIdx.x;
    int coeff_k_idx = threadIdx.x; // k-th MFCC coefficient

    if (frame_idx >= num_frames || coeff_k_idx >= n_dct_coeffs_in_matrix) return;

    float current_mfcc_val = 0.0f;
    const float* frame_log_mel_energies = log_mel_energies + frame_idx * n_mels;
    // dct_matrix_flat is coeff_k_idx (row) major: coeff_k_idx * n_mels (cols)
    const float* dct_row_k = dct_matrix_flat + coeff_k_idx * n_mels;

    for (int n = 0; n < n_mels; ++n) { // Sum over n_mels (input filters)
        current_mfcc_val += frame_log_mel_energies[n] * dct_row_k[n];
    }
    mfcc_output_full[frame_idx * n_dct_coeffs_in_matrix + coeff_k_idx] = current_mfcc_val;
}


__global__ void select_mfcc_coeffs_kernel(const float* mfcc_input_full, float* mfcc_output_final,
                                          int num_frames, int n_mels_input, int n_mfcc_to_keep, bool exclude_zeroth) {
    // Each block for a frame. Each thread for one of the n_mfcc_to_keep coefficients.
    int frame_idx = blockIdx.x;
    int k_out = threadIdx.x; // k_out is the index for the output (0 to n_mfcc_to_keep-1)

    if (frame_idx >= num_frames || k_out >= n_mfcc_to_keep) return;

    int k_in; // Index in mfcc_input_full (which has n_mels_input coefficients)
    if (exclude_zeroth) {
        k_in = k_out + 1; // Output[0] = Input[1], Output[1] = Input[2], ...
    } else {
        k_in = k_out;     // Output[0] = Input[0], Output[1] = Input[1], ...
    }

    if (k_in < n_mels_input) { // Ensure we don't read out of bounds from input
        mfcc_output_final[frame_idx * n_mfcc_to_keep + k_out] = mfcc_input_full[frame_idx * n_mels_input + k_in];
    } else {
        // Should not happen if n_mfcc_to_keep is reasonable (e.g. <= n_mels_input - (exclude_zeroth ? 1:0) )
        // Handle error or set to zero
        mfcc_output_final[frame_idx * n_mfcc_to_keep + k_out] = 0.0f;
    }
}
