# Day 066: GPU-Accelerated MFCC Feature Extraction

## Overview

This project implements the extraction of Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal, accelerated using CUDA. MFCCs are a crucial feature representation in Automatic Speech Recognition (ASR) and other audio processing tasks. They capture the perceptually relevant characteristics of the audio signal.

The process involves several steps:
1.  **Reading Audio**: Loading a `.wav` file.
2.  **(Optional) Pre-emphasis**: Boosting high-frequency components.
3.  **Framing**: Dividing the audio signal into short, overlapping frames.
4.  **Windowing**: Applying a window function (e.g., Hamming) to each frame to reduce spectral leakage.
5.  **Fast Fourier Transform (FFT)**: Converting each frame from the time domain to the frequency domain.
6.  **Power Spectrum**: Calculating the magnitude squared of the FFT output.
7.  **Mel Filterbank Application**: Applying a set of triangular filters, spaced according to the Mel scale, to the power spectrum to get Mel energies.
8.  **Logarithm**: Taking the logarithm of the Mel energies.
9.  **Discrete Cosine Transform (DCT)**: Applying DCT to the log Mel energies to decorrelate them and obtain MFCCs.

This project aims to parallelize these steps on the GPU using CUDA and cuFFT.

## Implementation Details

### WAV File Reading
A C++ function `readWavFile` in `audio_features.cu` handles reading mono, 16-bit PCM `.wav` files. It parses the header and extracts raw audio samples, normalizing them to floating-point values between -1.0 and 1.0.

### MFCC Parameterization
The following parameters are used (based on a 44.1 kHz sample rate, configurable via constants/constructor):
- **Sample Rate**: 44100 Hz (determined from input WAV)
- **Window Duration**: 25 ms (`N_WINDOW = 1102` samples)
- **Hop Duration**: 10 ms (`N_HOP = 441` samples)
- **FFT Size**: 2048 (`N_FFT`, next power of 2 from `N_WINDOW`)
- **Number of Mel Filters**: 40 (`N_MELS`)
- **Number of MFCC Coefficients**: 13 (`N_MFCC`, typically coefficients 1-13)
- **Pre-emphasis Alpha**: 0.97 (if applied)

### CUDA Pipeline (`GpuMfccExtractor` class)
The core logic is encapsulated in the `GpuMfccExtractor` class.

1.  **Memory Management**:
    - Device memory is allocated for various stages: raw audio, pre-emphasized audio, framed/windowed audio, FFT output, power spectrum, Mel energies, log Mel energies, and final MFCCs.
    - Mel filterbank and DCT matrices (if custom DCT is used) are pre-computed on the host and copied to device memory.

2.  **Pre-emphasis Kernel (`preemphasis_kernel`)**:
    - A simple kernel applies `y[t] = x[t] - alpha * x[t-1]`. (Currently stubbed, needs full implementation and launch configuration).

3.  **Framing and Windowing Kernel (`frame_and_window_kernel`)**:
    - Each CUDA block processes one audio frame.
    - Threads within a block cooperatively load audio data for the frame into shared memory.
    - A Hamming window (`w[n] = 0.54 - 0.46 * cos(2 * PI * n / (N_WINDOW - 1))`) is applied to the `N_WINDOW` samples.
    - The windowed frame is zero-padded to `N_FFT` and written to global memory.
    - *Requires shared memory allocation at launch: `n_window * sizeof(float)` per block.*

4.  **FFT (cuFFT)**:
    - `cufftPlanMany` is used to create a plan for batched 1D Real-to-Complex (R2C) FFTs.
    - `cufftExecR2C` processes all frames in parallel. Output size per frame is `(N_FFT/2 + 1)` complex values.

5.  **Power Spectrum Kernel (`power_spectrum_kernel`)**:
    - Calculates `P = real*real + imag*imag` for each complex FFT output bin.

6.  **Mel Filterbank Application Kernel (`mel_filterbank_kernel`)**:
    - The Mel filterbank matrix (N_MELS x (N_FFT/2 + 1)) is stored in global device memory (flattened).
    - Each block computes Mel energies for one frame. Each thread within the block computes one of the `N_MELS` energies by performing a dot product between the frame's power spectrum and a Mel filter.

7.  **Log Energies Kernel (`log_energies_kernel`)**:
    - Computes `log(mel_energy + epsilon)` for each Mel energy.

8.  **DCT (cuFFT)**:
    - `cufftPlanMany` is used to create a plan for batched 1D Real-to-Real (R2R) DCTs (Type II).
    - `cufftExecR2R` processes the log Mel energies of all frames. Output is `N_MELS` DCT coefficients per frame.
    - (A custom `dct_kernel` is also prototyped for matrix-multiplication based DCT if cuFFT's R2R is not suitable or for comparison).

9.  **Select MFCC Coefficients Kernel (`select_mfcc_coeffs_kernel`)**:
    - Selects the desired number of MFCCs (e.g., coefficients 1 through 13, discarding the 0th).

### Host Orchestration (`mfcc_main.cu`)
- Parses command-line arguments for input WAV and output MFCC file.
- Calls `readWavFile`.
- Initializes `GpuMfccExtractor` with appropriate parameters.
- Calls `computeMfccs` method to run the GPU pipeline.
- Times the GPU processing.
- Saves the resulting MFCC matrix to a text file.

## Key CUDA Features Used
- **cuFFT**: For batched FFT (R2C) and DCT (R2R) operations.
- **Shared Memory**: Used in the `frame_and_window_kernel` to efficiently apply the windowing function.
- **CUDA Kernels**: Custom kernels for pre-emphasis, framing/windowing, power spectrum, Mel filterbank application, log energies, and MFCC selection.
- **Batched Operations**: Leveraging cuFFT's ability to process multiple transforms (frames) in parallel.
- **CUDA Events**: Used for accurate timing of the GPU execution part (to be added in `mfcc_main.cu` around `computeMfccs` call).

## Performance Considerations
- **Data Transfer**: Minimizing Host-to-Device and Device-to-Host transfers is crucial. Currently, raw audio is copied once, and final MFCCs are copied back.
- **Kernel Efficiency**: Optimizing grid/block dimensions and memory access patterns within kernels.
- **Batched cuFFT**: Significantly faster than processing FFTs/DCTs frame by frame sequentially.
- **Shared Memory Usage**: Reduces global memory accesses in the windowing kernel.
- **Occupancy**: Choosing appropriate block sizes for kernels to maximize GPU utilization.

## Building and Running

### Prerequisites
- CUDA Toolkit (>= 10.x, compatible with Compute Capability 5.3 for Jetson Nano)
- CMake (>= 3.10)
- A C++14 compatible compiler (e.g., g++)
- Google Test (will be fetched by the root `CMakeLists.txt` or needs to be installed)

### Build Steps (from the root project directory `100-days-of-cuda`)
1.  Ensure the `day066` subdirectory is added to the root `CMakeLists.txt`:
    ```cmake
    # In root CMakeLists.txt
    add_subdirectory(day066)
    ```
2.  Configure and build:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make mfcc_benchmark mfcc_tests -j$(nproc) 
    # Or simply 'make' to build everything
    ```

### Running the Benchmark
The executable will be in `build/day066/`.
```bash
./build/day066/mfcc_benchmark ../inputs/audio-sample-cuda-challenge.wav output_mfccs_day066.txt
```
(Assuming `audio-sample-cuda-challenge.wav` is in `build/inputs/` relative to the project root, or provide the full/correct relative path from where you run the executable).
The output MFCCs will be saved to `output_mfccs_day066.txt` (or the specified output file) in the directory where the command is run.

### Running Tests
```bash
cd build # (if not already there)
ctest --output-on-failure -R day066_mfcc_extraction # Run tests for this day
# Or directly: ./day066/mfcc_tests
```

## Execution Results

The code was compiled and run on an NVIDIA Jetson Nano.

### Benchmark Output (`mfcc_benchmark`)
The benchmark executable was run with the sample audio file:
```bash
drboom@JetNano ~/g/1/build> ./day066/mfcc_benchmark inputs/audio-sample-cuda-challenge.wav output/mfccs_day066.txt
Reading WAV file: inputs/audio-sample-cuda-challenge.wav
WAV file read successfully:
  Sample Rate: 44100 Hz
  Channels: 1
  Bits per Sample: 16
  Total Samples: 731136
  Duration: 16.579 seconds

MFCC Parameters:
  N_Window: 1102 samples (25 ms)
  N_Hop: 441 samples (10 ms)
  N_FFT: 2048
  N_Mels: 40
  N_MFCC (coeffs to keep): 13

Starting MFCC computation on GPU...
MFCC computation successful.
  Time taken for GPU MFCC extraction: 1413.86 ms
  Output: 1656 frames x 13 coefficients.
MFCCs saved to output/mfccs_day066.txt
```
The output MFCC features were saved to `output/mfccs_day066.txt`. The file contains a matrix of 1656 frames by 13 coefficients.

### Test Output (`mfcc_tests`)
The Google Tests for the project passed:
```bash
drboom@JetNano ~/g/1/build> ./day066/mfcc_tests 
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from AudioFeaturesTest
[ RUN      ] AudioFeaturesTest.WavReadTest_Basic
[       OK ] AudioFeaturesTest.WavReadTest_Basic (0 ms)
[ RUN      ] AudioFeaturesTest.MelFilterbankCreation
[       OK ] AudioFeaturesTest.MelFilterbankCreation (0 ms)
[ RUN      ] AudioFeaturesTest.DctMatrixCreation
[       OK ] AudioFeaturesTest.DctMatrixCreation (0 ms)
[ RUN      ] AudioFeaturesTest.GpuMfccExtractor_SimpleRun
GpuMfccExtractor_SimpleRun produced 98 frames and 13 coeffs.
First MFCC value: 26.9937
[       OK ] AudioFeaturesTest.GpuMfccExtractor_SimpleRun (1315 ms)
[----------] 4 tests from AudioFeaturesTest (1316 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (1316 ms total)
[  PASSED  ] 4 tests.
```

## Learnings and Observations
- Successfully set up a complex audio feature extraction pipeline (MFCC) using CUDA.
- Utilized cuFFT for FFT operations and a custom kernel for DCT due to potential compatibility issues with `CUFFT_R2R` on the target platform.
- Implemented various CUDA kernels for different stages: pre-emphasis, framing/windowing (with shared memory), power spectrum, Mel filterbank application, log energies, and coefficient selection.
- Integrated Google Test for unit testing components.
- The initial run on Jetson Nano shows the pipeline is functional, producing output of the correct dimensions. The reported time for MFCC extraction for a ~16.5s audio file was ~1.4 seconds.
- The custom DCT kernel approach worked as a fallback, highlighting the importance of understanding potential API differences or limitations across CUDA versions/environments.
- Further work would involve detailed numerical validation against a reference library (e.g., Librosa) and performance profiling/optimization.
- Challenges in implementing each step of the MFCC pipeline.
- Debugging CUDA kernels and cuFFT usage.
- Performance characteristics on the Jetson Nano.
- Comparison with CPU-based MFCC extraction (if attempted).

## Future Improvements
- Implement and test the pre-emphasis kernel.
- Fully implement and optimize all custom CUDA kernels (framing/windowing, power spectrum, Mel filterbank, log energies, MFCC selection).
- Add more robust error handling and parameter validation.
- Compare results against a reference implementation like Librosa for numerical accuracy.
- Investigate further optimizations (e.g., kernel fusion, more advanced cuFFT planning).
- Add support for stereo audio or different bit depths.

## References
- Librosa documentation: [https://librosa.org/doc/latest/feature.html](https://librosa.org/doc/latest/feature.html)
- Jurafsky, D., & Martin, J. H. (2009). *Speech and Language Processing*. Prentice Hall. (Chapter on Phonetics, Speech Features)
- CUDA cuFFT Library Documentation: [https://docs.nvidia.com/cuda/cufft/index.html](https://docs.nvidia.com/cuda/cufft/index.html)
