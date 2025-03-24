/*
 * Fractional Hausdorff Distance (FHD) implementation for grayscale images
 * Based on: https://github.com/a-hamdi/GPU/blob/main/day%2014/cmpFHd_real_image.cu
 * 
 * Build with: 
 * nvcc fhd_image.cu -o fhd_image `pkg-config --cflags --libs opencv4`
 * or use CMake
 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

using namespace cv;
using namespace std;

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

/**
 * CUDA kernel to compute Fractional Hausdorff Distance
 * 
 * @param rPhi Real part of the FHD
 * @param iPhi Imaginary part of the FHD
 * @param phiMag Magnitude of the FHD (not used in this implementation)
 * @param x X coordinates
 * @param y Y coordinates
 * @param z Z coordinates (pixel intensities)
 * @param rMu Real part of the frequency components
 * @param iMu Imaginary part of the frequency components
 * @param M Number of frequency components in this chunk
 */
__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
                       float* x, float* y, float* z,
                       float* rMu, float* iMu, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    
    float xn_r = x[n]; 
    float yn_r = y[n]; 
    float zn_r = z[n];

    float rFhDn_r = rPhi[n]; 
    float iFhDn_r = iPhi[n];

    for (int m = 0; m < M; m++) {
        float expFhD = 2 * PI * (kx_c[m] * xn_r + ky_c[m] * yn_r + kz_c[m] * zn_r);
        
        float cArg = __cosf(expFhD);
        float sArg = __sinf(expFhD);

        rFhDn_r += rMu[m] * cArg - iMu[m] * sArg;
        iFhDn_r += iMu[m] * cArg + rMu[m] * sArg;
    }

    rPhi[n] = rFhDn_r;
    iPhi[n] = iFhDn_r;
}

/**
 * Main function
 */
int main(int argc, char** argv) {
    // Check if image path is provided
    string imagePath = "lena_gray.png";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    // Load an image using OpenCV
    cout << "Loading image: " << imagePath << endl;
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not open the image!" << endl;
        cerr << "Please provide a valid grayscale image path." << endl;
        return -1;
    }

    // Print image dimensions
    cout << "Image dimensions: " << image.cols << "x" << image.rows << endl;

    // Normalize image to range [0,1]
    image.convertTo(image, CV_32F, 1.0 / 255);

    int N = image.rows * image.cols; // Number of pixels
    int M = 256; // Number of frequency components
    
    // Check if we have enough threads
    if (N % FHD_THREADS_PER_BLOCK != 0) {
        cerr << "Warning: Image size is not a multiple of thread block size." << endl;
        cerr << "Some pixels may not be processed correctly." << endl;
    }
  
    float *x, *y, *z, *rMu, *iMu, *rPhi, *iPhi, *phiMag;
    
    // Allocate CUDA memory
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));
    cudaMallocManaged(&rMu, M * sizeof(float));
    cudaMallocManaged(&iMu, M * sizeof(float));
    cudaMallocManaged(&rPhi, N * sizeof(float));
    cudaMallocManaged(&iPhi, N * sizeof(float));
    cudaMallocManaged(&phiMag, N * sizeof(float));

    // Initialize x, y coordinates from image pixels
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            x[idx] = (float)j / image.cols;  // Normalize to [0,1]
            y[idx] = (float)i / image.rows;  // Normalize to [0,1]
            z[idx] = image.at<float>(i, j);  // Use intensity as "z"
            rPhi[idx] = z[idx];              // Initial real part
            iPhi[idx] = 0.0f;                // Initial imaginary part
        }
    }

    // Use random device for better randomization
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0, 1.0);

    // Initialize rMu and iMu with random values
    for (int i = 0; i < M; i++) {
        rMu[i] = dis(gen);
        iMu[i] = dis(gen);
    }

    // Calculate grid size
    int gridSize = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;
    cout << "CUDA grid size: " << gridSize << " blocks of " << FHD_THREADS_PER_BLOCK << " threads" << endl;

    // Process in chunks
    for (int i = 0; i < M / CHUNK_SIZE; i++) {
        // Copy chunks of kx, ky, kz to constant memory
        cudaMemcpyToSymbol(kx_c, &x[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(ky_c, &y[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(kz_c, &z[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        // Launch CUDA kernel
        cmpFHd<<<gridSize, FHD_THREADS_PER_BLOCK>>>(rPhi, iPhi, phiMag, x, y, z, rMu, iMu, CHUNK_SIZE);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
            return -1;
        }
        
        cudaDeviceSynchronize();
        cout << "Processed chunk " << i+1 << " of " << M/CHUNK_SIZE << endl;
    }

    // Convert results back to image format
    Mat outputImage(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            // Calculate magnitude of complex number
            outputImage.at<float>(i, j) = sqrt(rPhi[idx] * rPhi[idx] + iPhi[idx] * iPhi[idx]);
        }
    }

    // Normalize and save output image
    normalize(outputImage, outputImage, 0, 255, NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8U);
    
    string outputPath = "fhd_output.jpg";
    imwrite(outputPath, outputImage);

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(rMu);
    cudaFree(iMu);
    cudaFree(rPhi);
    cudaFree(iPhi);
    cudaFree(phiMag);

    cout << "FHD processing complete. Output saved as " << outputPath << endl;
    return 0;
}
