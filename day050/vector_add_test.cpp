#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include <stdexcept> // For std::runtime_error

// Forward declaration of the kernel and helper function from the .cu file
// Alternatively, create a header file (.h or .cuh)
extern "C" {
    void checkCudaError(cudaError_t err, const char *msg);
    __global__ void vectorAddKernel(const float *A, const float *B, float *C, int numElements);
}

// Helper function for CPU vector addition (for comparison)
void vectorAddCPU(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C_expected) {
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_C_expected[i] = h_A[i] + h_B[i];
    }
}

// Google Test fixture for Vector Add tests
class VectorAddTest : public ::testing::Test {
protected:
    int numElements;
    size_t sizeBytes;
    std::vector<float> h_A, h_B, h_C, h_C_expected;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // SetUp runs before each test case
    void SetUp() override {
        numElements = 1024 * 1024; // Test with a reasonable size
        sizeBytes = numElements * sizeof(float);

        // Allocate host memory using std::vector for easier management
        h_A.resize(numElements);
        h_B.resize(numElements);
        h_C.resize(numElements);
        h_C_expected.resize(numElements);

        // Initialize host vectors with random data
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for (int i = 0; i < numElements; ++i) {
            h_A[i] = distribution(generator);
            h_B[i] = distribution(generator);
        }

        // Allocate device memory
        checkCudaError(cudaMalloc((void **)&d_A, sizeBytes), "cudaMalloc d_A in SetUp");
        checkCudaError(cudaMalloc((void **)&d_B, sizeBytes), "cudaMalloc d_B in SetUp");
        checkCudaError(cudaMalloc((void **)&d_C, sizeBytes), "cudaMalloc d_C in SetUp");
    }

    // TearDown runs after each test case
    void TearDown() override {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        d_A = d_B = d_C = nullptr;
        // Host vectors are automatically freed by std::vector destructor
    }

    // Helper function to run the kernel and check results
    void runTest() {
        // Copy input data from host to device
        checkCudaError(cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
        checkCudaError(cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");

        // Launch the kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        checkCudaError(cudaGetLastError(), "vectorAddKernel launch");
        checkCudaError(cudaDeviceSynchronize(), "vectorAddKernel execution");

        // Copy result back from device to host
        checkCudaError(cudaMemcpy(h_C.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C");

        // Compute expected result on CPU
        vectorAddCPU(h_A, h_B, h_C_expected);

        // Verify the result using Google Test assertions
        for (int i = 0; i < numElements; ++i) {
            ASSERT_NEAR(h_C[i], h_C_expected[i], 1e-5) << "Mismatch at element " << i;
        }
    }
};

// Test case using the fixture
TEST_F(VectorAddTest, BasicVerification) {
    runTest();
}

// Example of another test case (e.g., testing with zeros)
TEST_F(VectorAddTest, ZeroVectors) {
    // Overwrite initial data with zeros
    std::fill(h_A.begin(), h_A.end(), 0.0f);
    std::fill(h_B.begin(), h_B.end(), 0.0f);
    runTest(); // Expected result is all zeros, which vectorAddCPU will compute
}

// Example of a test case with negative numbers
TEST_F(VectorAddTest, NegativeNumbers) {
    // Overwrite initial data with negative numbers
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0f, 0.0f);
     for (int i = 0; i < numElements; ++i) {
        h_A[i] = distribution(generator);
        h_B[i] = distribution(generator);
    }
    runTest();
}

// Note: The main function is typically provided by gtest_main library,
// which will be linked via CMake. No need to write main() here.
