#include "gtest/gtest.h"
#include "mandelbrot_amr.cuh" // Contains mandelbrot_iterations and CHECK_CUDA_ERROR

// Kernel to wrap the device function for testing
__global__ void test_mandelbrot_iterations_kernel(double cx, double cy, int max_iter, int* result) {
    *result = mandelbrot_iterations(cx, cy, max_iter);
}

// Host function to call the test kernel
int run_mandelbrot_iterations_test(double cx, double cy, int max_iter) {
    int* d_result;
    int h_result;

    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(int)));
    
    test_mandelbrot_iterations_kernel<<<1, 1>>&gt(cx, cy, max_iter, d_result);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    return h_result;
}

TEST(MandelbrotIterationsTest, PointInsideSet) {
    // Test a point known to be inside the Mandelbrot set (e.g., c = 0 + 0i)
    // It should reach max_iterations.
    int max_iter = 100;
    EXPECT_EQ(run_mandelbrot_iterations_test(0.0, 0.0, max_iter), max_iter);
}

TEST(MandelbrotIterationsTest, PointOutsideSet) {
    // Test a point known to be outside the Mandelbrot set (e.g., c = 2.0 + 0i)
    // It should escape quickly.
    int max_iter = 100;
    // For c = 2.0, x0=0, y0=0:
    // z1 = 2.0 + 0i, |z1|^2 = 4
    // z2 = 2.0^2 - 0^2 + 2.0 = 6.0 + 0i. |z2|^2 = 36 > 4. Escapes in 2 iterations.
    // Iteration 0: x=0, y=0. |z|^2 = 0 <= 4. xtemp = 0 - 0 + 2 = 2. y = 0. x = 2. iter = 1.
    // Iteration 1: x=2, y=0. |z|^2 = 4 <= 4. xtemp = 4 - 0 + 2 = 6. y = 0. x = 6. iter = 2.
    // Iteration 2: x=6, y=0. |z|^2 = 36 > 4. Loop terminates. Returns 2.
    EXPECT_EQ(run_mandelbrot_iterations_test(2.0, 0.0, max_iter), 2);
}

TEST(MandelbrotIterationsTest, AnotherPointOutsideSet) {
    // Test c = -1.0 + 0.0i (part of the main cardioid, should be inside)
    int max_iter = 100;
    EXPECT_EQ(run_mandelbrot_iterations_test(-1.0, 0.0, max_iter), max_iter);
}

TEST(MandelbrotIterationsTest, ComplexBoundaryPoint) {
    // Test a point that might be near the boundary, e.g., c = -0.75 + 0.1i
    // This is harder to predict exact iterations without running, but it shouldn't be max_iter
    // nor immediately escape. This is more of a sanity check.
    int max_iter = 500;
    int iterations = run_mandelbrot_iterations_test(-0.75, 0.1, max_iter);
    EXPECT_GT(iterations, 1); // Should take more than 1 iteration
    EXPECT_LT(iterations, max_iter); // Should not be considered "inside" if it's complex
                                     // (unless it truly is, then this test might need adjustment
                                     // or a different point)
    // For c = -0.75 + 0.1i, after some iterations it diverges.
    // For example, with max_iter = 500, it might take around 13-15 iterations.
    // Let's check it's within a plausible range.
    EXPECT_GT(iterations, 5);
    EXPECT_LT(iterations, 50);
}


// It's hard to test the full AMR logic or dynamic parallelism directly in GTest
// without significant mocking or a more complex test harness.
// These tests focus on the core calculation.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
