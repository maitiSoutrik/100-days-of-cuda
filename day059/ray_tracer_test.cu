#include "gtest/gtest.h"
#include "ray_tracer.cuh" // For Vec3, constants, and CHECK_CUDA_ERROR

// Test fixture for Vec3 tests
class Vec3Test : public ::testing::Test {
protected:
    Vec3 v1{1.0f, 2.0f, 3.0f};
    Vec3 v2{4.0f, 5.0f, 6.0f};
    Vec3 zero{0.0f, 0.0f, 0.0f};
};

// Host-side wrapper for device Vec3 operations for testing
// These functions copy Vec3 to device, call the __device__ function via a kernel, and copy back.
// This is a bit heavy for simple struct ops but demonstrates testing device code.
// A simpler approach for __device__ functions not inside __global__ kernels is often
// to just trust nvcc or test them as part of a larger kernel's output.
// For this example, we'll make them testable by calling them from a simple kernel.

__global__ void testVec3AddKernel(Vec3 a, Vec3 b, Vec3* result) {
    *result = a + b;
}
__global__ void testVec3SubKernel(Vec3 a, Vec3 b, Vec3* result) {
    *result = a - b;
}
__global__ void testVec3MulKernel(Vec3 a, float s, Vec3* result) {
    *result = a * s;
}
__global__ void testVec3DotKernel(Vec3 a, Vec3 b, float* result) {
    *result = a.dot(b);
}
__global__ void testVec3NormalizeKernel(Vec3 a, Vec3* result) {
    *result = a.normalize();
}

TEST_F(Vec3Test, Addition) {
    Vec3 expected(5.0f, 7.0f, 9.0f);
    Vec3 *d_result;
    Vec3 h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(Vec3)));
    testVec3AddKernel<<<1,1>>>(v1, v2, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(Vec3), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result.x, expected.x);
    EXPECT_FLOAT_EQ(h_result.y, expected.y);
    EXPECT_FLOAT_EQ(h_result.z, expected.z);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

TEST_F(Vec3Test, Subtraction) {
    Vec3 expected(-3.0f, -3.0f, -3.0f);
    Vec3 *d_result;
    Vec3 h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(Vec3)));
    testVec3SubKernel<<<1,1>>>(v1, v2, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(Vec3), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result.x, expected.x);
    EXPECT_FLOAT_EQ(h_result.y, expected.y);
    EXPECT_FLOAT_EQ(h_result.z, expected.z);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

TEST_F(Vec3Test, ScalarMultiplication) {
    Vec3 expected(2.0f, 4.0f, 6.0f);
    Vec3 *d_result;
    Vec3 h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(Vec3)));
    testVec3MulKernel<<<1,1>>>(v1, 2.0f, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(Vec3), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result.x, expected.x);
    EXPECT_FLOAT_EQ(h_result.y, expected.y);
    EXPECT_FLOAT_EQ(h_result.z, expected.z);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

TEST_F(Vec3Test, DotProduct) {
    float expected = 1.0f * 4.0f + 2.0f * 5.0f + 3.0f * 6.0f; // 4 + 10 + 18 = 32
    float *d_result;
    float h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(float)));
    testVec3DotKernel<<<1,1>>>(v1, v2, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result, expected);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

TEST_F(Vec3Test, Normalization) {
    Vec3 v(3.0f, 4.0f, 0.0f); // Length 5
    Vec3 expected(3.0f/5.0f, 4.0f/5.0f, 0.0f/5.0f);
    Vec3 *d_result;
    Vec3 h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(Vec3)));
    testVec3NormalizeKernel<<<1,1>>>(v, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(Vec3), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result.x, expected.x);
    EXPECT_FLOAT_EQ(h_result.y, expected.y);
    EXPECT_FLOAT_EQ(h_result.z, expected.z);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

TEST_F(Vec3Test, NormalizationZeroVector) {
    Vec3 expected(0.0f, 0.0f, 0.0f);
     Vec3 *d_result;
    Vec3 h_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(Vec3)));
    testVec3NormalizeKernel<<<1,1>>>(zero, d_result);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(Vec3), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h_result.x, expected.x);
    EXPECT_FLOAT_EQ(h_result.y, expected.y);
    EXPECT_FLOAT_EQ(h_result.z, expected.z);
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

// Test for the main render kernel (basic check)
TEST(RenderTest, KernelExecutesAndProducesOutput) {
    unsigned char *d_image, *h_image;
    size_t image_size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);

    ASSERT_EQ(cudaMalloc((void **)&d_image, image_size), cudaSuccess);
    h_image = (unsigned char *)malloc(image_size);
    ASSERT_NE(h_image, nullptr);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    render<<<numBlocks, threadsPerBlock>>>(d_image);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost), cudaSuccess);

    // Basic check: ensure the first pixel is not garbage (e.g. all 0xFF or uninitialized)
    // This is a very weak check. A better check might involve rendering a known simple scene
    // and checking specific pixel values, or checksumming the output image.
    // For now, just check if some data was written.
    // If the background is (15, 25, 40), the first pixel should be this if it doesn't hit the sphere.
    // Let's assume the top-left pixel (0,0) misses the sphere.
    if (WIDTH > 0 && HEIGHT > 0) {
         // From ray_tracer.cu, background is R=15, G=25, B=40
        EXPECT_EQ(h_image[0], 15); 
        EXPECT_EQ(h_image[1], 25);
        EXPECT_EQ(h_image[2], 40);
    }

    // Attempt to save the image for visual inspection if tests are run manually
    // In CI, this might not be useful unless artifacts are saved.
    // FILE *f = fopen("test_output.ppm", "wb");
    // if (f) {
    //     fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    //     fwrite(h_image, 1, image_size, f);
    //     fclose(f);
    // }

    cudaFree(d_image);
    free(h_image);
}

// Entry point for Google Test
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
// The main for GTest is usually linked via GTest::gtest_main
