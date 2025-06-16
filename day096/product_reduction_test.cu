#include "product_reduction.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <random>

#define CHECK_CUDA_ERROR_GTEST(err) \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)

// CPU reference implementation
static void cpu_product_reduce(const std::vector<float>& input,
                               int dim,
                               const std::vector<size_t>& shape,
                               std::vector<float>& output) {
    size_t ndim = shape.size();
    size_t before = 1, after = 1;
    for (int i = 0; i < dim; ++i) before *= shape[i];
    for (size_t i = dim + 1; i < ndim; ++i) after *= shape[i];
    size_t dim_size = shape[dim];

    output.resize(before * after, 1.0f);
    for (size_t b = 0; b < before; ++b) {
        for (size_t a = 0; a < after; ++a) {
            float prod = 1.f;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = (b * dim_size + d) * after + a;
                prod *= input[idx];
            }
            output[b * after + a] = prod;
        }
    }
}

class ProductReductionTest : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, int>> {};

TEST_P(ProductReductionTest, MatchesCPU) {
    auto params = GetParam();
    const std::vector<size_t> shape = std::get<0>(params);
    const int dim = std::get<1>(params);

    size_t total_elems = 1;
    for (size_t s : shape) total_elems *= s;
    if (total_elems == 0) GTEST_SKIP() << "Zero-sized tensor; behaviour undefined for this test.";

    std::vector<float> h_input(total_elems);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.5f, 2.0f);
    for (float& v : h_input) v = dist(rng);

    // Device allocations
    float *d_input, *d_output; // size_t *d_shape; // d_shape no longer needed on device
    size_t before=1, after=1; for(int i=0;i<dim;++i) before*=shape[i]; for(size_t i=dim+1;i<shape.size();++i) after*=shape[i];
    size_t out_elems = before*after;

    CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input, total_elems*sizeof(float)));
    CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output, out_elems*sizeof(float)));
    // CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_shape, shape.size()*sizeof(size_t))); // d_shape no longer needed

    CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input, h_input.data(), total_elems*sizeof(float), cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_shape, shape.data(), shape.size()*sizeof(size_t), cudaMemcpyHostToDevice)); // d_shape no longer needed

    product_reduction_dimension_cuda(d_input, dim, d_output, shape.data(), shape.size()); // Pass host shape.data()
    CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize());

    std::vector<float> h_output(out_elems);
    CHECK_CUDA_ERROR_GTEST(cudaMemcpy(h_output.data(), d_output, out_elems*sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_ref;
    cpu_product_reduce(h_input, dim, shape, h_ref);

    for (size_t i = 0; i < out_elems; ++i) {
        ASSERT_NEAR(h_ref[i], h_output[i], 1e-4) << "Mismatch at index " << i;
    }

    cudaFree(d_input); cudaFree(d_output); // cudaFree(d_shape); // d_shape was not allocated
}

INSTANTIATE_TEST_SUITE_P(
    ProductReductionTests,
    ProductReductionTest,
    ::testing::Values(
        std::make_tuple(std::vector<size_t>{2,3,4},0),
        std::make_tuple(std::vector<size_t>{2,3,4},1),
        std::make_tuple(std::vector<size_t>{2,3,4},2),
        std::make_tuple(std::vector<size_t>{128,64},1),
        std::make_tuple(std::vector<size_t>{16,16,16,8},3)
    ));

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
