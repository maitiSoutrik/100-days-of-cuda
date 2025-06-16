#include "product_reduction.cuh"
#include <iostream>
#include <vector>
#include <numeric>

// Helper: CPU reference implementation for verification
void cpu_product_reduce(const std::vector<float>& input,
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

int main() {
    // Example tensor of shape 2 x 3 x 4
    std::vector<size_t> shape = {2, 3, 4};
    size_t total_elems = 2 * 3 * 4;
    std::vector<float> h_input(total_elems);
    std::iota(h_input.begin(), h_input.end(), 1.0f); // 1..24

    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    // size_t* d_shape = nullptr; // d_shape is no longer needed on device

    int reduce_dim = 1; // reduce over middle dimension (size 3)

    size_t before = shape[0];
    size_t after = shape[2];
    size_t out_elems = before * after;

    cudaMalloc(&d_input, sizeof(float) * total_elems);
    cudaMalloc(&d_output, sizeof(float) * out_elems);
    // cudaMalloc(&d_shape, sizeof(size_t) * shape.size()); // d_shape is no longer needed

    cudaMemcpy(d_input, h_input.data(), sizeof(float) * total_elems, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_shape, shape.data(), sizeof(size_t) * shape.size(), cudaMemcpyHostToDevice); // d_shape is no longer needed

    product_reduction_dimension_cuda(d_input, reduce_dim, d_output, shape.data(), shape.size()); // Pass host shape.data()

    std::vector<float> h_output(out_elems);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * out_elems, cudaMemcpyDeviceToHost);

    // CPU reference
    std::vector<float> ref_output;
    cpu_product_reduce(h_input, reduce_dim, shape, ref_output);

    std::cout << "GPU result vs CPU reference:\n";
    bool correct = true;
    for (size_t i = 0; i < out_elems; ++i) {
        std::cout << "  " << h_output[i] << "  (ref=" << ref_output[i] << ")\n";
        if (fabs(h_output[i] - ref_output[i]) > 1e-4f) correct = false;
    }
    std::cout << (correct ? "SUCCESS" : "MISMATCH") << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    // cudaFree(d_shape); // d_shape was not allocated
    return correct ? 0 : 1;
}
