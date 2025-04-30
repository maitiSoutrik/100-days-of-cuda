#include <gtest/gtest.h>
#include "min_gru_scan.h" // Include the header with function declarations
#include <vector>
#include <cmath> // For fabsf
#include <stdlib.h> // For rand, srand
#include <time.h>   // For time

// Define CHECK_CUDA_ERROR locally for tests if not in a common header
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        FAIL() << "CUDA error: " << cudaGetErrorString(err); \
    } \
}


// Test fixture for MinGRU tests
class MinGRUScanTest : public ::testing::Test {
protected:
    MinGRUCell cell;
    int input_size = 16;  // Smaller dimensions for faster tests
    int hidden_size = 32;
    int seq_length = 10;

    float* h_x = nullptr;
    float* h_h0 = nullptr;
    float* h_out_cpu = nullptr;
    float* h_out_cuda = nullptr;

    size_t input_seq_elems;
    size_t hidden_state_elems;
    size_t hidden_seq_elems;

    void SetUp() override {
        // Seed random for consistent tests (though weights are random)
        // Use a fixed seed for reproducibility if needed: srand(12345);
        srand((unsigned int)time(NULL));

        input_seq_elems = (size_t)seq_length * input_size;
        hidden_state_elems = (size_t)hidden_size;
        hidden_seq_elems = (size_t)seq_length * hidden_size;

        // Initialize cell
        init_min_gru_cell(&cell, input_size, hidden_size);

        // Allocate host memory
        h_x = (float*)malloc(input_seq_elems * sizeof(float));
        h_h0 = (float*)malloc(hidden_state_elems * sizeof(float));
        h_out_cpu = (float*)malloc(hidden_seq_elems * sizeof(float));
        h_out_cuda = (float*)malloc(hidden_seq_elems * sizeof(float));

        ASSERT_NE(h_x, nullptr);
        ASSERT_NE(h_h0, nullptr);
        ASSERT_NE(h_out_cpu, nullptr);
        ASSERT_NE(h_out_cuda, nullptr);

        // Generate random data
        generate_random_data(h_x, input_seq_elems, -0.5f, 0.5f); // Smaller range potentially
        generate_random_data(h_h0, hidden_state_elems, -0.5f, 0.5f);
    }

    void TearDown() override {
        free(h_x);
        free(h_h0);
        free(h_out_cpu);
        free(h_out_cuda);
        free_min_gru_cell(&cell);
    }

    // Helper to run comparison
    void RunAndCompare(float tolerance = 1e-4f) {
        // Run CPU version
        min_gru_process_sequence_cpu(&cell, h_x, h_h0, seq_length, h_out_cpu);

        // Run CUDA version
        min_gru_process_sequence_cuda(&cell, h_x, h_h0, seq_length, h_out_cuda);

        // Compare results
        for (size_t i = 0; i < hidden_seq_elems; ++i) {
            EXPECT_NEAR(h_out_cpu[i], h_out_cuda[i], tolerance)
                << "Mismatch at index " << i;
            // Break early on first failure for clarity, if desired
            if (HasFailure()) {
                 printf("First mismatch: CPU=%.8f, CUDA=%.8f at index %zu\n",
                        h_out_cpu[i], h_out_cuda[i], i);
                 break;
            }
        }
    }
};

// Define the test case
TEST_F(MinGRUScanTest, CompareCPUAndCUDA) {
    RunAndCompare(); // Use default tolerance
}

TEST_F(MinGRUScanTest, CompareCPUAndCUDA_ShortSequence) {
    // Override sequence length for this specific test
    seq_length = 5; // Test the sequential path in CUDA scan impl
    TearDown(); // Clean up previous setup
    SetUp();    // Re-setup with new seq_length
    RunAndCompare();
}

// Add more tests if needed (e.g., different dimensions, edge cases)


// Main function to run tests (usually provided by gtest_main)
// If not linking gtest_main, uncomment this:
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
