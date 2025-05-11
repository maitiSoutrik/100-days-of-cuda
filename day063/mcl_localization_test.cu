#include "gtest/gtest.h"
#include "mcl_localization.cuh"
#include <vector>
#include <numeric> // For std::accumulate
#include <cmath>   // For std::abs

// Test fixture for MCL tests
class MCLTest : public ::testing::Test {
protected:
    const int test_grid_dim = 4; // Small grid for testing: 4x4 = 16 states
    const int test_num_states = test_grid_dim * test_grid_dim;
    TransitionMatrix* matrix;

    void SetUp() override {
        matrix = new TransitionMatrix(test_num_states);
        initialize_synthetic_grid_world(*matrix, test_grid_dim);
    }

    void TearDown() override {
        delete matrix;
        matrix = nullptr;
    }
};

// Test TransitionMatrix construction and basic properties
TEST_F(MCLTest, MatrixInitialization) {
    ASSERT_NE(matrix->data, nullptr);
    ASSERT_EQ(matrix->num_states, test_num_states);

    std::vector<float> host_matrix(test_num_states * test_num_states);
    matrix->copy_to_host(host_matrix.data()); // Pass raw pointer
    // ASSERT_EQ(host_matrix.size(), static_cast<size_t>(test_num_states * test_num_states)); // Size is guaranteed by vector construction

    // Check if columns are somewhat normalized (sum to approx 1.0)
    // This is a basic check for initialize_synthetic_grid_world's normalization
    for (int j = 0; j < test_num_states; ++j) {
        float col_sum = 0.0f;
        for (int i = 0; i < test_num_states; ++i) {
            col_sum += host_matrix[i * test_num_states + j];
        }
        ASSERT_NEAR(col_sum, 1.0f, 1e-5f) << "Column " << j << " does not sum to 1.";
    }
}

// Test a single MCL iteration (Expansion + Inflation)
TEST_F(MCLTest, SingleMCLIteration) {
    std::vector<float> matrix_before_iter(test_num_states * test_num_states);
    matrix->copy_to_host(matrix_before_iter.data());

    float inflation_factor = 2.0f;
    mcl_iteration_cuda(*matrix, inflation_factor);

    std::vector<float> matrix_after_iter(test_num_states * test_num_states);
    matrix->copy_to_host(matrix_after_iter.data());

    // ASSERT_EQ(matrix_after_iter.size(), matrix_before_iter.size()); // Size is guaranteed

    // Check that columns are still normalized after iteration
    for (int j = 0; j < test_num_states; ++j) {
        float col_sum = 0.0f;
        for (int i = 0; i < test_num_states; ++i) {
            col_sum += matrix_after_iter[i * test_num_states + j];
        }
        ASSERT_NEAR(col_sum, 1.0f, 1e-4f) << "Column " << j << " does not sum to 1 after iteration.";
    }
    
    // A very basic check: ensure the matrix changed.
    // This is not a strong test of correctness but catches complete failures.
    bool changed = false;
    for(size_t i = 0; i < matrix_before_iter.size(); ++i) {
        if (std::abs(matrix_before_iter[i] - matrix_after_iter[i]) > 1e-7f) {
            changed = true;
            break;
        }
    }
    // It's possible for some specific initial matrices and inflation factors that it doesn't change much
    // or converges in one step, but for typical synthetic data, it should change.
    // If this fails, it might indicate an issue or a very stable initial state.
    // For now, we expect some change.
    EXPECT_TRUE(changed) << "Matrix did not change after one MCL iteration.";
}


// Test cluster extraction (very basic)
TEST_F(MCLTest, ClusterExtraction) {
    // Run a few iterations to make probabilities more distinct
    for(int i=0; i<5; ++i) {
        mcl_iteration_cuda(*matrix, 2.0f);
    }

    float threshold = 0.001f; // Low threshold for a small grid
    std::vector<State> clusters = extract_clusters_from_probabilities(*matrix, threshold, test_grid_dim);
    
    // For a small grid and synthetic data, we expect some states to be identified.
    // The exact number depends heavily on the initialization and iterations.
    // This is more of a sanity check that the function runs and returns something.
    // A more robust test would require a known input matrix that produces specific clusters.
    EXPECT_FALSE(clusters.empty()) << "No clusters extracted, or all probabilities are below threshold.";

    for(const auto& state : clusters) {
        EXPECT_GE(state.probability, threshold);
        EXPECT_GE(state.x, 0);
        EXPECT_LT(state.x, test_grid_dim);
        EXPECT_GE(state.y, 0);
        EXPECT_LT(state.y, test_grid_dim);
    }
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
