#include "gtest/gtest.h"
#include "bitonic_sort.cuh"
#include <vector>
#include <algorithm> // For std::sort, std::is_sorted
#include <stdlib.h>  // For rand(), srand()
#include <time.h>    // For time()

// N_CONST is defined in bitonic_sort.cu (e.g., 1024)
// We need this value for test array declarations.
// For consistency, it should ideally be in bitonic_sort.cuh or a shared config.
// Here, we'll use a local constant that must match the one in bitonic_sort.cu.
const int TEST_N_CONST = 1024;

// Helper function to compare two float arrays
::testing::AssertionResult AreArraysEqual(const float* arr1, const float* arr2, int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(arr1[i] - arr2[i]) > 1e-5) { // Using an epsilon for float comparison
            return ::testing::AssertionFailure() << "Arrays differ at index " << i
                                                 << ": arr1[" << i << "] = " << arr1[i]
                                                 << ", arr2[" << i << "] = " << arr2[i];
        }
    }
    return ::testing::AssertionSuccess();
}

TEST(BitonicSortTest, RandomFloats) {
    std::vector<float> h_array_gpu(TEST_N_CONST);
    std::vector<float> h_array_cpu(TEST_N_CONST);

    srand(time(NULL));
    for (int i = 0; i < TEST_N_CONST; ++i) {
        float val = (float)(rand() % 10000) / 100.0f;
        h_array_gpu[i] = val;
        h_array_cpu[i] = val;
    }

    bitonic_sort_gpu(h_array_gpu.data(), TEST_N_CONST);
    std::sort(h_array_cpu.begin(), h_array_cpu.end());

    EXPECT_TRUE(AreArraysEqual(h_array_gpu.data(), h_array_cpu.data(), TEST_N_CONST));
    ASSERT_TRUE(std::is_sorted(h_array_gpu.begin(), h_array_gpu.end()));
}

TEST(BitonicSortTest, AlreadySorted) {
    std::vector<float> h_array_gpu(TEST_N_CONST);
    for (int i = 0; i < TEST_N_CONST; ++i) {
        h_array_gpu[i] = (float)i;
    }

    std::vector<float> h_array_cpu = h_array_gpu; // Copy for CPU sort comparison

    bitonic_sort_gpu(h_array_gpu.data(), TEST_N_CONST);
    // h_array_cpu is already sorted

    EXPECT_TRUE(AreArraysEqual(h_array_gpu.data(), h_array_cpu.data(), TEST_N_CONST));
    ASSERT_TRUE(std::is_sorted(h_array_gpu.begin(), h_array_gpu.end()));
}

TEST(BitonicSortTest, ReverseSorted) {
    std::vector<float> h_array_gpu(TEST_N_CONST);
    for (int i = 0; i < TEST_N_CONST; ++i) {
        h_array_gpu[i] = (float)(TEST_N_CONST - 1 - i);
    }
    std::vector<float> h_array_cpu = h_array_gpu;

    bitonic_sort_gpu(h_array_gpu.data(), TEST_N_CONST);
    std::sort(h_array_cpu.begin(), h_array_cpu.end());
    
    EXPECT_TRUE(AreArraysEqual(h_array_gpu.data(), h_array_cpu.data(), TEST_N_CONST));
    ASSERT_TRUE(std::is_sorted(h_array_gpu.begin(), h_array_gpu.end()));
}

TEST(BitonicSortTest, AllSameElements) {
    std::vector<float> h_array_gpu(TEST_N_CONST);
    float val = 42.42f;
    for (int i = 0; i < TEST_N_CONST; ++i) {
        h_array_gpu[i] = val;
    }
    std::vector<float> h_array_cpu = h_array_gpu;

    bitonic_sort_gpu(h_array_gpu.data(), TEST_N_CONST);
    // h_array_cpu is already sorted (all same elements)

    EXPECT_TRUE(AreArraysEqual(h_array_gpu.data(), h_array_cpu.data(), TEST_N_CONST));
    ASSERT_TRUE(std::is_sorted(h_array_gpu.begin(), h_array_gpu.end()));
}

// It's good practice to have a main function for GTest in a separate file or use GTest::Main.
// However, for simplicity in the 100-days structure where each day is self-contained,
// we can include it here if not using gtest_discover_tests with a separate main.
// The CMakeLists.txt will link against GTest::gtest_main which provides a main.
// So, no main() function is needed here.
