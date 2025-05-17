#include "gtest/gtest.h"
#include "password_cracker.cuh"
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Host-side FNV-1a for generating test cases (can be refactored if already in main.cu and linked)
// For simplicity in a test file, sometimes it's easier to redefine or ensure linkage.
// Assuming it's accessible or we redefine it here for test self-containment.
unsigned int fnv1a_hash_bytes_host_for_test(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]);
        hash = hash * FNV_PRIME;
    }
    return hash;
}

unsigned int generate_target_hash_for_test(const std::string& password, int R) {
    unsigned int current_hash = 0;
    unsigned char hash_input_bytes[4];

    current_hash = fnv1a_hash_bytes_host_for_test(reinterpret_cast<const unsigned char*>(password.c_str()), password.length());

    for (int round = 1; round < R; ++round) {
        hash_input_bytes[0] = (current_hash >> 0) & 0xFF;
        hash_input_bytes[1] = (current_hash >> 8) & 0xFF;
        hash_input_bytes[2] = (current_hash >> 16) & 0xFF;
        hash_input_bytes[3] = (current_hash >> 24) & 0xFF;
        current_hash = fnv1a_hash_bytes_host_for_test(hash_input_bytes, 4);
    }
    return current_hash;
}

struct TestParams {
    std::string password;
    int R;
};

class PasswordCrackerTest : public ::testing::TestWithParam<TestParams> {
protected:
    char* d_output_password;
    char h_output_password[MAX_PW_LEN + 1];

    void SetUp() override {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_password, (MAX_PW_LEN + 1) * sizeof(char)));
    }

    void TearDown() override {
        CHECK_CUDA_ERROR(cudaFree(d_output_password));
    }
};

TEST_P(PasswordCrackerTest, FindsCorrectPassword) {
    TestParams params = GetParam();
    std::string expected_password = params.password;
    int R = params.R;
    int password_length = expected_password.length();
    unsigned int target_hash = generate_target_hash_for_test(expected_password, R);

    ASSERT_LE(password_length, MAX_PW_LEN) << "Test password exceeds MAX_PW_LEN";

    solve(target_hash, password_length, R, d_output_password);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_password, d_output_password, (password_length + 1) * sizeof(char), cudaMemcpyDeviceToHost));
    h_output_password[password_length] = '\0'; // Ensure null termination for comparison

    EXPECT_STREQ(expected_password.c_str(), h_output_password);
}

INSTANTIATE_TEST_SUITE_P(
    CrackerTests,
    PasswordCrackerTest,
    ::testing::Values(
        TestParams{"a", 1},
        TestParams{"b", 1},
        TestParams{"z", 1},
        TestParams{"aa", 1},
        TestParams{"ab", 1},
        TestParams{"az", 1},
        TestParams{"ba", 1},
        TestParams{"zz", 1},
        TestParams{"abc", 1},
        TestParams{"abc", 2}, // Example 1 from problem
        TestParams{"xyz", 3},
        TestParams{"cuda", 1},
        TestParams{"test", 5},
        TestParams{"fnv", 10},
        TestParams{"leet", 20}
        // Add more test cases as needed.
        // Be mindful of test duration for longer passwords or high R values.
        // TestParams{"abcdef", 1} // Length 6, R=1
        // TestParams{"qwerty", 5} // Length 6, R=5
    )
);

// It might be useful to have a standalone test for the FNV hash device function itself,
// but that would require launching a kernel just to call it.
// For now, the end-to-end test via solve() covers its usage.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
