#include "password_cracker.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip> // For std::setw and std::setfill
#include <chrono> // For timing

// Host-side FNV-1a for verification and generating test cases
unsigned int fnv1a_hash_bytes_host(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]);
        hash = hash * FNV_PRIME;
    }
    return hash;
}

unsigned int generate_target_hash(const std::string& password, int R) {
    unsigned int current_hash = 0;
    unsigned char hash_input_bytes[4];

    // First round
    current_hash = fnv1a_hash_bytes_host(reinterpret_cast<const unsigned char*>(password.c_str()), password.length());

    // Subsequent R-1 rounds
    for (int round = 1; round < R; ++round) {
        hash_input_bytes[0] = (current_hash >> 0) & 0xFF;
        hash_input_bytes[1] = (current_hash >> 8) & 0xFF;
        hash_input_bytes[2] = (current_hash >> 16) & 0xFF;
        hash_input_bytes[3] = (current_hash >> 24) & 0xFF;
        current_hash = fnv1a_hash_bytes_host(hash_input_bytes, 4);
    }
    return current_hash;
}

void run_test(const std::string& expected_password, int R) {
    int password_length = expected_password.length();
    unsigned int target_hash = generate_target_hash(expected_password, R);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Test Case:" << std::endl;
    std::cout << "  Expected Password: \"" << expected_password << "\"" << std::endl;
    std::cout << "  Password Length  : " << password_length << std::endl;
    std::cout << "  Hashing Rounds (R): " << R << std::endl;
    std::cout << "  Target Hash      : " << target_hash << " (0x" << std::hex << target_hash << std::dec << ")" << std::endl;

    char* d_output_password;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_password, (MAX_PW_LEN + 1) * sizeof(char)));
    
    // Time the solve function
    auto start_time = std::chrono::high_resolution_clock::now();
    
    solve(target_hash, password_length, R, d_output_password);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

    char h_output_password[MAX_PW_LEN + 1];
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_password, d_output_password, (password_length + 1) * sizeof(char), cudaMemcpyDeviceToHost));
    h_output_password[password_length] = '\0'; // Ensure null termination

    std::cout << "  Found Password   : \"" << h_output_password << "\"" << std::endl;
    std::cout << "  Execution Time   : " << duration_ms.count() << " ms" << std::endl;

    if (expected_password == h_output_password) {
        std::cout << "  Result: PASSED" << std::endl;
    } else {
        std::cout << "  Result: FAILED" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaFree(d_output_password));
    std::cout << "----------------------------------------" << std::endl << std::endl;
}


int main() {
    // Example 1 from problem description
    // Input:  target_hash = 537089824, password_length = 3, R = 2
    // Output: output_password = "abc"
    // We need to find the password that produces this hash.
    // Let's verify "abc" with R=2 gives 537089824
    // unsigned int hash_abc_r2 = generate_target_hash("abc", 2); // 537089824
    run_test("abc", 2);

    // Example 2 from problem description
    // Input:  target_hash = 440920331, password_length = 3, R = 1
    // Output: output_password = "abc"
    // unsigned int hash_abc_r1 = generate_target_hash("abc", 1); // 440920331
    run_test("abc", 1);

    // Additional test cases
    run_test("a", 1);
    run_test("z", 1);
    run_test("aa", 1);
    run_test("az", 2);
    run_test("zy", 3);
    run_test("cuda", 1); 
    run_test("test", 5);
    
    // A slightly longer password to test performance (length 5)
    // For length 6, it might take a noticeable amount of time.
    // "hello" with R=1. Hash: 2948336902
    run_test("hello", 1); 
    // "world" with R=10
    run_test("world", 10);

    // Test with max length and more rounds (might be slow)
    // run_test("abcdef", 10); // This will be very slow for a simple main test.
                               // Better for dedicated benchmark or longer test suite.
                               // 26^6 is ~308 million.

    std::cout << "All tests completed." << std::endl;

    return 0;
}
