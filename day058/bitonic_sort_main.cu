#include "bitonic_sort.cuh"
#include <stdio.h>
#include <stdlib.h> // For rand(), srand()
#include <time.h>   // For time()

// N_CONST is defined in bitonic_sort.cu, but we need it here for array declaration
// Alternatively, it could be defined in the .cuh file if it's meant to be globally visible
// For this project structure, it's better to have it consistently available.
// Let's assume N_CONST from bitonic_sort.cu is 1024.
// If N_CONST were to change, this file would also need an update or a common definition.
// For simplicity in this example, we'll use a local constant that matches.
const int MAIN_N_CONST = 1024; 

int main() {
    float h_array[MAIN_N_CONST];

    // Initialize random seed
    srand(time(NULL));

    // Fill array with random numbers
    printf("Generating %d random float numbers...\n", MAIN_N_CONST);
    for (int i = 0; i < MAIN_N_CONST; i++) {
        h_array[i] = (float)(rand() % 10000) / 100.0f; // Random floats between 0.00 and 99.99
    }

    printf("Unsorted array (first 20 elements and last 10 elements if N > 30):\n");
    if (MAIN_N_CONST <= 30) {
        print_array_host(h_array, MAIN_N_CONST);
    } else {
        print_array_host(h_array, 20);
        printf("...\n");
        // Print last 10 elements
        for (int i = MAIN_N_CONST - 10; i < MAIN_N_CONST; ++i) {
            printf("%f ", h_array[i]);
        }
        printf("\n\n");
    }


    printf("Starting Bitonic Sort on GPU...\n");
    bitonic_sort_gpu(h_array, MAIN_N_CONST);
    printf("Bitonic Sort on GPU finished.\n\n");

    printf("Sorted array (first 20 elements and last 10 elements if N > 30):\n");
    if (MAIN_N_CONST <= 30) {
        print_array_host(h_array, MAIN_N_CONST);
    } else {
        print_array_host(h_array, 20);
        printf("...\n");
        // Print last 10 elements
        for (int i = MAIN_N_CONST - 10; i < MAIN_N_CONST; ++i) {
            printf("%f ", h_array[i]);
        }
        printf("\n");
    }
    
    // Verification (simple check)
    bool sorted_correctly = true;
    for (int i = 0; i < MAIN_N_CONST - 1; i++) {
        if (h_array[i] > h_array[i+1]) {
            sorted_correctly = false;
            printf("\nError: Array not sorted correctly at index %d and %d! Values: %f, %f\n", i, i+1, h_array[i], h_array[i+1]);
            break;
        }
    }

    if (sorted_correctly) {
        printf("\nVerification: Array is sorted correctly.\n");
    } else {
        printf("\nVerification: Array is NOT sorted correctly.\n");
    }

    return 0;
}
