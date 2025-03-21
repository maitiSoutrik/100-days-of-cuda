#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Device function to find the co-rank of two sorted arrays
__device__ void co_rank(const int* A, const int* B, int k, const int N, const int M, int* i_out, int* j_out) {
    int low = max(0, k-M);
    int high = min(k, N);
    
    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;
        
        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > M) {
            low = i + 1;
            continue;
        }

        if (i > 0 && j < M && A[i-1] > B[j]) {
            high = i - 1;
        }
        else if (j > 0 && i < N && B[j-1] > A[i]) {
            low = i + 1;
        }
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

// Kernel for parallel merge of two sorted arrays
__global__ void parallel_merge(const int* A, const int* B, int* C, const int N, const int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N + M) {
        int i, j;
        co_rank(A, B, tid, N, M, &i, &j);
        
        if (j >= M || (i < N && A[i] <= B[j])) {
            C[tid] = A[i];
        } else {
            C[tid] = B[j];
        }
    }
}

// Host function to merge two sorted arrays
void mergeArrays(int* A, int N, int* B, int M, int* C) {
    int *d_A, *d_B, *d_C;
    
    // Allocate memory on device
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, M * sizeof(int));
    cudaMalloc(&d_C, (N+M) * sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * sizeof(int), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(256);
    dim3 grid((N+M + block.x-1) / block.x);
    
    // Launch kernel
    parallel_merge<<<grid, block>>>(d_A, d_B, d_C, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, d_C, (N+M) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Sequential merge sort implementation for CPU comparison
void sequentialMergeSort(int* arr, int* temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        // Sort first and second halves
        sequentialMergeSort(arr, temp, left, mid);
        sequentialMergeSort(arr, temp, mid + 1, right);
        
        // Merge the sorted halves
        int i = left;
        int j = mid + 1;
        int k = left;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }
        
        // Copy remaining elements
        while (i <= mid) {
            temp[k++] = arr[i++];
        }
        
        while (j <= right) {
            temp[k++] = arr[j++];
        }
        
        // Copy back to original array
        for (i = left; i <= right; i++) {
            arr[i] = temp[i];
        }
    }
}

// Recursive function to perform merge sort using CUDA
void parallelMergeSort(int* arr, int* temp, int n) {
    // Base case: if array size is 1 or less, it's already sorted
    if (n <= 1) return;
    
    int mid = n / 2;
    
    // Recursively sort the two halves
    parallelMergeSort(arr, temp, mid);
    parallelMergeSort(arr + mid, temp + mid, n - mid);
    
    // Merge the sorted halves using CUDA
    mergeArrays(arr, mid, arr + mid, n - mid, temp);
    
    // Copy the merged result back to the original array
    for (int i = 0; i < n; i++) {
        arr[i] = temp[i];
    }
}

// Function to check if array is sorted
bool isSorted(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

// Function to print array
void printArray(int* arr, int n, const char* label) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    // Set random seed
    srand(time(NULL));
    
    // Array size
    const int N = 1024;
    
    // Allocate memory for arrays
    int* arr1 = (int*)malloc(N * sizeof(int));
    int* arr2 = (int*)malloc(N * sizeof(int));
    int* temp1 = (int*)malloc(N * sizeof(int));
    int* temp2 = (int*)malloc(N * sizeof(int));
    
    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        arr1[i] = rand() % 10000;
        arr2[i] = arr1[i]; // Copy for sequential sort
    }
    
    // Print first few elements of unsorted array
    printf("Array size: %d\n", N);
    printf("First 10 elements of unsorted array:\n");
    for (int i = 0; i < min(10, N); i++) {
        printf("%d ", arr1[i]);
    }
    printf("\n\n");
    
    // Time the parallel merge sort
    clock_t start = clock();
    parallelMergeSort(arr1, temp1, N);
    clock_t end = clock();
    double parallel_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Time the sequential merge sort
    start = clock();
    sequentialMergeSort(arr2, temp2, 0, N - 1);
    end = clock();
    double sequential_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Print first few elements of sorted arrays
    printf("First 10 elements after parallel merge sort:\n");
    for (int i = 0; i < min(10, N); i++) {
        printf("%d ", arr1[i]);
    }
    printf("\n\n");
    
    printf("First 10 elements after sequential merge sort:\n");
    for (int i = 0; i < min(10, N); i++) {
        printf("%d ", arr2[i]);
    }
    printf("\n\n");
    
    // Verify sorting
    printf("Parallel merge sort %s\n", isSorted(arr1, N) ? "successful" : "failed");
    printf("Sequential merge sort %s\n", isSorted(arr2, N) ? "successful" : "failed");
    
    // Print timing results
    printf("Parallel merge sort time: %f seconds\n", parallel_time);
    printf("Sequential merge sort time: %f seconds\n", sequential_time);
    printf("Speedup: %f\n", sequential_time / parallel_time);
    
    // Free memory
    free(arr1);
    free(arr2);
    free(temp1);
    free(temp2);
    
    return 0;
}
