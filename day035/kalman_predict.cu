#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip> // For std::setprecision

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- Error Checking Macros ---
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << err << " \"" << func << "\" (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS_ERROR(val) checkCublasError((val), #val, __FILE__, __LINE__)
inline void checkCublasError(cublasStatus_t err, const char* const func, const char* const file, const int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Error at " << file << ":" << line << " code=" << err << " \"" << func << "\"" << std::endl;
        // Add mapping from cublasStatus_t to string if desired
        exit(EXIT_FAILURE);
    }
}

// --- Kalman Filter Parameters (Example for a simple constant velocity model using accel data) ---
// Adjust these based on the actual IMU data structure and desired model
const int STATE_DIM = 4; // e.g., [x, y, vx, vy]
const int MEASUREMENT_DIM = 2; // Using [a_RS_S_x, a_RS_S_y] as proxy measurements
const int CONTROL_DIM = 0; // Assuming no control input for this example

// Indices for relevant columns in the CSV (0-based)
const int ACCEL_X_COL = 4;
const int ACCEL_Y_COL = 5;


// --- Function Declarations ---

// CPU Implementation
void kalmanPredictCPU(const std::vector<float>& x_in, const std::vector<float>& P_in,
                      const std::vector<float>& F_mat, const std::vector<float>& Q_mat,
                      std::vector<float>& x_out, std::vector<float>& P_out);
// (Add B and u if CONTROL_DIM > 0)

// GPU Implementation (using cuBLAS)
void kalmanPredictGPU(cublasHandle_t handle,
                      const float* d_x_in, const float* d_P_in,
                      const float* d_F_mat, const float* d_Q_mat,
                      float* d_x_out, float* d_P_out,
                      int num_states); // For potential batching (start with 1)

// Data Loading
bool loadIMUData(const std::string& filename, std::vector<std::vector<float>>& data); // Placeholder

// Matrix Utilities (Optional, for CPU version or verification)
void matrixMultiplyCPU(const float* A, const float* B, float* C, int M, int N, int K);
void matrixAddCPU(const float* A, const float* B, float* C, int M, int N);
void matrixTransposeCPU(const float* A, float* At, int M, int N);
void printMatrix(const char* name, const float* mat, int rows, int cols);

// --- Main Function ---
int main(int argc, char *argv[]) {
    std::cout << "Day 35: Kalman Filter Prediction Step using cuBLAS" << std::endl;

    // --- Parameters & Command Line Argument Parsing ---
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_imu_data.csv>" << std::endl;
        return 1;
    }
    const std::string data_filename = argv[1];
    int num_data_points = 0; // Determined by loaded data

    // --- Load Data ---
    std::vector<std::vector<float>> imu_data; // Stores MEASUREMENT_DIM relevant columns
    std::cout << "Attempting to load data from: " << data_filename << std::endl;
    if (!loadIMUData(data_filename, imu_data)) {
        std::cerr << "Error: Could not load IMU data from " << data_filename << std::endl;
        // Optionally: fallback to placeholder data if needed for testing without a file
        // For now, we require the file.
        return 1;
        // Example placeholder generation (if desired):
        // std::cerr << "Warning: Using placeholder data." << std::endl;
        // num_data_points = 100;
        // imu_data.resize(num_data_points, std::vector<float>(MEASUREMENT_DIM, 0.0f));
        for(int i = 0; i < num_data_points; ++i) {
             for (int j = 0; j < MEASUREMENT_DIM; ++j) {
                 imu_data[i][j] = static_cast<float>(i + j); // Simple placeholder
             }
        }
    } else {
        num_data_points = imu_data.size();
        std::cout << "Successfully loaded " << num_data_points << " data points." << std::endl;
    }
    if (num_data_points == 0) {
        std::cerr << "Error: No data points to process." << std::endl;
        return 1;
    }

    // --- Initialize Kalman Filter State (Example) ---
    // Note: Using acceleration directly isn't standard for a pos/vel state.
    // A proper implementation would integrate acceleration in the model (F, B matrices)
    // or use a different state like [x, y, vx, vy, ax, ay].
    // For this example, we'll just initialize position based on the first accel 'measurement'
    // as a placeholder, acknowledging this simplification.
    std::vector<float> h_x(STATE_DIM); // Host initial state vector
    h_x[0] = imu_data[0][0]; // Initial x pos = first accel_x reading (simplification)
    h_x[1] = imu_data[0][1]; // Initial y pos = first accel_y reading (simplification)
    h_x[2] = 0.0f;           // Initial vx = 0
    h_x[3] = 0.0f;           // Initial vy = 0

    std::vector<float> h_P(STATE_DIM * STATE_DIM); // Host initial covariance matrix (Identity * large value)
    for (int i = 0; i < STATE_DIM; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            h_P[i * STATE_DIM + j] = (i == j) ? 1000.0f : 0.0f;
        }
    }

    // --- Define Kalman Matrices (Constant Velocity Model Example) ---
    float dt = 0.1f; // Assume time step (needs to be derived from data if possible)
    std::vector<float> h_F(STATE_DIM * STATE_DIM); // State Transition Matrix F
    // F = [1, 0, dt, 0]
    //     [0, 1, 0, dt]
    //     [0, 0, 1,  0]
    //     [0, 0, 0,  1]
    for(int i=0; i<STATE_DIM*STATE_DIM; ++i) h_F[i] = 0.0f;
    h_F[0*STATE_DIM + 0] = 1.0f; h_F[0*STATE_DIM + 2] = dt;
    h_F[1*STATE_DIM + 1] = 1.0f; h_F[1*STATE_DIM + 3] = dt;
    h_F[2*STATE_DIM + 2] = 1.0f;
    h_F[3*STATE_DIM + 3] = 1.0f;

    std::vector<float> h_Q(STATE_DIM * STATE_DIM); // Process Noise Covariance Q
    // Simplified Q: assumes noise affects acceleration -> integrates to velocity/position
    // Tune this matrix based on expected system noise
    float noise_accel = 0.1f;
    float q1 = pow(dt, 4) / 4.0 * noise_accel;
    float q2 = pow(dt, 3) / 2.0 * noise_accel;
    float q3 = pow(dt, 2) * noise_accel;
    // Q = [q1, 0,  q2, 0 ]
    //     [0,  q1, 0,  q2]
    //     [q2, 0,  q3, 0 ]
    //     [0,  q2, 0,  q3]
     for(int i=0; i<STATE_DIM*STATE_DIM; ++i) h_Q[i] = 0.0f;
     h_Q[0*STATE_DIM + 0] = q1; h_Q[0*STATE_DIM + 2] = q2;
     h_Q[1*STATE_DIM + 1] = q1; h_Q[1*STATE_DIM + 3] = q2;
     h_Q[2*STATE_DIM + 0] = q2; h_Q[2*STATE_DIM + 2] = q3;
     h_Q[3*STATE_DIM + 1] = q2; h_Q[3*STATE_DIM + 3] = q3;


    // --- CPU Execution ---
    std::cout << "\n--- Running CPU Kalman Prediction ---" << std::endl;
    std::vector<float> h_x_cpu_pred(STATE_DIM);
    std::vector<float> h_P_cpu_pred(STATE_DIM * STATE_DIM);
    std::vector<float> h_x_cpu_current = h_x; // Start with initial state
    std::vector<float> h_P_cpu_current = h_P; // Start with initial covariance

    auto start_cpu = std::chrono::high_resolution_clock::now();
    // In a real filter, this runs in a loop over measurements
    // Here, just run the prediction step once as an example
    kalmanPredictCPU(h_x_cpu_current, h_P_cpu_current, h_F, h_Q, h_x_cpu_pred, h_P_cpu_pred);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    std::cout << "CPU Prediction Step Duration: " << cpu_duration.count() << " ms" << std::endl;
    printMatrix("CPU Predicted State (x_pred)", h_x_cpu_pred.data(), STATE_DIM, 1);
    // printMatrix("CPU Predicted Covariance (P_pred)", h_P_cpu_pred.data(), STATE_DIM, STATE_DIM); // Can be large

    // --- GPU Execution ---
    std::cout << "\n--- Running GPU Kalman Prediction (cuBLAS) ---" << std::endl;

    // Initialize cuBLAS
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

    // Allocate device memory
    float *d_x_in, *d_P_in, *d_F, *d_Q, *d_x_out, *d_P_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_in, STATE_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_P_in, STATE_DIM * STATE_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_F, STATE_DIM * STATE_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Q, STATE_DIM * STATE_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_out, STATE_DIM * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_P_out, STATE_DIM * STATE_DIM * sizeof(float)));

    // Transfer data to device (use initial state for this example)
    CHECK_CUDA_ERROR(cudaMemcpy(d_x_in, h_x.data(), STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_P_in, h_P.data(), STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_F, h_F.data(), STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q.data(), STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    // Warm-up GPU (optional, but good practice for timing)
    kalmanPredictGPU(cublasHandle, d_x_in, d_P_in, d_F, d_Q, d_x_out, d_P_out, 1);

    // Record start event
    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu, 0));

    // Execute GPU Kalman Prediction (single step for now)
    kalmanPredictGPU(cublasHandle, d_x_in, d_P_in, d_F, d_Q, d_x_out, d_P_out, 1);

    // Record stop event and synchronize
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    // Calculate elapsed time
    float gpu_duration_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu));

    std::cout << "GPU Prediction Step Duration: " << gpu_duration_ms << " ms" << std::endl;

    // Transfer results back to host
    std::vector<float> h_x_gpu_pred(STATE_DIM);
    std::vector<float> h_P_gpu_pred(STATE_DIM * STATE_DIM);
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_gpu_pred.data(), d_x_out, STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_P_gpu_pred.data(), d_P_out, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    printMatrix("GPU Predicted State (x_pred)", h_x_gpu_pred.data(), STATE_DIM, 1);
    // printMatrix("GPU Predicted Covariance (P_pred)", h_P_gpu_pred.data(), STATE_DIM, STATE_DIM);

    // --- Verification (Compare CPU and GPU results) ---
    std::cout << "\n--- Verification ---" << std::endl;
    double diff_x = 0.0, diff_P = 0.0;
    for (int i = 0; i < STATE_DIM; ++i) {
        diff_x += std::abs(h_x_cpu_pred[i] - h_x_gpu_pred[i]);
    }
    for (int i = 0; i < STATE_DIM * STATE_DIM; ++i) {
        diff_P += std::abs(h_P_cpu_pred[i] - h_P_gpu_pred[i]);
    }
    std::cout << "Sum absolute difference (State x): " << diff_x << std::endl;
    std::cout << "Sum absolute difference (Covariance P): " << diff_P << std::endl;

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_x_in));
    CHECK_CUDA_ERROR(cudaFree(d_P_in));
    CHECK_CUDA_ERROR(cudaFree(d_F));
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_x_out));
    CHECK_CUDA_ERROR(cudaFree(d_P_out));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

    std::cout << "\nFinished Day 35." << std::endl;
    return 0;
}


// --- Function Implementations ---

bool loadIMUData(const std::string& filename, std::vector<std::vector<float>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: Could not read header line or file is empty." << std::endl;
        file.close();
        return false;
    }
    if (line.empty() || line[0] != '#') {
        std::cerr << "Warning: First line doesn't look like a header comment: " << line << std::endl;
        // Optional: Rewind or re-open if the first line might be data
    }

    int line_num = 1; // Start counting after header
    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> full_row; // Store all parsed floats from the row

        while (std::getline(ss, cell, ',')) {
            try {
                full_row.push_back(std::stof(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Line " << line_num << ": Could not parse value '" << cell << "'. Skipping value, using 0.0." << std::endl;
                full_row.push_back(0.0f);
            } catch (const std::out_of_range& e) {
                std::cerr << "Warning: Line " << line_num << ": Value '" << cell << "' out of range. Skipping value, using 0.0." << std::endl;
                full_row.push_back(0.0f);
            }
        }

        // Check if we have enough columns for accel_x and accel_y
        if (full_row.size() > ACCEL_X_COL && full_row.size() > ACCEL_Y_COL) {
            std::vector<float> measurement_row(MEASUREMENT_DIM);
            measurement_row[0] = full_row[ACCEL_X_COL]; // a_RS_S_x
            measurement_row[1] = full_row[ACCEL_Y_COL]; // a_RS_S_y
            data.push_back(measurement_row);
        } else {
            std::cerr << "Warning: Line " << line_num << ": Skipped row due to insufficient columns (need at least "
                      << std::max(ACCEL_X_COL, ACCEL_Y_COL) + 1 << "). Line: " << line << std::endl;
        }
    }

    file.close();
    if (data.empty()) {
         std::cerr << "Error: No valid data rows parsed from the file." << std::endl;
         return false;
    }
    return true; // Return true if at least one valid row was loaded
}


// Simple CPU Matrix Multiplication (Row-major)
void matrixMultiplyCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < N; ++l) {
                sum += A[i * N + l] * B[l * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Simple CPU Matrix Addition (Row-major)
void matrixAddCPU(const float* A, const float* B, float* C, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}

// Simple CPU Matrix Transpose (Row-major)
void matrixTransposeCPU(const float* A, float* At, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            At[j * M + i] = A[i * N + j];
        }
    }
}

void printMatrix(const char* name, const float* mat, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    if (!mat) {
        std::cout << "  [Null Pointer]" << std::endl;
        return;
    }
     std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << (j == cols - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
     std::cout << std::defaultfloat << std::setprecision(6); // Reset precision
}


// CPU Kalman Prediction Step:
// x_pred = F * x_in
// P_pred = F * P_in * F^T + Q
void kalmanPredictCPU(const std::vector<float>& x_in, const std::vector<float>& P_in,
                      const std::vector<float>& F_mat, const std::vector<float>& Q_mat,
                      std::vector<float>& x_out, std::vector<float>& P_out)
{
    if(x_in.size() != STATE_DIM || P_in.size() != STATE_DIM*STATE_DIM ||
       F_mat.size() != STATE_DIM*STATE_DIM || Q_mat.size() != STATE_DIM*STATE_DIM ||
       x_out.size() != STATE_DIM || P_out.size() != STATE_DIM*STATE_DIM) {
        std::cerr << "CPU Kalman Predict: Dimension mismatch!" << std::endl;
        return;
    }

    // 1. Predict state: x_pred = F * x_in
    // F (STATE_DIM x STATE_DIM), x_in (STATE_DIM x 1) -> x_out (STATE_DIM x 1)
    matrixMultiplyCPU(F_mat.data(), x_in.data(), x_out.data(), STATE_DIM, STATE_DIM, 1);

    // 2. Predict covariance: P_pred = F * P_in * F^T + Q
    // Need intermediate matrices
    std::vector<float> P_F_T(STATE_DIM * STATE_DIM); // F * P_in
    std::vector<float> F_T(STATE_DIM * STATE_DIM);   // F^T

    // Calculate F^T
    matrixTransposeCPU(F_mat.data(), F_T.data(), STATE_DIM, STATE_DIM);

    // Calculate F * P_in
    // F (STATE_DIM x STATE_DIM), P_in (STATE_DIM x STATE_DIM) -> P_F_T (STATE_DIM x STATE_DIM)
    matrixMultiplyCPU(F_mat.data(), P_in.data(), P_F_T.data(), STATE_DIM, STATE_DIM, STATE_DIM);

    // Calculate (F * P_in) * F^T = P_F_T * F_T
    // P_F_T (STATE_DIM x STATE_DIM), F_T (STATE_DIM x STATE_DIM) -> P_out (STATE_DIM x STATE_DIM)
    matrixMultiplyCPU(P_F_T.data(), F_T.data(), P_out.data(), STATE_DIM, STATE_DIM, STATE_DIM);

    // Add Process Noise: P_pred = (F * P_in * F^T) + Q
    matrixAddCPU(P_out.data(), Q_mat.data(), P_out.data(), STATE_DIM, STATE_DIM);
}


// GPU Kalman Prediction Step (using cuBLAS):
// x_pred = F * x_in
// P_pred = F * P_in * F^T + Q
// Note: cuBLAS uses column-major order by default. We are using row-major in CPU,
// so we need to be careful with matrix dimensions and transpose flags in cuBLAS calls,
// OR transpose matrices before calling cuBLAS.
// Let's stick to the formula and manage row/column major carefully.
// cuBLAS expects pointers to device memory.
void kalmanPredictGPU(cublasHandle_t handle,
                      const float* d_x_in, const float* d_P_in,
                      const float* d_F_mat, const float* d_Q_mat,
                      float* d_x_out, float* d_P_out,
                      int num_states) // num_states = 1 for now
{
    if (num_states != 1) {
         std::cerr << "GPU Kalman Predict: Batching not yet implemented." << std::endl;
         return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int N = STATE_DIM; // Dimension for square matrices

    // 1. Predict state: x_pred = F * x_in
    // Operation: d_x_out = alpha * F * d_x_in + beta * d_x_out
    // F (NxN), d_x_in (Nx1), d_x_out (Nx1)
    // cublasSgemv (Matrix-Vector multiply)
    // Assuming F and x are in row-major, we treat F as col-major by swapping dims and using CUBLAS_OP_N?
    // Or, we treat the vector x as a Nx1 matrix and use Sgemm. Let's try Sgemm.
    // C = alpha*op(A)*op(B) + beta*C
    // d_x_out = 1.0 * F * d_x_in + 0.0 * d_x_out
    // A=F (NxN), B=d_x_in (Nx1), C=d_x_out (Nx1)
    // M=N, N=1, K=N
    // Since our matrices (F) are row-major, and cuBLAS expects column-major,
    // performing F(row)*x(col) is equivalent to x^T(row) * F^T(row) in column-major? No.
    // Easiest way: Provide matrices to cuBLAS as if they were column-major,
    // but swap M and N in the call, and use CUBLAS_OP_T for both A and B? Let's verify.
    // If A, B, C are row-major, C = A * B becomes C^T = B^T * A^T in column-major.
    // So, for x_out = F * x_in (all row-major):
    // x_out^T (1xN) = x_in^T (1xN) * F^T (NxN) in column-major.
    // Let's try calling sgemm as if inputs were column-major, interpreting our row-major data.
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC)
    // A=F(col-major -> our row-major), B=x_in(col-major -> our row-major vector), C=x_out(...)
    // M=N (rows of F), N=1 (cols of x_in), K=N (cols of F / rows of x_in)
    // LDA=N (leading dim of F), LDB=N (leading dim of x_in - treated as Nx1 matrix), LDC=N (leading dim of x_out)
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, 1, N,             // M, N, K
                                   &alpha,
                                   d_F_mat, N,          // A, LDA
                                   d_x_in, N,           // B, LDB (treat x_in as Nx1)
                                   &beta,
                                   d_x_out, N));        // C, LDC


    // 2. Predict covariance: P_pred = F * P_in * F^T + Q
    // Step 2a: Calculate Temp = F * P_in
    // Temp = 1.0 * F * P_in + 0.0 * Temp
    // A=F (NxN), B=P_in (NxN), C=Temp (NxN)
    // M=N, N=N, K=N
    float* d_Temp; // Temporary matrix for F * P_in
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Temp, N * N * sizeof(float)));
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, N, N,             // M, N, K
                                   &alpha,
                                   d_F_mat, N,          // A=F, LDA
                                   d_P_in, N,           // B=P_in, LDB
                                   &beta,
                                   d_Temp, N));         // C=Temp, LDC

    // Step 2b: Calculate P_pred_noQ = Temp * F^T
    // P_pred = 1.0 * Temp * F^T + 0.0 * P_pred
    // A=Temp (NxN), B=F (NxN) needs transpose, C=P_out (NxN)
    // M=N, N=N, K=N
    // Use CUBLAS_OP_T for F (B matrix)
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                   N, N, N,             // M, N, K
                                   &alpha,
                                   d_Temp, N,           // A=Temp, LDA
                                   d_F_mat, N,          // B=F, LDB (use Transpose)
                                   &beta,
                                   d_P_out, N));        // C=P_out, LDC

    // Step 2c: Add Q: P_pred = P_pred + 1.0 * Q
    // Use cublasSaxpy for vectorized addition, or geam for matrix addition.
    // cublasSgeam (General Matrix Addition/Transpose)
    // C = alpha*op(A) + beta*op(B)
    // P_out = 1.0 * P_out + 1.0 * Q
    // opA = N, opB = N
    CHECK_CUBLAS_ERROR(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, N,                // M, N
                                   &alpha,              // alpha = 1.0
                                   d_P_out, N,          // A = current P_out, LDA
                                   &alpha,              // beta = 1.0 (using alpha here)
                                   d_Q_mat, N,          // B = Q, LDB
                                   d_P_out, N));        // C = P_out, LDC


    // Free temporary matrix
    CHECK_CUDA_ERROR(cudaFree(d_Temp));
}
