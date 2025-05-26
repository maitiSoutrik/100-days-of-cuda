# 100 Days of CUDA Challenge

[![Deploy to Jetson Nano](https://github.com/maitiSoutrik/100-days-of-cuda/actions/workflows/deploy-to-jetson.yml/badge.svg)](https://github.com/maitiSoutrik/100-days-of-cuda/actions/workflows/deploy-to-jetson.yml)

This repository tracks my journey through the 100 Days of CUDA Challenge. Each day, I'll be coding CUDA kernels and documenting my progress.

## About the Challenge

The 100 Days of CUDA Challenge is about consistently coding CUDA kernels for 100 days without any gaps. The challenge encourages learning and practicing GPU programming using NVIDIA's CUDA platform.

## Resources

- [Programming Massively Parallel Processors (PMPP)](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) - Recommended book for learning CUDA
- [Original Challenge Repository](https://github.com/hkproj/100-days-of-gpu)
- [Reference Repositories](https://github.com/1y33/100Days) and [Hamdi's Repository](https://github.com/a-hamdi/cuda/)

## Hardware

I'll be developing code on my laptop and running it on a Jetson Nano for testing and execution.

## Progress

| Day | Date | Description | Code |
|-----|------|-------------|------|
| 1   | 2025-03-10 | Getting Started with CUDA - Vector Addition | [Link](./day001/) |
| 2   | 2025-03-11 | Matrix Addition in CUDA | [Link](./day002/) |
| 3   | 2025-03-12 | Matrix Multiplication in CUDA | [Link](./day003/) |
| 4   | 2025-03-13 | Parallel Reduction - Partial Sum | [Link](./day004/) |
| 5   | 2025-03-14 | Layer Normalization in CUDA | [Link](./day005/) |
| 6   | 2025-03-15 | Matrix Transpose with CPU/GPU Benchmarking | [Link](./day006/) |
| 7   | 2025-03-16 | 1D & 2D Convolution in CUDA | [Link](./day007/) |
| 8   | 2025-03-17 | Parallel Prefix Sum - Exclusive Scan | [Link](./day008/) |
| 9   | 2025-03-18 | Flash Attention Forward Pass | [Link](./day009/) |
| 10  | 2025-03-19 | Sparse Matrix-Vector Multiplication (SpMV) | [Link](./day010/) |
| 11  | 2025-03-20 | Merge Sort with CUDA | [Link](./day011/) |
| 12  | 2025-03-21 | Breadth-First Search (BFS) with CUDA | [Link](./day012/) |
| 13  | 2025-03-22 | Optimized BFS with Shared Memory | [Link](./day013/) |
| 14  | 2025-03-23 | Fractional Hausdorff Distance (FHD) for Image Processing | [Link](./day014/) |
| 15  | 2025-03-24 | Convolutional Neural Network (CNN) in CUDA | [Link](./day015/) |
| 16  | 2025-03-25 | Parallel Particle System Simulation | [Link](./day016/) |
| 17  | 2025-03-26 | Naive Bayes Classifier Training | [Link](./day017/) |
| 18  | 2025-03-27 | Matrix Multiplication using CUBLAS | [Link](./day018/) |
| 19  | 2025-03-28 | Fast Fourier Transform (FFT) Implementation | [Link](./day019/) |
| 20  | 2025-03-29 | Monte Carlo Option Pricing with CUDA | [Link](./day020/) |
| 21  | 2025-03-30 | Particle Swarm Optimization (PSO) with CUDA | [Link](./day021/) |
| 22  | 2025-03-31 | CUDA-accelerated Reinforcement Learning (Q-Learning) | [Link](./day022/) |
| 23  | 2025-04-01 | Genetic Algorithm Optimization with CUDA | [Link](./day023/) |
| 24  | 2025-04-02 | Gated Linear Unit (GLU) Implementation | [Link](./day024/) |
| 25  | 2025-04-03 | Parallel Point Cloud PassThrough Filter | [Link](./day025/) |
| 26  | 2025-04-04 | Kernel Density Estimation (KDE) | [Link](./day026/) |
| 27  | 2025-04-05 | Mirror Descent (STE) for Quantization | [Link](./day027/) |
| 28  | 2025-04-06 | Mini-Batch SGD for Linear Regression | [Link](./day028/) |
| 29  | 2025-04-07 | K-Means Assignment Step (File Input) | [Link](./day029/) |
| 30  | 2025-04-08 | Headless Camera Processing (Grayscale + Avg Intensity) | [Link](./day030/) |
| 31  | 2025-04-09 | 2D Heat Simulation (Basic vs Shared Memory) | [Link](./day031/) |
| 32  | 2025-04-10 | CUDA Streams for Overlap (Matrix Multiply) | [Link](./day032/) |
| 33  | 2025-04-11 | Parallel Reduction Optimization (Warp Shuffle) | [Link](./day033/) |
| 34  | 2025-04-12 | Point Cloud Voxel Grid Filter (Atomics) | [Link](./day034/) |
| 35  | 2025-04-13 | Kalman Filter Prediction Step (cuBLAS) | [Link](./day035/) | :heavy_check_mark: |
| 36  | 2025-04-14 | SpMV with cuSPARSE          | [Link](./day036/) | :heavy_check_mark: |
| 37  | 2025-04-15 | Simple NN Forward Pass (GEMM + Activation) | [Link](./day037/) | |
| 38  | 2025-04-16 | Batch Normalization Kernel (Forward Pass) | [Link](./day038/) | :heavy_check_mark: |
| 39  | 2025-04-17 | Thrust Library Basics                     | [Link](./day039/) |                    |
| 40  | 2025-04-18 | Image Interpolation (Texture Memory)    | [Link](./day040/) | :heavy_check_mark: |
| 41  | 2025-04-19 | Parallel Radix Sort (Basic Single Pass) | [Link](./day041/) |                    |
| 42  | 2025-04-20 | N-Body Simulation Optimization (Shared Memory) | [Link](./day042/) |                    |
| 43  | 2025-04-21 | Simple cuDNN Convolution (Forward) | [Link](./day043/) |                    |
| 44  | 2025-04-22 | Occupancy Grid Mapping Update | [Link](./day044/) |                    |
| 45  | 2025-04-23 | Optical Flow Gradient Step (Lucas-Kanade) | [Link](./day045/) |                    |
| 46  | 2025-04-24 | Simple Backpropagation Step (Fully Connected Layer) | [Link](./day046/) | :heavy_check_mark: |
| 47  | 2025-04-25 | Dynamic Parallelism (Simple Example) | [Link](./day047/) | :heavy_check_mark: |
| 48  | 2025-04-26 | Parallel AABB Collision Detection | [Link](./day048/) |                    |
| 49  | 2025-04-27 | Mini-Project: Perception Pipeline (Grayscale -> Blur -> Sobel -> Reduction) | [Link](./day049/) | :heavy_check_mark: |
| 50  | 2025-04-28 | Unit Testing CUDA Kernels with Google Test | [Link](./day050/) | :heavy_check_mark: |
| 51  | 2025-04-29 | Exploring TensorRT (Simple ONNX Inference) | [Link](./day051/) |                    |
| 52  | 2025-04-30 | Minimal GRU (minGRU) with Parallel Scan | [Link](./day052/) |                    |
| 53  | 2025-05-01 | Bidirectional LSTM Implementation                     | [Link](./day053/) | :heavy_check_mark: |
| 54  | 2025-05-02 | AdaHessian Optimizer Kernel                       | [Link](./day054/) | :heavy_check_mark: |
| 55  | 2025-05-03 | Quantization Comparison (FP32/FP16/SimFP8)       | [Link](./day055/) |                    |
| 56  | 2025-05-04 | Mish Activation Function Benchmark              | [Link](./day056/) | :heavy_check_mark: |
| 57  | 2025-05-05 | Conjugate Gradient Method (CGM) using cuBLAS    | [Link](./day057/) |                    |
| 58  | 2025-05-06 | Bitonic Sort with Shared Memory Optimization    | [Link](./day058/) |                    |
| 59  | 2025-05-07 | Basic Ray Tracing with CUDA                     | [Link](./day059/) |                    |
| 60  | 2025-05-08 | Muon Optimization - Newton-Schulz Iteration     | [Link](./day060/) |                    |
| 61  | 2025-05-09 | Fisher Information Matrix                       | [Link](./day061/) |                    |
| 62  | 2025-05-10 | Batched Vector L2 Norm (Shared Memory Reduction) | [Link](./day062/) |                    |
| 63  | 2025-05-11 | Parallel Markov Chain Clustering for Robot Localization | [Link](./day063/) | :heavy_check_mark: |
| 64  | 2025-05-12 | Spectral Normalization in GANs (cuBLAS Power Iteration) | [Link](./day064/) |                    |
| 65  | 2025-05-13 | GEGLU Activation Function Implementation | [Link](./day065/) |                    |
| 66  | 2025-05-14 | GPU-Accelerated MFCC Feature Extraction | [Link](./day066/) |                    |
| 67  | 2025-05-15 | SwiGLU Activation and Gradient Computation | [Link](./day067/) |                    |
| 68  | 2025-05-16 | LoRA Implementation and Benchmarking | [Link](./day068/) |                    |
| 69  | 2025-05-17 | Parallel Password Cracking (FNV-1a) | [Link](./day069/) |                    |
| 70  | 2025-05-18 | Mean Squared Error (MSE) Calculation | [Link](./day070/) |                    |
| 71  | 2025-05-19 | Group Normalization Forward Pass | [Link](./day071/) |                    |
| 72  | 2025-05-20 | Total Variation Distance (TVD) Loss | [Link](./day072/) |                    |
| 73  | 2025-05-21 | 1D Rotary Positional Embedding (RoPE) | [Link](./day073/) | :heavy_check_mark: |
| 74  | 2025-05-22 | 2D Rotary Positional Embeddings (RoPE-2D) in CUDA | [Link](./day074/) |                    |
| 75  | 2025-05-23 | Fused Linear Transformation and Softmax Cross-Entropy Loss | [Link](./day075/) | :heavy_check_mark: |
| 76  | 2025-05-24 | Contrastive Loss (Forward & Backward) | [Link](./day076/) |                    |
| 77  | 2025-05-25 | Huber Loss Implementation in CUDA | [Link](./day077/) |                    |
| 78  | 2025-05-26 | Dynamic Tanh (DyT) Operation | [Link](./day078/) |                    |

## Rules

1. Code CUDA kernels consistently for 100 days without any gaps
2. Document what I did each day
3. Every 10 days, I can claim a badge from the challenge
4. No code, no badge, no challenge
