# Day 15: Convolutional Neural Network (CNN) in CUDA

This implementation focuses on creating a Convolutional Neural Network (CNN) using CUDA. The code includes both forward and backward passes with pooling layers, and utilizes the unrolling trick for improved performance.

## What is a CNN?

A Convolutional Neural Network is a specialized neural network architecture designed primarily for processing grid-like data, such as images. CNNs are particularly effective at detecting spatial hierarchies and patterns in data.

## Core Components

1. **Convolutional Layers**: Apply filters to input data to detect features
2. **Activation Functions**: Introduce non-linearity (typically ReLU)
3. **Pooling Layers**: Reduce spatial dimensions and provide translation invariance
4. **Fully Connected Layers**: Connect extracted features for classification

## Implementation Details

This CUDA implementation includes:

- Forward pass for convolution, activation, and pooling operations
- Backward pass (backpropagation) for training
- Optimization using the im2col (unrolling) technique for efficient matrix operations
- Parallel execution on GPU for improved performance

## Performance Considerations

- Memory coalescing for efficient memory access
- Shared memory usage for frequently accessed data
- Proper thread organization for maximum parallelism
- Balance between parallelism and resource usage

## Usage

Compile the code with:

```bash
cmake -B build && cmake --build build
```

Run the executable:

```bash
./build/cnn
```
