# Day 014: Fractional Hausdorff Distance (FHD) for Image Processing

This implementation demonstrates the Fractional Hausdorff Distance (FHD) algorithm applied to grayscale images using CUDA. The FHD is a measure of how similar two sets of points are to each other, commonly used in image processing and computer vision applications.

## Implementation Details

The implementation processes grayscale images by:

1. Converting pixel coordinates and intensities into a 3D space (x, y, z)
2. Computing the FHD using a CUDA kernel that processes chunks of frequency components
3. Generating an output image based on the magnitude of the complex FHD values

## Requirements

- CUDA Toolkit
- OpenCV 4.x
- CMake (for building)

## Building the Code

```bash
mkdir -p build
cd build
cmake ..
make
```

Or directly with NVCC:

```bash
nvcc fhd_image.cu -o fhd_image `pkg-config --cflags --libs opencv4`
```

## Running the Program

```bash
./fhd_image [path_to_grayscale_image]
```

If no image path is provided, the program will attempt to use "lena_gray.png" by default.

## Output

The program generates an output image named "fhd_output.jpg" showing the FHD transformation of the input image.

## Notes

- Any grayscale image can be used as input
- The image dimensions should ideally be multiples of the thread block size (256) for optimal performance
- The implementation uses CUDA Unified Memory for simplicity
