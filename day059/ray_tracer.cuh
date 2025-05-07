#ifndef RAY_TRACER_CUH
#define RAY_TRACER_CUH

#include <cuda_runtime.h>
#include <cmath> // For sqrtf, fmaxf

// Image dimensions and sphere properties
#define WIDTH 1024
#define HEIGHT 768
#define SPHERE_RADIUS 0.5f
#define SPHERE_CENTER_X 0.0f
#define SPHERE_CENTER_Y 0.0f
#define SPHERE_CENTER_Z -1.5f

// Simple CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

struct Vec3 {
    float x, y, z;

    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    __device__ Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __device__ float dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }
    
    __device__ Vec3 normalize() const {
        float len = sqrtf(x*x + y*y + z*z);
        // Avoid division by zero
        if (len == 0.0f) return Vec3(0.0f, 0.0f, 0.0f);
        return Vec3(x/len, y/len, z/len);
    }
};

// CUDA kernel to trace rays
__global__ void render(unsigned char *image);

#endif // RAY_TRACER_CUH
