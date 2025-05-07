#include "ray_tracer.cuh"
#include <stdio.h> // For kernel printf if needed for debugging, not for CHECK_CUDA_ERROR

// CUDA kernel to trace rays
__global__ void render(unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = (y * WIDTH + x) * 3; // Each pixel has 3 color channels (RGB)

    // Convert pixel to normalized screen space (-1 to 1)
    // Aspect ratio correction can be added here if needed
    float aspect_ratio = (float)WIDTH / (float)HEIGHT;
    float u = (2.0f * (x + 0.5f) / WIDTH - 1.0f) * aspect_ratio; // Centering pixels
    float v = (1.0f - 2.0f * (y + 0.5f) / HEIGHT); // Flipped Y, centering pixels

    // Ray origin (camera at (0,0,0))
    Vec3 ray_origin(0.0f, 0.0f, 0.0f);
    // Ray direction (from camera to screen)
    Vec3 ray_dir(u, v, -1.0f); // Assuming camera looks along -Z
    ray_dir = ray_dir.normalize();

    // Sphere properties
    Vec3 sphere_center(SPHERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_CENTER_Z);
    
    // Compute ray-sphere intersection (quadratic equation)
    Vec3 oc = ray_origin - sphere_center;
    float a = ray_dir.dot(ray_dir); // Should be 1.0 if ray_dir is normalized
    float b_coeff = 2.0f * oc.dot(ray_dir);
    float c_coeff = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    float discriminant = b_coeff * b_coeff - 4 * a * c_coeff;

    if (discriminant >= 0) {
        // Compute the nearest intersection (smallest positive t)
        float t0 = (-b_coeff - sqrtf(discriminant)) / (2.0f * a);
        float t1 = (-b_coeff + sqrtf(discriminant)) / (2.0f * a);
        
        float t = -1.0f;
        if (t0 > 0.0f && t1 > 0.0f) {
            t = fminf(t0, t1);
        } else if (t0 > 0.0f) {
            t = t0;
        } else if (t1 > 0.0f) {
            t = t1;
        }

        if (t > 0.0f) { // Check if intersection is in front of the camera
            Vec3 hit_point = ray_origin + ray_dir * t;
            Vec3 normal = (hit_point - sphere_center).normalize();

            // Simple diffuse shading based on light direction
            Vec3 light_dir(1.0f, 1.0f, 1.0f); // Light from top-right-front
            light_dir = light_dir.normalize();
            float intensity = fmaxf(0.0f, normal.dot(light_dir));

            // Color the sphere (red tone)
            image[idx]     = (unsigned char)(200 * intensity + 55 * (1.0f - intensity)); // Mix with ambient
            image[idx + 1] = (unsigned char)(50 * intensity  + 20 * (1.0f - intensity));
            image[idx + 2] = (unsigned char)(50 * intensity  + 20 * (1.0f - intensity));
        } else {
            // Background color (dark blue)
            image[idx]     = 15;  // R
            image[idx + 1] = 25;  // G
            image[idx + 2] = 40;  // B
        }
    } else {
        // Background color (dark blue)
        image[idx]     = 15;  // R
        image[idx + 1] = 25;  // G
        image[idx + 2] = 40;  // B
    }
}
