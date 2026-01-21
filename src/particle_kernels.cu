// Pure CUDA kernels - no OpenGL headers
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "particle_system.cuh"

// CUDA kernel to initialize particles
__global__ void InitializeParticlesKernel(Particle* particles, int numParticles,
                                          int numSegments, curandState* randStates, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Initialize random state
    curand_init(seed, idx, 0, &randStates[idx]);

    // Random position along river
    float riverIdx = curand_uniform(&randStates[idx]) * (numSegments - 1);
    float crossPos = curand_uniform(&randStates[idx]);

    particles[idx].position = make_float3(0.0f, 0.0f, riverIdx * 1.5f);
    particles[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
    particles[idx].age = curand_uniform(&randStates[idx]) * 5.0f;
    particles[idx].lifetime = 5.0f;
    particles[idx].riverIndex = riverIdx;
    particles[idx].crossPosition = crossPos;
}

// CUDA kernel to update particles based on flow field
__global__ void UpdateParticlesKernel(Particle* particles, int numParticles,
                                      float4* simData, int numSegments,
                                      float deltaTime, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];

    // Age the particle
    p.age += deltaTime;

    // Respawn if too old
    if (p.age > p.lifetime) {
        // Spawn at upstream end
        p.riverIndex = 0.0f;
        p.crossPosition = 0.3f + curand_uniform(&randStates[idx]) * 0.4f; // Center bias
        p.age = 0.0f;
        p.lifetime = 3.0f + curand_uniform(&randStates[idx]) * 4.0f;
        p.position.y = 0.0f;
    }

    // Get simulation data for current segment
    int segIdx = (int)p.riverIndex;
    if (segIdx < 0) segIdx = 0;
    if (segIdx >= numSegments) segIdx = numSegments - 1;

    float4 sim = simData[segIdx];
    float depth = sim.x;      // Water depth
    float velocity = sim.y;   // Flow velocity

    // Update position based on velocity
    // Simplified: assume flow is along Z-axis
    float flowSpeed = velocity;
    p.riverIndex += flowSpeed * deltaTime / 1.5f; // 1.5 is segment spacing

    // Respawn if reached end
    if (p.riverIndex >= numSegments - 1) {
        p.riverIndex = 0.0f;
        p.age = 0.0f;
    }

    // Update 3D position
    segIdx = (int)p.riverIndex;
    float segFrac = p.riverIndex - segIdx;

    // Interpolate between segments
    float z = p.riverIndex * 1.5f;
    float x = sinf(p.riverIndex / 10.0f) * 5.0f; // Match curve in main.cpp

    // Add cross-river position
    float halfWidth = 4.0f; // Match bottomWidth/2 from main.cpp
    float crossOffset = (p.crossPosition - 0.5f) * 2.0f * halfWidth;

    // Add some randomness to cross position for natural look
    crossOffset += sinf(p.age * 2.0f + idx * 0.1f) * 0.5f;

    p.position.x = x + crossOffset;
    p.position.y = depth * 0.9f; // Float at 90% of water depth
    p.position.z = z;

    // Add small vertical bobbing
    p.position.y += sinf(p.age * 3.0f + idx * 0.5f) * 0.1f;

    // Update velocity for rendering
    p.velocity = make_float3(0.0f, 0.0f, flowSpeed);
}

// CUDA kernel to convert particles to vertex buffer format
__global__ void ParticlesToVertexBufferKernel(Particle* particles, float4* vbo, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];

    // Pack particle data for rendering
    // xyz = position, w = age/lifetime for alpha blending
    vbo[idx] = make_float4(p.position.x, p.position.y, p.position.z,
                           1.0f - (p.age / p.lifetime));
}
