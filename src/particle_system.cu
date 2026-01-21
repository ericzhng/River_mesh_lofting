// Include CUDA headers first
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

// Include our header (no OpenGL)
#include "particle_system.cuh"

// Only include OpenGL headers in host code sections
#ifndef __CUDA_ARCH__
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#endif

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

// ParticleSystem Implementation
ParticleSystem::ParticleSystem()
    : d_particles(nullptr), numParticles(0), cuda_vbo_resource(nullptr),
      particleVBO(0), d_randStates(nullptr), spawnRate(100.0f),
      particleLifetime(5.0f), numRiverSegments(0) {
}

ParticleSystem::~ParticleSystem() {
    Cleanup();
}

void ParticleSystem::Initialize(int numParticles, int numRiverSegments, float spawnRate) {
    this->numParticles = numParticles;
    this->numRiverSegments = numRiverSegments;
    this->spawnRate = spawnRate;

    // Allocate device memory
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMalloc(&d_randStates, numParticles * sizeof(curandState));

    // Create OpenGL VBO for particles
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, particleVBO,
                                 cudaGraphicsRegisterFlagsWriteDiscard);

    // Initialize particles
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    unsigned int seed = 12345;
    InitializeParticlesKernel<<<gridSize, blockSize>>>(
        d_particles, numParticles, numRiverSegments, d_randStates, seed);

    cudaDeviceSynchronize();
}

void ParticleSystem::Update(float deltaTime, cudaGraphicsResource* simDataResource) {
    if (!d_particles) return;

    // Map simulation data
    cudaGraphicsMapResources(1, &simDataResource, 0);
    float4* d_simData;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_simData, &num_bytes, simDataResource);

    // Update particles
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    UpdateParticlesKernel<<<gridSize, blockSize>>>(
        d_particles, numParticles, d_simData, numRiverSegments,
        deltaTime, d_randStates);

    cudaGraphicsUnmapResources(1, &simDataResource, 0);

    // Map VBO and update
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    float4* d_vbo;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &num_bytes, cuda_vbo_resource);

    ParticlesToVertexBufferKernel<<<gridSize, blockSize>>>(d_particles, d_vbo, numParticles);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    cudaDeviceSynchronize();
}

void ParticleSystem::Cleanup() {
    if (cuda_vbo_resource) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        cuda_vbo_resource = nullptr;
    }
    if (particleVBO) {
        glDeleteBuffers(1, &particleVBO);
        particleVBO = 0;
    }
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
    if (d_randStates) {
        cudaFree(d_randStates);
        d_randStates = nullptr;
    }
}
