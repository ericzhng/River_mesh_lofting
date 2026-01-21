#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Forward declarations to avoid including OpenGL headers in CUDA code
struct cudaGraphicsResource;

// Particle structure for GPU
struct Particle {
    float3 position;      // Current 3D position
    float3 velocity;      // Current velocity
    float age;            // Particle age (for fading)
    float lifetime;       // Maximum lifetime
    float riverIndex;     // Which segment the particle is on
    float crossPosition;  // Position across river (0=left, 1=right)
};

class ParticleSystem {
private:
    Particle* d_particles;           // Device particles
    int numParticles;

    cudaGraphicsResource* cuda_vbo_resource;
    unsigned int particleVBO;

    // Random state for particle spawning
    curandState* d_randStates;

    // Simulation parameters
    float spawnRate;
    float particleLifetime;
    int numRiverSegments;

public:
    ParticleSystem();
    ~ParticleSystem();

    void Initialize(int numParticles, int numRiverSegments, float spawnRate = 100.0f);
    void Update(float deltaTime, cudaGraphicsResource* simDataResource);
    void Cleanup();

    unsigned int GetVBO() const { return particleVBO; }
    int GetNumParticles() const { return numParticles; }
};

// CUDA kernel declarations
__global__ void InitializeParticlesKernel(Particle* particles, int numParticles,
                                          int numSegments, curandState* randStates, unsigned int seed);

__global__ void UpdateParticlesKernel(Particle* particles, int numParticles,
                                      float4* simData, int numSegments,
                                      float deltaTime, curandState* randStates);

__global__ void ParticlesToVertexBufferKernel(Particle* particles, float4* vbo, int numParticles);
