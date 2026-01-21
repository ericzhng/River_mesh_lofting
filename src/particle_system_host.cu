// Host-side particle system code with OpenGL interop
// This must be compiled with NVCC to support kernel launches (<<<>>>)

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "particle_system.cuh"

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
