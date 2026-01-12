#include <GL/glew.h>
#include "simulation.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h> // For graphics resource mapping

// --- CUDA Kernel and Device Functions ---

__constant__ float G = 9.81f;
__constant__ float DT = 0.01f;
__constant__ float DX = 1.0f;

__device__ float2 ComputeFlux(State L, State R)
{
    float AL = L.h;
    float AR = R.h;
    float QL = L.h * L.u;
    float QR = R.h * R.u;

    float flux_A = 0.5f * (QL + QR);
    float flux_Q = 0.5f * ((QL * L.u + 0.5f * G * AL * AL) + (QR * R.u + 0.5f * G * AR * AR));

    return make_float2(flux_A, flux_Q);
}

__global__ void RiverSimulationKernel(
    State *oldState,
    State *newState,
    float4 *renderBuffer,
    int numCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= numCells - 1)
        return;

    State L = oldState[i - 1];
    State C = oldState[i];
    State R = oldState[i + 1];

    float2 fluxL = ComputeFlux(L, C);
    float2 fluxR = ComputeFlux(C, R);

    float A_new = C.h - (DT / DX) * (fluxR.x - fluxL.x);
    float Q_new = (C.h * C.u) - (DT / DX) * (fluxR.y - fluxL.y);

    float h_next = A_new;
    float u_next = (A_new > 0.001f) ? Q_new / A_new : 0.0f;

    newState[i] = {h_next, u_next};

    if (renderBuffer) {
        renderBuffer[i] = make_float4(h_next, u_next, 0.0f, 0.0f);
    }
}

// --- RiverSimulation Class Implementation ---

RiverSimulation::RiverSimulation() {}

RiverSimulation::~RiverSimulation()
{
    cudaFree(d_oldState);
    cudaFree(d_newState);
}

void RiverSimulation::Initialize(int numCells, const std::vector<State>& initialState)
{
    m_numCells = numCells;
    size_t dataSize = numCells * sizeof(State);

    cudaMalloc(&d_oldState, dataSize);
    cudaMalloc(&d_newState, dataSize);

    cudaMemcpy(d_oldState, initialState.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_newState, initialState.data(), dataSize, cudaMemcpyHostToDevice);
}

void RiverSimulation::Step(cudaGraphicsResource* resource)
{
    if (m_numCells == 0) return;

    // 1. Map OpenGL buffer if provided
    float4* d_renderPtr = nullptr;
    if (resource) {
        size_t size;
        cudaGraphicsMapResources(1, &resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_renderPtr, &size, resource);
    }

    // 2. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_numCells + threadsPerBlock - 1) / threadsPerBlock;
    RiverSimulationKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_oldState,
        d_newState,
        d_renderPtr,
        m_numCells);

    // 3. Unmap resource
    if (resource) {
        cudaGraphicsUnmapResources(1, &resource, 0);
    }
    
    // 4. Swap buffers for next iteration
    State* temp = d_oldState;
    d_oldState = d_newState;
    d_newState = temp; // Old state now holds the result, which is fine
}
