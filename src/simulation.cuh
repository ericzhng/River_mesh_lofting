#pragma once

#include <vector>

// Forward declaration for CUDA graphics resource
struct cudaGraphicsResource;

// Simulation state for one cell
struct State
{
    float h; // Depth
    float u; // Velocity
};

class RiverSimulation
{
public:
    RiverSimulation();
    ~RiverSimulation();

    void Initialize(int numCells, const std::vector<State>& initialState);
    void Step(cudaGraphicsResource* resource);
    int GetNumCells() const { return m_numCells; }

private:
    int m_numCells = 0;
    State* d_oldState = nullptr;
    State* d_newState = nullptr;
};
