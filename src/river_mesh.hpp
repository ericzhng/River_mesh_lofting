#pragma once

#include <vector>
#include <cmath>

// Use GLM for vector math, it's the standard and much more robust.
// We'll assume it's in the vendor/ directory.
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

// --- Data Structures ---

// Input: A single point on your 1D river centerline
struct RiverNode
{
    glm::vec3 position;  // The center of the river bottom (x, y, z)
    float bottomWidth; // Width of the flat bottom
    float bankSlope;   // Horizontal distance for the banks (trapezoid shape)
    float bankHeight;  // How high the banks go
};

// Output: A single vertex for OpenGL
struct Vertex
{
    glm::vec3 pos;      // 3D Position
    glm::vec3 normal;   // For lighting
    glm::vec2 uv;       // Texture coordinates (u=along flow, v=across)
    float simIndex;     // Corresponds to index 'i' in your 1D Simulation
};

// The Mesh Container
struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
};

// --- Mesh Generator Function Declarations ---

Mesh GenerateRiverBedMesh(const std::vector<RiverNode>& nodes);
Mesh GenerateWaterSurfaceMesh(const std::vector<RiverNode>& nodes);
