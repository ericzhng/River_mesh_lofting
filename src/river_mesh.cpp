#include "river_mesh.hpp"
#include <iostream>

Mesh GenerateRiverBedMesh(const std::vector<RiverNode>& nodes)
{
    Mesh mesh;
    if (nodes.size() < 2) return mesh;

    // We create 4 vertices per cross-section (Trapezoid for the river bed):
    // 0: Left Bank Top, 1: Left Bank Bottom, 2: Right Bank Bottom, 3: Right Bank Top
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const RiverNode& current = nodes[i];

        // 1. Calculate Direction (Tangent)
        glm::vec3 tangent;
        if (i < nodes.size() - 1) {
            tangent = glm::normalize(nodes[i + 1].position - current.position);
        } else {
            tangent = glm::normalize(current.position - nodes[i - 1].position);
        }

        // 2. Calculate the "Right" vector
        glm::vec3 up = {0, 1, 0};
        glm::vec3 right = glm::normalize(glm::cross(tangent, up));

        // 3. Create the 4 points of the cross-section
        float halfWidth = current.bottomWidth / 2.0f;
        glm::vec3 pLeftBottom = current.position - (right * halfWidth);
        glm::vec3 pRightBottom = current.position + (right * halfWidth);
        glm::vec3 pLeftTop = pLeftBottom - (right * current.bankSlope) + (up * current.bankHeight);
        glm::vec3 pRightTop = pRightBottom + (right * current.bankSlope) + (up * current.bankHeight);

        // 4. Add Vertices to Mesh
        float sIdx = (float)i;
        glm::vec3 nUp = {0, 1, 0}; // Simplified normal, pointing up.

        // Add vertices in order: LeftTop, LeftBot, RightBot, RightTop
        // UV.v maps to transverse position (0=left bank, 1=right bank)
        mesh.vertices.push_back({pLeftTop, nUp, {(float)i, 0.0f}, sIdx});
        mesh.vertices.push_back({pLeftBottom, nUp, {(float)i, 0.2f}, sIdx});
        mesh.vertices.push_back({pRightBottom, nUp, {(float)i, 0.8f}, sIdx});
        mesh.vertices.push_back({pRightTop, nUp, {(float)i, 1.0f}, sIdx});

        // 5. Generate Triangles (Indices)
        if (i > 0) {
            unsigned int currBase = i * 4;
            unsigned int prevBase = (i - 1) * 4;

            auto addQuad = [&](unsigned int p1, unsigned int p2, unsigned int c1, unsigned int c2) {
                mesh.indices.push_back(p1); mesh.indices.push_back(c1); mesh.indices.push_back(p2);
                mesh.indices.push_back(c2); mesh.indices.push_back(p2); mesh.indices.push_back(c1);
            };

            addQuad(prevBase + 0, prevBase + 1, currBase + 0, currBase + 1); // Left Bank
            addQuad(prevBase + 1, prevBase + 2, currBase + 1, currBase + 2); // River Bed
            addQuad(prevBase + 2, prevBase + 3, currBase + 2, currBase + 3); // Right Bank
        }
    }
    return mesh;
}

Mesh GenerateWaterSurfaceMesh(const std::vector<RiverNode>& nodes)
{
    Mesh mesh;
    if (nodes.size() < 2) return mesh;

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const RiverNode& current = nodes[i];

        glm::vec3 tangent;
        if (i < nodes.size() - 1)
            tangent = glm::normalize(nodes[i + 1].position - current.position);
        else
            tangent = glm::normalize(current.position - nodes[i - 1].position);

        glm::vec3 up = {0, 1, 0};
        glm::vec3 right = glm::normalize(glm::cross(tangent, up));

        float halfWidth = current.bottomWidth / 2.0f;
        glm::vec3 pLeft = current.position - (right * halfWidth);
        glm::vec3 pRight = current.position + (right * halfWidth);

        glm::vec3 nUp = {0, 1, 0};
        float sIdx = (float)i;

        // Two vertices per cross-section (a ribbon that will be deformed in the shader)
        // v coordinate: 0 = Left Edge, 1 = Right Edge
        mesh.vertices.push_back({pLeft, nUp, {(float)i, 0.0f}, sIdx});
        mesh.vertices.push_back({pRight, nUp, {(float)i, 1.0f}, sIdx});

        if (i > 0) {
            unsigned int currBase = i * 2;
            unsigned int prevBase = (i - 1) * 2;
            mesh.indices.push_back(prevBase + 0); mesh.indices.push_back(currBase + 0); mesh.indices.push_back(prevBase + 1);
            mesh.indices.push_back(prevBase + 1); mesh.indices.push_back(currBase + 0); mesh.indices.push_back(currBase + 1);
        }
    }
    return mesh;
}
