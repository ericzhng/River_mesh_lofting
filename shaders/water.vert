#version 430 core

// Input attributes from our C++ Vertex struct
layout (location = 0) in vec3 aPos;         // Base position (on river bed)
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;   // u = along river, v = across (0=L, 1=R)
layout (location = 3) in float aSimIndex;   // The index 'i' in the 1D simulation

// Simulation Data from CUDA
uniform samplerBuffer uSimData; // Pack: [h1, u1, 0, 0, h2, u2, 0, 0, ...]
uniform float uBankSlope;       // Horizontal distance per 1 unit of height

// Camera Matrices
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uModel;

// Outputs to Fragment Shader
out vec2 vTexCoords;
out float vVelocity;
out vec3 vWorldPos;

void main() {
    // 1. Fetch simulation results for this specific river segment
    // texelFetch pulls data from the buffer at the specific integer index
    vec4 simResults = texelFetch(uSimData, int(aSimIndex));
    float h = simResults.x;         // Water Depth
    float velocity = simResults.y;  // Flow Velocity

    // 2. Calculate the "Expansion" vector
    // We need to know which way is 'Right' relative to the river flow.
    // In our C++ mesh, we already baked the 'Right' direction logic into the geometry,
    // but here we simply displace based on whether we are the Left or Right vertex.
    
    vec3 displacedPos = aPos;
    
    // Lift the water to the current depth
    displacedPos.y += h;

    // Expand the surface width-wise based on the bank slope
    // aTexCoords.y is 0.0 for Left edge and 1.0 for Right edge
    float sideSign = (aTexCoords.y > 0.5) ? 1.0 : -1.0;
    
    // Assuming the river flows primarily along Z, 'Right' is along X.
    // For a more robust solution, you could pass the 'Right' vector as a vertex attribute.
    displacedPos.x += sideSign * (h * uBankSlope);

    // 3. Output to Fragment Shader
    vTexCoords = aTexCoords;
    vVelocity = velocity;
    vWorldPos = vec3(uModel * vec4(displacedPos, 1.0));

    gl_Position = uProjection * uView * vec4(vWorldPos, 1.0);
}
