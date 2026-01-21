#version 430 core

// Input attributes from our C++ Vertex struct
layout (location = 0) in vec3 aPos;         // Base position (on river bed)
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;   // u = along river, v = across (0=L, 1=R)
layout (location = 3) in float aSimIndex;   // The index 'i' in the 1D simulation

// Simulation Data from CUDA
uniform samplerBuffer uSimData; // Pack: [h1, u1, 0, 0, h2, u2, 0, 0, ...]
uniform float uBankSlope;       // Horizontal distance per 1 unit of height
uniform float uTime;            // For animated waves

// Camera Matrices
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uModel;

// Outputs to Fragment Shader
out vec2 vTexCoords;
out float vVelocity;
out float vDepth;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vViewPos;
out vec3 vTangent;
out vec3 vBitangent;

// Procedural wave function
vec3 calculateWaves(vec3 pos, float velocity, out vec3 normal) {
    vec3 wavePos = pos;

    // Base wave parameters influenced by velocity
    float waveSpeed = 0.5 + abs(velocity) * 0.3;
    float waveFreq = 2.0 + abs(velocity) * 0.5;
    float waveAmp = 0.05 + abs(velocity) * 0.02;

    // Primary wave (flow direction)
    float wave1 = sin(pos.z * waveFreq + uTime * waveSpeed) * waveAmp;

    // Secondary wave (perpendicular)
    float wave2 = sin(pos.x * waveFreq * 0.7 + uTime * waveSpeed * 0.8) * waveAmp * 0.5;

    // Tertiary detail waves
    float wave3 = sin((pos.z + pos.x) * waveFreq * 2.0 + uTime * waveSpeed * 1.5) * waveAmp * 0.3;

    wavePos.y += wave1 + wave2 + wave3;

    // Calculate normal using derivatives
    float dx = waveAmp * 0.5 * cos(pos.x * waveFreq * 0.7 + uTime * waveSpeed * 0.8) * waveFreq * 0.7;
    dx += waveAmp * 0.3 * cos((pos.z + pos.x) * waveFreq * 2.0 + uTime * waveSpeed * 1.5) * waveFreq * 2.0;

    float dz = waveAmp * cos(pos.z * waveFreq + uTime * waveSpeed) * waveFreq;
    dz += waveAmp * 0.3 * cos((pos.z + pos.x) * waveFreq * 2.0 + uTime * waveSpeed * 1.5) * waveFreq * 2.0;

    normal = normalize(vec3(-dx, 1.0, -dz));

    return wavePos;
}

void main() {
    // 1. Fetch simulation results for this specific river segment
    vec4 simResults = texelFetch(uSimData, int(aSimIndex));
    float h = simResults.x;         // Water Depth
    float velocity = simResults.y;  // Flow Velocity

    vec3 displacedPos = aPos;

    // Lift the water to the current depth
    displacedPos.y += h;

    // Expand the surface width-wise based on the bank slope
    float sideSign = (aTexCoords.y > 0.5) ? 1.0 : -1.0;
    displacedPos.x += sideSign * (h * uBankSlope);

    // Add procedural waves
    vec3 waveNormal;
    displacedPos = calculateWaves(displacedPos, velocity, waveNormal);

    // Calculate tangent space for normal mapping
    vec3 tangent = normalize(vec3(0, 0, 1));  // Flow direction
    vec3 bitangent = normalize(cross(waveNormal, tangent));
    tangent = cross(bitangent, waveNormal);

    // Output to Fragment Shader
    vTexCoords = aTexCoords;
    vVelocity = velocity;
    vDepth = h;
    vWorldPos = vec3(uModel * vec4(displacedPos, 1.0));
    vNormal = normalize(mat3(uModel) * waveNormal);
    vTangent = normalize(mat3(uModel) * tangent);
    vBitangent = normalize(mat3(uModel) * bitangent);

    // View position for Fresnel calculation
    vec4 viewPos = uView * vec4(vWorldPos, 1.0);
    vViewPos = viewPos.xyz;

    gl_Position = uProjection * viewPos;
}
