#version 430 core

layout (location = 0) in vec4 aParticleData; // xyz = position, w = life factor

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uParticleSize;
uniform samplerBuffer uSimData;  // Access to simulation data for velocity

out float vLifeFactor;
out vec3 vVelocity;
out float vDepth;

void main() {
    vec3 position = aParticleData.xyz;
    vLifeFactor = aParticleData.w;

    // Get simulation data for this particle's location
    // Note: This is a simplified approach - you'd need to pass segment index
    // For now, we'll estimate from position
    int segmentIdx = int(position.z / 1.5); // Assuming 1.5 spacing
    segmentIdx = clamp(segmentIdx, 0, 99); // Assuming 100 segments

    vec4 simData = texelFetch(uSimData, segmentIdx);
    vDepth = simData.x;
    float velocity = simData.y;

    // Estimate velocity direction (flow along Z)
    vVelocity = vec3(0.0, 0.0, velocity);

    // Billboard the particle to face the camera
    vec4 viewPos = uView * vec4(position, 1.0);

    gl_Position = uProjection * viewPos;

    // Size based on life (fade out as particle ages)
    // Larger when moving fast
    float speedFactor = 1.0 + abs(velocity) * 0.2;
    gl_PointSize = uParticleSize * vLifeFactor * speedFactor;
}
