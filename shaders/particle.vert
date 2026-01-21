#version 430 core

layout (location = 0) in vec4 aParticleData; // xyz = position, w = life factor

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uParticleSize;

out float vLifeFactor;

void main() {
    vec3 position = aParticleData.xyz;
    vLifeFactor = aParticleData.w;

    // Billboard the particle to face the camera
    vec4 viewPos = uView * vec4(position, 1.0);

    gl_Position = uProjection * viewPos;

    // Point size based on life (fade out as particle ages)
    gl_PointSize = uParticleSize * vLifeFactor;
}
