#version 430 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

// Camera Matrices
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uModel;

// Outputs to Fragment Shader
out vec3 vNormal;
out vec3 vWorldPos;

void main() {
    vWorldPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;

    gl_Position = uProjection * uView * vec4(vWorldPos, 1.0);
}
