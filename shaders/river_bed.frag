#version 430 core

in vec3 vNormal;
in vec3 vWorldPos;

out vec4 FragColor;

void main() {
    // A simple brown/grey color for the river bed
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.7));
    float diff = max(dot(normalize(vNormal), lightDir), 0.2);
    vec3 color = vec3(0.4, 0.35, 0.3) * diff;

    FragColor = vec4(color, 1.0);
}
