#version 430 core

in vec2 vTexCoords;
in float vVelocity;
in vec3 vWorldPos;

out vec4 FragColor;

void main() {
    // Basic blue color for water
    vec3 waterColor = vec3(0.1, 0.4, 0.8);

    // Visualize velocity - mix in some white for faster areas
    float velocityFactor = smoothstep(0.0, 5.0, abs(vVelocity));
    vec3 finalColor = mix(waterColor, vec3(1.0), velocityFactor * 0.5);

    FragColor = vec4(finalColor, 1.0);
}
