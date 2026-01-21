#version 430 core

in float vLifeFactor;

out vec4 FragColor;

void main() {
    // Create circular particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    if (dist > 0.5) {
        discard;
    }

    // Soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    alpha *= vLifeFactor; // Fade based on particle age

    // Bright center, darker edges for glow effect
    float brightness = 1.0 - dist * 2.0;

    // Color gradient: white center to cyan edges
    vec3 centerColor = vec3(1.0, 1.0, 1.0);
    vec3 edgeColor = vec3(0.3, 0.7, 1.0);
    vec3 color = mix(edgeColor, centerColor, brightness);

    FragColor = vec4(color, alpha * 0.7);
}
