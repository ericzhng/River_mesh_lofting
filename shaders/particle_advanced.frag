#version 430 core

in float vLifeFactor;
in vec3 vVelocity;      // Particle velocity for direction
in float vDepth;        // Water depth at particle location

out vec4 FragColor;

void main() {
    // Create circular particle with soft edges
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    if (dist > 0.5) {
        discard;
    }

    // Elongate particle in direction of motion (motion blur)
    vec2 velocityDir = normalize(vVelocity.xz);
    float elongation = length(vVelocity.xz) * 0.15;
    vec2 elongatedCoord = coord;
    elongatedCoord -= velocityDir * dot(coord, velocityDir) * elongation;
    float elongatedDist = length(elongatedCoord);

    // Soft edges with elongation
    float alpha = 1.0 - smoothstep(0.2, 0.5, elongatedDist);
    alpha *= vLifeFactor; // Fade based on particle age

    // Brightness based on center distance (glowing core)
    float brightness = 1.0 - elongatedDist * 1.5;
    brightness = max(brightness, 0.0);

    // Color gradient based on depth and velocity
    vec3 shallowColor = vec3(0.4, 0.8, 1.0);    // Cyan for shallow
    vec3 deepColor = vec3(0.1, 0.4, 0.9);       // Blue for deep

    float depthFactor = smoothstep(0.5, 3.0, vDepth);
    vec3 baseColor = mix(shallowColor, deepColor, depthFactor);

    // Add velocity-based color shift (faster = whiter)
    float speed = length(vVelocity);
    float speedFactor = smoothstep(0.0, 5.0, speed);
    vec3 fastColor = vec3(0.9, 0.95, 1.0);
    baseColor = mix(baseColor, fastColor, speedFactor * 0.6);

    // Brighter core
    vec3 coreColor = vec3(1.0, 1.0, 1.0);
    vec3 finalColor = mix(baseColor, coreColor, brightness * 0.7);

    // Add slight sparkle variation
    float sparkle = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    finalColor += vec3(sparkle * 0.1) * brightness;

    // Depth-based intensity (brighter near surface)
    float surfaceIntensity = 1.0 - smoothstep(0.0, 2.0, vDepth) * 0.5;
    alpha *= surfaceIntensity;

    FragColor = vec4(finalColor, alpha * 0.8);
}
