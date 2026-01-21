#version 430 core

in vec2 vTexCoords;
in float vVelocity;
in float vDepth;
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vViewPos;
in vec3 vTangent;
in vec3 vBitangent;

uniform vec3 uCameraPos;
uniform float uTime;

out vec4 FragColor;

// Physically-based water colors
const vec3 deepWaterColor = vec3(0.02, 0.15, 0.3);     // Dark blue for deep water
const vec3 shallowWaterColor = vec3(0.1, 0.5, 0.7);    // Cyan for shallow
const vec3 foamColor = vec3(0.9, 0.95, 1.0);           // White foam
const vec3 skyColor = vec3(0.5, 0.7, 0.9);             // Sky reflection

// Fresnel effect (Schlick's approximation)
float fresnel(vec3 viewDir, vec3 normal, float f0) {
    float cosTheta = max(dot(viewDir, normal), 0.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// Procedural foam generation
float generateFoam(vec2 uv, float velocity, float time) {
    // Fast-flowing areas generate more foam
    float foamAmount = smoothstep(2.0, 8.0, abs(velocity));

    // Animated foam pattern
    float foam = 0.0;
    vec2 foamUV = uv * 10.0;

    foam += sin(foamUV.x * 2.0 + time * 2.0) * cos(foamUV.y * 2.0 + time * 1.5);
    foam += sin(foamUV.x * 3.0 - time * 3.0) * cos(foamUV.y * 3.0 - time * 2.5);
    foam = foam * 0.5 + 0.5;

    // Threshold for foam visibility
    foam = smoothstep(0.7, 0.9, foam) * foamAmount;

    return foam;
}

// Enhanced normal perturbation for detail
vec3 perturbNormal(vec3 normal, vec2 uv, float time, float velocity) {
    // Create flowing normal map effect
    vec2 flowUV = uv + vec2(time * 0.1 * (1.0 + abs(velocity) * 0.5), time * 0.05);

    // Detail ripples
    float ripple1 = sin(flowUV.x * 15.0 + time * 2.0) * cos(flowUV.y * 15.0);
    float ripple2 = sin(flowUV.x * 20.0 - time * 3.0) * cos(flowUV.y * 20.0);

    vec3 perturbation = vec3(ripple1, 0.0, ripple2) * 0.1;

    return normalize(normal + perturbation);
}

// Depth-based transparency and color
vec3 getWaterColor(float depth, vec3 viewDir, vec3 normal) {
    // Shallow to deep gradient
    float depthFactor = smoothstep(0.5, 4.0, depth);
    vec3 waterColor = mix(shallowWaterColor, deepWaterColor, depthFactor);

    // Add slight color variation based on depth
    waterColor += vec3(0.02, 0.05, 0.03) * sin(depth * 2.0);

    return waterColor;
}

// Specular highlight
vec3 calculateSpecular(vec3 normal, vec3 viewDir, vec3 lightDir, float roughness) {
    vec3 halfVec = normalize(viewDir + lightDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), (1.0 - roughness) * 128.0);
    return vec3(spec);
}

void main() {
    // Normalize interpolated vectors
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);

    // Enhance normal with procedural detail
    normal = perturbNormal(normal, vTexCoords, uTime, vVelocity);

    // Base water color based on depth
    vec3 waterColor = getWaterColor(vDepth, viewDir, normal);

    // Fresnel for reflections (water is more reflective at grazing angles)
    float fresnelTerm = fresnel(viewDir, normal, 0.02);

    // Sky/environment reflection (simplified - in VR you'd use cubemap)
    vec3 reflectionColor = skyColor + vec3(0.1) * sin(uTime * 0.5);

    // Mix water color with reflection
    vec3 finalColor = mix(waterColor, reflectionColor, fresnelTerm);

    // Velocity-based coloring (shows flow speed)
    float velocityIntensity = smoothstep(0.0, 5.0, abs(vVelocity));
    vec3 velocityColor = vec3(0.0, 0.2, 0.4) * velocityIntensity;
    finalColor += velocityColor * 0.3;

    // Add foam in high-velocity areas
    float foam = generateFoam(vTexCoords, vVelocity, uTime);
    finalColor = mix(finalColor, foamColor, foam);

    // Sunlight specular
    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.3));
    float roughness = 0.1 + abs(vVelocity) * 0.1; // Turbulent water is rougher
    vec3 specular = calculateSpecular(normal, viewDir, sunDir, roughness);
    finalColor += specular * 0.6;

    // Depth-based transparency
    float alpha = 0.85 + fresnelTerm * 0.15; // More opaque at edges due to Fresnel
    alpha = min(alpha, 0.95); // Keep slightly transparent for underwater view

    // Add subtle shimmer
    float shimmer = sin(vWorldPos.x * 10.0 + uTime * 5.0) *
                    cos(vWorldPos.z * 10.0 + uTime * 4.0) * 0.02;
    finalColor += vec3(shimmer);

    FragColor = vec4(finalColor, alpha);
}
