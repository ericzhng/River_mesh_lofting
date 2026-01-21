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
uniform sampler2D uReflectionTexture;  // Optional: for screen-space reflections
uniform bool uUseReflectionTexture;

out vec4 FragColor;

// Enhanced water colors with more variety
const vec3 deepWaterColor = vec3(0.01, 0.12, 0.25);
const vec3 shallowWaterColor = vec3(0.15, 0.55, 0.75);
const vec3 foamColor = vec3(0.95, 0.98, 1.0);
const vec3 skyColorHorizon = vec3(0.6, 0.75, 0.9);
const vec3 skyColorZenith = vec3(0.3, 0.5, 0.8);

// Improved Fresnel with more parameters
float fresnelSchlick(vec3 viewDir, vec3 normal, float f0, float power) {
    float cosTheta = max(dot(viewDir, normal), 0.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, power);
}

// Multi-octave noise for more natural patterns
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Smooth interpolation

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for(int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Advanced foam generation with turbulence
float generateAdvancedFoam(vec2 uv, float velocity, float depth, float time) {
    float foamAmount = smoothstep(3.0, 10.0, abs(velocity));

    // Shallow water foam
    foamAmount += smoothstep(0.8, 0.3, depth) * 0.4;

    // Turbulent foam pattern using FBM
    vec2 foamUV = uv * 8.0;
    foamUV.x += time * 0.3 * (1.0 + abs(velocity) * 0.2);
    foamUV.y += time * 0.15;

    float foamNoise = fbm(foamUV, 3);

    // Add velocity-based streaks
    float streaks = sin(uv.x * 20.0 + time * 2.0 * sign(velocity)) *
                    cos(uv.y * 15.0 - time * 1.5);
    streaks = streaks * 0.5 + 0.5;

    float foam = foamNoise * streaks;
    foam = smoothstep(0.55, 0.85, foam) * foamAmount;

    return foam;
}

// Advanced normal perturbation with multiple layers
vec3 calculateDetailedNormal(vec3 baseNormal, vec2 uv, float time, float velocity) {
    mat3 TBN = mat3(vTangent, vBitangent, baseNormal);

    // Multiple normal map layers
    vec2 flowUV1 = uv + vec2(time * 0.08, time * 0.03) * (1.0 + abs(velocity) * 0.4);
    vec2 flowUV2 = uv * 1.7 + vec2(-time * 0.05, time * 0.07) * (1.0 + abs(velocity) * 0.3);
    vec2 flowUV3 = uv * 2.3 + vec2(time * 0.03, -time * 0.04);

    // Calculate normals from procedural patterns
    float n1 = fbm(flowUV1 * 8.0, 2);
    float n2 = fbm(flowUV2 * 12.0, 2);
    float n3 = noise(flowUV3 * 20.0);

    // Combine into tangent-space normal
    vec3 tangentNormal;
    tangentNormal.x = (n1 - 0.5) * 0.3 + (n2 - 0.5) * 0.15;
    tangentNormal.y = (n2 - 0.5) * 0.3 + (n3 - 0.5) * 0.1;
    tangentNormal.z = 1.0;

    tangentNormal = normalize(tangentNormal);
    return normalize(TBN * tangentNormal);
}

// Depth-based color with atmospheric scattering
vec3 getEnhancedWaterColor(float depth, vec3 viewDir, vec3 normal) {
    float depthFactor = smoothstep(0.3, 5.0, depth);
    vec3 waterColor = mix(shallowWaterColor, deepWaterColor, depthFactor);

    // Add depth color variation (algae, sediment)
    float colorVariation = fbm(vWorldPos.xz * 0.5, 2);
    waterColor += vec3(0.02, 0.04, 0.02) * colorVariation * depthFactor;

    // Subsurface scattering approximation
    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.3));
    float subsurface = pow(max(dot(-viewDir, sunDir), 0.0), 3.0) * (1.0 - depthFactor);
    waterColor += vec3(0.1, 0.3, 0.2) * subsurface;

    return waterColor;
}

// Improved sky reflection
vec3 getSkyColor(vec3 reflectDir) {
    float skyGradient = max(reflectDir.y, 0.0);
    return mix(skyColorHorizon, skyColorZenith, skyGradient);
}

// Edge foam (where water meets banks)
float getEdgeFoam(float depth, vec2 uv, float time) {
    if (depth > 1.0) return 0.0;

    float edgeDist = smoothstep(0.5, 0.1, depth);
    float edgeNoise = fbm(uv * 15.0 + time * 0.2, 2);

    return edgeDist * smoothstep(0.4, 0.7, edgeNoise);
}

// Specular with anisotropic highlights
vec3 calculateAnisotropicSpecular(vec3 normal, vec3 viewDir, vec3 lightDir,
                                   vec3 tangent, float roughness, float anisotropy) {
    vec3 halfVec = normalize(viewDir + lightDir);

    vec3 bitangent = cross(normal, tangent);
    float dotTH = dot(tangent, halfVec);
    float dotBH = dot(bitangent, halfVec);
    float dotNH = dot(normal, halfVec);

    float aspect = sqrt(1.0 - anisotropy * 0.9);
    float ax = max(0.001, roughness / aspect);
    float ay = max(0.001, roughness * aspect);

    float D = 1.0 / (3.14159 * ax * ay * pow(dotNH, 4.0));
    D *= exp(-((dotTH * dotTH) / (ax * ax) + (dotBH * dotBH) / (ay * ay)) / (dotNH * dotNH));

    float spec = D * 0.25;
    return vec3(spec);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);

    // Enhanced normal with detail
    normal = calculateDetailedNormal(normal, vTexCoords, uTime, vVelocity);

    // Base water color with depth
    vec3 waterColor = getEnhancedWaterColor(vDepth, viewDir, normal);

    // Enhanced Fresnel
    float fresnelTerm = fresnelSchlick(viewDir, normal, 0.02, 5.0);

    // Reflection
    vec3 reflectDir = reflect(-viewDir, normal);
    vec3 reflectionColor = getSkyColor(reflectDir);

    // Add sun reflection
    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.3));
    float sunReflection = pow(max(dot(reflectDir, sunDir), 0.0), 128.0);
    reflectionColor += vec3(1.0, 0.95, 0.8) * sunReflection * 2.0;

    // Mix water color with reflection
    vec3 finalColor = mix(waterColor, reflectionColor, fresnelTerm * 0.9);

    // Velocity-based effects
    float velocityIntensity = smoothstep(0.0, 6.0, abs(vVelocity));
    vec3 velocityColor = vec3(0.0, 0.25, 0.4) * velocityIntensity;
    finalColor += velocityColor * 0.25;

    // Advanced foam
    float foam = generateAdvancedFoam(vTexCoords, vVelocity, vDepth, uTime);
    float edgeFoam = getEdgeFoam(vDepth, vTexCoords, uTime);
    float totalFoam = clamp(foam + edgeFoam, 0.0, 1.0);
    finalColor = mix(finalColor, foamColor, totalFoam);

    // Anisotropic specular (flow-aligned highlights)
    vec3 flowTangent = normalize(vTangent + vec3(0, 0.1, sign(vVelocity)));
    float roughness = 0.08 + abs(vVelocity) * 0.08;
    float anisotropy = 0.6;
    vec3 specular = calculateAnisotropicSpecular(normal, viewDir, sunDir,
                                                  flowTangent, roughness, anisotropy);
    finalColor += specular * vec3(1.0, 0.98, 0.9) * 1.5;

    // Depth fog (atmospheric perspective)
    float viewDistance = length(vViewPos);
    float fog = exp(-viewDistance * 0.01);
    finalColor = mix(vec3(0.5, 0.6, 0.7), finalColor, fog);

    // Depth-based transparency with Fresnel
    float alpha = 0.80 + fresnelTerm * 0.20;
    alpha = mix(alpha, 0.95, totalFoam);

    // Add subtle chromatic aberration at edges
    float edgeFresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.0);
    vec3 aberration = vec3(0.01, 0.0, -0.01) * edgeFresnel;
    finalColor += aberration * 0.3;

    FragColor = vec4(finalColor, alpha);
}
