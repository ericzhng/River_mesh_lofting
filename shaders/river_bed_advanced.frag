#version 430 core

in vec3 vNormal;
in vec3 vWorldPos;

uniform float uTime;
uniform vec3 uCameraPos;

out vec4 FragColor;

// Hash function for procedural textures
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Value noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractional Brownian Motion
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

// Voronoi pattern for rocks
vec2 voronoi(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float minDist = 1.0;
    vec2 minPoint;

    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash(i + neighbor) * vec2(1.0) + neighbor;
            float dist = length(point - f);

            if(dist < minDist) {
                minDist = dist;
                minPoint = point;
            }
        }
    }

    return vec2(minDist, hash(i + minPoint));
}

// Advanced caustics with refraction simulation
float advancedCaustics(vec2 uv, float time) {
    float caustic = 0.0;

    // Layer 1: Large caustic patterns
    vec2 p1 = uv * 3.0;
    for(int i = 0; i < 4; i++) {
        float t = time * 0.4 + float(i) * 0.5;
        p1 += vec2(cos(t - p1.y) * 0.5, sin(t + p1.x) * 0.5);
        float intensity = 1.0 / length(vec2(
            p1.x / (sin(p1.y + t) / 0.01),
            p1.y / (cos(p1.x + t) / 0.01)
        ));
        caustic += intensity * 0.025;
    }

    // Layer 2: Medium detail
    vec2 p2 = uv * 5.0 + time * 0.1;
    for(int i = 0; i < 3; i++) {
        float t = time * 0.6 + float(i) * 0.7;
        p2 += vec2(sin(t + p2.y) * 0.3, cos(t - p2.x) * 0.3);
        caustic += (1.0 / length(p2)) * 0.015;
    }

    // Layer 3: Fine detail ripples
    vec2 p3 = uv * 8.0;
    float ripple = sin(p3.x * 5.0 + time) * cos(p3.y * 5.0 - time * 0.7);
    caustic += max(ripple, 0.0) * 0.2;

    return clamp(caustic, 0.0, 2.0);
}

// Procedural riverbed texture with rocks and sand
vec3 getRiverbedTexture(vec2 uv) {
    // Voronoi for rock distribution
    vec2 voronoiResult = voronoi(uv * 2.0);
    float rockMask = smoothstep(0.15, 0.25, voronoiResult.x);

    // Base colors
    vec3 sandColor = vec3(0.6, 0.52, 0.42);
    vec3 rockColor = vec3(0.25, 0.22, 0.18);
    vec3 darkRockColor = vec3(0.15, 0.13, 0.11);

    // Rock variation
    float rockVariation = voronoiResult.y;
    vec3 currentRockColor = mix(darkRockColor, rockColor, rockVariation);

    // Add rock texture detail
    float rockDetail = fbm(uv * 50.0, 3) * 0.2;
    currentRockColor += vec3(rockDetail);

    // Sand noise
    float sandNoise = fbm(uv * 20.0, 4);
    sandColor += vec3(sandNoise * 0.1 - 0.05);

    // Mix based on rock mask
    vec3 baseColor = mix(currentRockColor, sandColor, rockMask);

    // Add pebbles in sand
    float pebbles = smoothstep(0.7, 0.9, noise(uv * 40.0)) * rockMask;
    baseColor = mix(baseColor, rockColor * 0.8, pebbles * 0.3);

    // Sediment lines (flow patterns)
    float sediment = sin(uv.x * 30.0 + fbm(uv * 5.0, 2) * 2.0);
    sediment = smoothstep(0.8, 1.0, sediment) * rockMask;
    baseColor += vec3(0.1, 0.08, 0.06) * sediment * 0.2;

    return baseColor;
}

// Subsurface scattering approximation
vec3 subsurfaceScattering(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 baseColor) {
    float scatter = pow(max(dot(-lightDir, viewDir), 0.0), 2.0);
    scatter *= max(0.0, -dot(normal, lightDir));
    return baseColor * scatter * vec3(1.2, 0.9, 0.6) * 0.3;
}

// Ambient occlusion
float ambientOcclusion(vec3 normal, vec2 uv) {
    float ao = 0.3 + 0.7 * (normal.y * 0.5 + 0.5);

    // Add detail AO from texture
    float detailAO = 1.0 - fbm(uv * 10.0, 2) * 0.3;
    return ao * detailAO;
}

void main() {
    vec3 normal = normalize(vNormal);
    vec2 uv = vWorldPos.xz * 0.15;

    // Base riverbed texture
    vec3 baseColor = getRiverbedTexture(uv);

    // Lighting setup
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 viewDir = normalize(uCameraPos - vWorldPos);

    // Diffuse lighting with wraparound
    float NdotL = dot(normal, lightDir);
    float diffuse = max(NdotL * 0.7 + 0.3, 0.0);

    // Ambient occlusion
    float ao = ambientOcclusion(normal, uv);

    // Apply lighting
    vec3 color = baseColor * diffuse * ao;

    // Add subsurface scattering
    color += subsurfaceScattering(normal, lightDir, viewDir, baseColor);

    // Advanced caustics
    float caustic = advancedCaustics(uv, uTime * 0.4);
    vec3 causticColor = vec3(0.85, 0.95, 1.0) * caustic;

    // Caustics are modulated by depth and surface normal
    float causticStrength = max(normal.y, 0.2);
    color += causticColor * causticStrength * 0.5;

    // Underwater ambient (bluish tint)
    vec3 waterTint = vec3(0.08, 0.18, 0.28);
    color = mix(color, waterTint, 0.12);

    // Specular highlights on wet surfaces
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 64.0);

    // Wet rock specular (more on rocks, less on sand)
    vec2 voronoiCheck = voronoi(uv * 2.0);
    float wetness = smoothstep(0.3, 0.1, voronoiCheck.x);
    color += vec3(spec * wetness * 0.4);

    // Add God rays effect (volumetric light shafts approximation)
    float godRays = 0.0;
    vec2 rayUV = uv + uTime * 0.05;
    for(int i = 0; i < 3; i++) {
        float angle = float(i) * 2.094; // 120 degrees
        vec2 dir = vec2(cos(angle), sin(angle));
        float ray = max(0.0, sin(dot(rayUV, dir) * 5.0 + uTime * 2.0));
        godRays += ray;
    }
    godRays *= 0.1 / 3.0;
    color += vec3(0.8, 0.9, 1.0) * godRays * causticStrength;

    // Depth-based darkening (water absorption)
    float waterDepth = vWorldPos.y; // Assuming water surface is above
    float absorption = exp(waterDepth * 0.1);
    color *= absorption;

    // Color grading for underwater look
    color = pow(color, vec3(1.1)); // Slight contrast boost
    color *= vec3(0.95, 1.0, 1.05); // Cool tint

    FragColor = vec4(color, 1.0);
}
