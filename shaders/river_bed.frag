#version 430 core

in vec3 vNormal;
in vec3 vWorldPos;

uniform float uTime;

out vec4 FragColor;

// Procedural caustics effect
float caustics(vec2 uv, float time) {
    vec2 p = uv * 5.0;

    // Layer 1: Primary caustic pattern
    vec2 i = vec2(p);
    float c = 1.0;
    float inten = 0.005;

    for (int n = 0; n < 5; n++) {
        float t = time * (1.0 - (3.5 / float(n + 1)));
        i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
        c += 1.0 / length(vec2(p.x / (sin(i.x + t) / inten),
                               p.y / (cos(i.y + t) / inten)));
    }
    c /= float(5);
    c = 1.17 - pow(c, 1.4);

    // Layer 2: Secondary caustic for detail
    vec2 p2 = uv * 7.0 + vec2(0.5);
    float c2 = 1.0;
    for (int n = 0; n < 3; n++) {
        float t = time * 0.8 * (1.0 - (3.0 / float(n + 1)));
        p2 += vec2(cos(t + p2.y), sin(t + p2.x));
        c2 += 1.0 / length(vec2(p2.x / (sin(p2.y + t) / inten),
                                 p2.y / (cos(p2.x + t) / inten)));
    }
    c2 /= float(3);
    c2 = 1.17 - pow(c2, 1.4);

    // Combine layers
    float caustic = pow(c * c2, 2.0) * 0.5;
    caustic = clamp(caustic, 0.0, 1.0);

    return caustic;
}

// Procedural rock/sand texture
vec3 getRiverbedTexture(vec2 uv) {
    // Base sand color
    vec3 sandColor = vec3(0.55, 0.47, 0.37);
    vec3 rockColor = vec3(0.3, 0.25, 0.2);

    // Simple noise pattern for variation
    float pattern = sin(uv.x * 10.0) * cos(uv.y * 10.0);
    pattern += sin(uv.x * 25.0 + 1.57) * cos(uv.y * 25.0);
    pattern = pattern * 0.5 + 0.5;

    // Mix between sand and rock
    vec3 baseColor = mix(sandColor, rockColor, smoothstep(0.4, 0.6, pattern));

    // Add detail variation
    float detail = sin(uv.x * 50.0) * cos(uv.y * 50.0) * 0.1;
    baseColor += vec3(detail);

    return baseColor;
}

void main() {
    vec3 normal = normalize(vNormal);

    // Base riverbed texture
    vec2 uv = vWorldPos.xz * 0.2;
    vec3 baseColor = getRiverbedTexture(uv);

    // Sunlight direction
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));

    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);

    // Ambient occlusion (darker in crevices)
    float ao = 0.3 + 0.7 * normal.y;

    // Apply basic lighting
    vec3 color = baseColor * (diff * 0.7 + 0.3) * ao;

    // Add animated caustics (light patterns through water)
    float causticPattern = caustics(uv, uTime * 0.5);
    vec3 causticColor = vec3(0.8, 0.9, 1.0) * causticPattern;
    color += causticColor * 0.4;

    // Underwater ambient light (bluish tint)
    vec3 waterTint = vec3(0.1, 0.2, 0.3);
    color = mix(color, waterTint, 0.15);

    // Add subtle specular on wet rocks
    vec3 viewDir = normalize(vec3(0.0, 1.0, 0.0)); // Simplified
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);
    color += vec3(spec * 0.2);

    FragColor = vec4(color, 1.0);
}
