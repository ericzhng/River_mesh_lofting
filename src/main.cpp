#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "shader_utils.hpp"
#include "river_mesh.hpp"
#include "opengl_utils.hpp"
#include "simulation.cuh"

// --- Globals & Constants ---
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// Camera
glm::vec3 camera_pos = glm::vec3(0.0f, 15.0f, 40.0f);
glm::vec3 camera_front = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 5.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera_pos += cameraSpeed * camera_front;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera_pos -= cameraSpeed * camera_front;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera_pos -= glm::normalize(glm::cross(camera_front, camera_up)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera_pos += glm::normalize(glm::cross(camera_front, camera_up)) * cameraSpeed;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    // 1. Initialize GLFW and GLEW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CUDA River Simulation", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // 2. Load Shaders
    Shader riverBedShader("shaders/river_bed.vert", "shaders/river_bed.frag");
    Shader waterShader("shaders/water.vert", "shaders/water.frag");

    // 3. Define the 1D River Path
    std::vector<RiverNode> riverPath;
    const int numSegments = 100;
    for (int i = 0; i < numSegments; ++i) {
        float z = (float)i * 1.5f;
        float x = sin(i / 10.0f) * 5.0f; // Gentle curve
        riverPath.push_back({
            {x, 0.0f, z}, // Position
            8.0f,         // bottomWidth
            2.0f,         // bankSlope
            4.0f          // bankHeight
        });
    }

    // 4. Generate Meshes and Upload to GPU
    Mesh riverBedMesh = GenerateRiverBedMesh(riverPath);
    Mesh waterSurfaceMesh = GenerateWaterSurfaceMesh(riverPath);

    RenderObject riverBedRO, waterSurfaceRO;
    riverBedRO.UploadMesh(riverBedMesh);
    waterSurfaceRO.UploadMesh(waterSurfaceMesh);

    // 5. Initialize CUDA Simulation
    RiverSimulation sim;
    std::vector<State> initialState(numSegments);
    for (int i = 0; i < numSegments; ++i) {
        // Initial condition: calm water
        initialState[i] = {2.5f, 0.0f}; 
    }
    // Create a dam break initial condition
    for (int i = 0; i < numSegments / 4; ++i) {
        initialState[i] = { 5.0f, 0.0f };
    }
    sim.Initialize(numSegments, initialState);

    // 6. Setup CUDA-OpenGL Interop
    GLuint sim_tbo;
    glGenBuffers(1, &sim_tbo);
    glBindBuffer(GL_TEXTURE_BUFFER, sim_tbo);
    glBufferData(GL_TEXTURE_BUFFER, numSegments * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    
    GLuint sim_tex;
    glGenTextures(1, &sim_tex);
    glBindTexture(GL_TEXTURE_BUFFER, sim_tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, sim_tbo);
    
    cudaGraphicsResource* cuda_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_resource, sim_tbo, cudaGraphicsRegisterFlagsWriteDiscard);

    // 7. Render Loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        // Run CUDA Simulation
        sim.Step(cuda_resource);

        // Render
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 200.0f);
        glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
        glm::mat4 model = glm::mat4(1.0f);

        // Draw River Bed
        riverBedShader.use();
        riverBedShader.setMat4("uProjection", projection);
        riverBedShader.setMat4("uView", view);
        riverBedShader.setMat4("uModel", model);
        riverBedRO.Draw();

        // Draw Water Surface
        waterShader.use();
        waterShader.setMat4("uProjection", projection);
        waterShader.setMat4("uView", view);
        waterShader.setMat4("uModel", model);
        waterShader.setFloat("uBankSlope", 2.0f); // Should match RiverNode
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, sim_tex);
        waterShader.setInt("uSimData", 0);
        waterSurfaceRO.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 8. Cleanup
    cudaGraphicsUnregisterResource(cuda_resource);
    glfwTerminate();
    return 0;
}
