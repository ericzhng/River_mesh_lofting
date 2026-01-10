#pragma once

#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstddef> // For offsetof

#include "river_mesh.hpp" // Needs Vertex definition

class RenderObject {
    GLuint VAO, VBO, EBO;
    GLsizei indexCount;
    GLenum usage;

public:
    RenderObject() : VAO(0), VBO(0), EBO(0), indexCount(0), usage(GL_STATIC_DRAW) {}

    // Cleanup
    ~RenderObject() {
        if (VAO) glDeleteVertexArrays(1, &VAO);
        if (VBO) glDeleteBuffers(1, &VBO);
        if (EBO) glDeleteBuffers(1, &EBO);
    }

    void UploadMesh(const Mesh& mesh, GLenum usage = GL_STATIC_DRAW) {
        this->usage = usage;
        indexCount = (GLsizei)mesh.indices.size();

        if (VAO == 0) {
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);
        }

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex), mesh.vertices.data(), usage);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);

        GLsizei stride = sizeof(Vertex);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(Vertex, pos));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(Vertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(Vertex, uv));
        
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(Vertex, simIndex));

        glBindVertexArray(0);
    }

    void Draw() {
        if (VAO == 0) return;
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    GLuint GetVBO() const { return VBO; }
};
