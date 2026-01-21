cmake_minimum_required(VERSION 3.21)

project(RiverMeshLofting LANGUAGES CXX CUDA)

# --- Instructions for vcpkg ---
# To build with libraries from vcpkg (GLEW, GLFW), you MUST configure
# CMake with the vcpkg toolchain file. In your IDE settings or when
# running cmake from the command line, add this argument:
#
#   -DCMAKE_TOOLCHAIN_FILE="C:/path/to/your/vcpkg/scripts/buildsystems/vcpkg.cmake"
#
# Replace the path with the actual location of your vcpkg installation.
# In VS Code, set this in the "cmake.configureArgs" setting.

# --- Set Standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture automatically. For broader compatibility, you could
# also set this explicitly, e.g., "75;86" for Ampere and Turing.
set(CMAKE_CUDA_ARCHITECTURES native)

# --- Find Dependencies ---
# Find CUDA Toolkit to ensure CUDA::cudart target is available
find_package(CUDAToolkit REQUIRED)

# Find GLEW and GLFW3 (provided by vcpkg)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Add GLM from the vendor directory. This creates the 'glm' target.
add_subdirectory(vendor/glm)

# --- Project Sources ---
# Gather all source and header files from the src directory.
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE HEADERS "src/*.hpp" "src/*.cuh")

# Remove old particle files (replaced by particle_kernels.cu + particle_system_host.cu)
list(FILTER SOURCES EXCLUDE REGEX ".*particle_system\\.cu$")
list(FILTER SOURCES EXCLUDE REGEX ".*particle_system\\.cpp$")

# --- Executable Target ---
add_executable(river_mesh_lofting ${SOURCES} ${HEADERS})

# --- Include Directories ---
# The only include path we need to add manually is for our own source directory.
# Include paths for GLEW, GLFW, GLM, and CUDA are handled automatically
# by linking their respective CMake targets.
target_include_directories(river_mesh_lofting PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# --- CUDA Specific Settings ---
# Set CUDA flags to properly handle device code compilation
set_target_properties(river_mesh_lofting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA compiler flags to handle OpenGL headers properly
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(river_mesh_lofting PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --extended-lambda
            -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
        >
    )
endif()

# Add CUDA include directories
target_include_directories(river_mesh_lofting PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# --- Link Libraries ---
# Link against all required libraries. Using the imported targets
# (GLEW::glew, glfw, glm) is the modern way and handles dependencies correctly.
target_link_libraries(river_mesh_lofting PRIVATE
    # OpenGL libraries
    GLEW::glew
    glfw

    # Math library
    glm

    # CUDA runtime library
    CUDA::cudart
    CUDA::curand
)

# --- Platform-Specific Settings ---
if(WIN32)
    # This definition is required by GLEW on Windows when linking statically.
    # vcpkg typically provides a static build of GLEW. If you get linker
    # errors about __glew..., it is likely installed as a shared library and
    # this definition should be removed or commented out.
    # target_compile_definitions(river_mesh_lofting PRIVATE GLEW_STATIC)
endif()

# --- Post-Build Actions ---
# Copy the shaders directory to the executable's output directory.
# This uses a post-build command which is aware of the build configuration
# (e.g., Debug, Release) and ensures the shaders are in the right place.
add_custom_command(TARGET river_mesh_lofting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/shaders"
    "$<TARGET_FILE_DIR:river_mesh_lofting>/shaders"
    COMMENT "Copying shaders to output directory..."
)

# --- User-Friendly Messages ---
message(STATUS "Project '${PROJECT_NAME}' configured successfully.")
message(STATUS "Build with: cmake --build .")
message(STATUS "Run the executable from its output directory (e.g., build/Debug or build/Release).")
