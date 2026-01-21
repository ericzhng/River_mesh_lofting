# Build Fix Instructions

## Issue
The CUDA compiler (NVCC) was trying to parse OpenGL headers, causing compilation errors.

## Solution
The particle system has been restructured into separate files:

### New File Structure
1. **`src/particle_system.cuh`** - Header with declarations
2. **`src/particle_kernels.cu`** - Pure CUDA device kernels
3. **`src/particle_system_host.cu`** - Host-side OpenGL interop (compiled with NVCC)

### Required Action
**DELETE old files** (OPTIONAL - already filtered out by CMake):
- `src/particle_system.cu`
- `src/particle_system.cpp`

These are replaced by the new files above. CMakeLists.txt filters them out automatically.

## Steps to Build

1. **Reconfigure CMake:**
   ```bash
   cd build
   cmake ..
   ```

2. **Build the project:**
   ```bash
   cmake --build . --config Release
   ```

3. **Optional - Delete old files for cleanliness:**
   ```bash
   # On Windows PowerShell:
   Remove-Item src\particle_system.cu, src\particle_system.cpp

   # On Windows CMD:
   del src\particle_system.cu src\particle_system.cpp

   # On Linux/Mac:
   rm src/particle_system.cu src/particle_system.cpp
   ```

## What Changed

### File Separation Strategy
- **CUDA kernels** (`.cu` files) - Pure device code, no OpenGL headers
- **C++ host code** (`.cpp` files) - OpenGL interop and host functions
- **Headers** (`.cuh` files) - Shared declarations with forward declarations

This separation prevents NVCC from seeing OpenGL headers, which it cannot properly parse.

### CMakeLists.txt Updates
- Added `CUDA::curand` library linking
- Added CUDA-specific compiler flags
- Enabled separable compilation for CUDA

## Verification

After building, you should see:
- No CUDA compilation errors about `WINGDIAPI`, `APIENTRY`, etc.
- Successful linking of all three components
- Executable ready to run

## If You Still Get Errors

1. Make sure `src/particle_system.cu` is **completely deleted**
2. Clean your build directory:
   ```bash
   rm -rf build/*
   cmake ..
   ```
3. Check that GLEW is installed via vcpkg
4. Verify CUDA Toolkit version is 11.0 or higher

## Files Modified
- `CMakeLists.txt` - Added CUDA flags, curand library, and filters
- `src/particle_system.cuh` - Header with CUDA types (includes curand_kernel.h)
- `src/particle_kernels.cu` - NEW: Pure CUDA device kernels
- `src/particle_system_host.cu` - NEW: Host-side OpenGL interop
- `src/particle_system.cu` - OLD: Delete (filtered by CMake)
- `src/particle_system.cpp` - OLD: Delete (filtered by CMake)
