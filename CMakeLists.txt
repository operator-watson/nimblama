cmake_minimum_required(VERSION 3.20)
project(Nimblama CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

# ---- Enable ccache ----
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ---- GPU backend toggles ----
option(GGML_CUDA "Enable ggml CUDA backend" ON)
option(GGML_CUDA_FORCE_CUBLAS "Force cuBLAS (instead of mmq kernels)" ON)

# ---- Suppress deprecated GPU targets warning ----
if(GGML_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
endif()

# ---- Preprocessor flags for your code ----
option(USE_CUDA_DEFINES "Define GGML_USE_CUBLAS / LLAMA_USE_CUDA for app code" ON)

# ---- llama.cpp vendored subproject ----
# Make sure llama.cpp sees the CUDA options
set(LLAMA_BUILD_CUDA ${GGML_CUDA} CACHE BOOL "Enable CUDA in llama.cpp")
set(LLAMA_FORCE_CUBLAS ${GGML_CUDA_FORCE_CUBLAS} CACHE BOOL "Force cuBLAS kernels")
add_subdirectory(extern/llama.cpp llama-build EXCLUDE_FROM_ALL)

# ---- Your executable ----
add_executable(nimblama
src/main.cpp
src/llm/llama_chat_app.cpp
)

# Include llama.cpp headers
target_include_directories(nimblama PRIVATE
    extern/llama.cpp
    src/llm
    )

# Pass CUDA macros to your code for print_cuda_status()
if (USE_CUDA_DEFINES)
    target_compile_definitions(nimblama PRIVATE
        $<$<BOOL:${GGML_CUDA}>:GGML_USE_CUBLAS>
        $<$<BOOL:${GGML_CUDA}>:LLAMA_USE_CUDA>
    )
endif()

# Link llama.cpp library
target_link_libraries(nimblama PRIVATE llama)

# Output binary to build/bin
set_target_properties(nimblama PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
