cmake_minimum_required(VERSION 3.22) # 3.1 << C_STANDARD 11

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project(neural LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} main.cpp
        inc/Layer.h
        inc/Neuron.h
        inc/NeuralNetwork.h
        src/NeuralNetwork.cpp
        inc/input_parse.h
        src/input_parse.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=100)
# Use libc++ when compiling with Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()
