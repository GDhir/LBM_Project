cmake_minimum_required(VERSION 3.16)
project(LBM LANGUAGES CUDA CXX)


add_executable(serialLidDrivenCavity serialLidDrivenCavity.cpp)

macro(add_application target gridSize SOURCE_FILES)
    add_executable(${target}-${gridSize} ${SOURCE_FILES})
    set_target_properties(${target}-${gridSize} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "70;80")
    target_compile_definitions(${target}-${gridSize} PUBLIC NX=${gridSize} NY=${gridSize})
endmacro()

add_application(cudaLidDrivenCavity 128  cudaLidDrivenCavity.cu)
add_application(cudaLidDrivenCavity 256  cudaLidDrivenCavity.cu)
add_application(cudaLidDrivenCavity 512  cudaLidDrivenCavity.cu)
add_application(cudaLidDrivenCavity 1024 cudaLidDrivenCavity.cu)
add_application(cudaLidDrivenCavity 2048 cudaLidDrivenCavity.cu)
