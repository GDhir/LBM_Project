cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(lbm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

set(LBM_SRC src/mainfile.cpp src/CUDACodes/cudalbm.cu
            src/SerialCodes/seriallbmv1.cpp)

set(LBM_INC include/seriallbm.hpp )

add_executable(lbm-gpu ${LBM_INC} ${LBM_SRC})
target_include_directories(lbm-gpu PUBLIC include)
