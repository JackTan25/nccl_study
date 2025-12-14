cmake_minimum_required(VERSION 3.18)
project(nccl_study LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDAToolkit_LIBRARY_DIR}")

# 查找 MPI
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_INCLUDE_DIRS}")

# 查找 NCCL
# NCCL 通常在 CUDA 目录下或者单独安装
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        /usr/local/cuda/include
        /usr/local/cuda/targets/x86_64-linux/include
        ENV NCCL_ROOT
    PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
        ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64
)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found! Please set NCCL_ROOT or install NCCL.")
endif()

message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")

# 创建 NCCL imported target
add_library(NCCL::NCCL SHARED IMPORTED)
set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION ${NCCL_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR}
)

# ============ Demo 可执行文件 ============
# 添加 nccl_chap1 子目录中的 demo
add_executable(demo ${CMAKE_SOURCE_DIR}/nccl_chap1/demo.cpp)

# 设置 CUDA 架构 (可根据你的 GPU 调整)
# A10 是 Ampere 架构, sm_86
set_target_properties(demo PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

# 链接库
target_link_libraries(demo
    PRIVATE
        CUDA::cudart
        MPI::MPI_CXX
        NCCL::NCCL
)

target_include_directories(demo
    PRIVATE
        ${NCCL_INCLUDE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
)

# ============ 安装规则 ============
install(TARGETS demo DESTINATION bin)

# ============ 自定义运行目标 ============
# 添加方便运行的目标
add_custom_target(run
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} --allow-run-as-root
            $<TARGET_FILE:demo>
    DEPENDS demo
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running demo with 2 MPI processes"
)

# 显示配置信息
message(STATUS "")
message(STATUS "=== NCCL Study Build Configuration ===")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "MPI Implementation: ${MPI_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=======================================")

