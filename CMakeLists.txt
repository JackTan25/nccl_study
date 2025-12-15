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

# ============ 官方示例 (offical_examples) ============
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/nccl_chap1/offical_examples)

# Example 1: 单进程、单线程、多设备 (不需要 MPI)
add_executable(example1 ${EXAMPLES_DIR}/example1.cpp)
set_target_properties(example1 PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(example1 PRIVATE CUDA::cudart NCCL::NCCL)
target_include_directories(example1 PRIVATE ${NCCL_INCLUDE_DIR})

# Example 2: 每进程一个设备 (需要 MPI)
add_executable(example2 ${EXAMPLES_DIR}/example2.cpp)
set_target_properties(example2 PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(example2 PRIVATE CUDA::cudart MPI::MPI_CXX NCCL::NCCL)
target_include_directories(example2 PRIVATE ${NCCL_INCLUDE_DIR} ${MPI_CXX_INCLUDE_DIRS})

# Example 3: 每线程多个设备 (需要 MPI)
add_executable(example3 ${EXAMPLES_DIR}/example3.cpp)
set_target_properties(example3 PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(example3 PRIVATE CUDA::cudart MPI::MPI_CXX NCCL::NCCL)
target_include_directories(example3 PRIVATE ${NCCL_INCLUDE_DIR} ${MPI_CXX_INCLUDE_DIRS})

# Example 4: 每设备多个通信器 (需要 MPI)
add_executable(example4 ${EXAMPLES_DIR}/example4.cpp)
set_target_properties(example4 PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(example4 PRIVATE CUDA::cudart MPI::MPI_CXX NCCL::NCCL)
target_include_directories(example4 PRIVATE ${NCCL_INCLUDE_DIR} ${MPI_CXX_INCLUDE_DIRS})

# Example Reduce Verify: 规约操作验证 (不需要 MPI)
add_executable(example_reduce_verify ${EXAMPLES_DIR}/example_reduce_verify.cpp)
set_target_properties(example_reduce_verify PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
target_link_libraries(example_reduce_verify PRIVATE CUDA::cudart NCCL::NCCL)
target_include_directories(example_reduce_verify PRIVATE ${NCCL_INCLUDE_DIR})

# ============ 安装规则 ============
install(TARGETS demo example1 example2 example3 example4 example_reduce_verify DESTINATION bin)

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

# 运行官方示例
add_custom_target(run_example1
    COMMAND $<TARGET_FILE:example1>
    DEPENDS example1
    COMMENT "Running example1 (single process, 4 GPUs)"
)

add_custom_target(run_example2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} --allow-run-as-root
            $<TARGET_FILE:example2>
    DEPENDS example2
    COMMENT "Running example2 with 4 MPI processes"
)

add_custom_target(run_example3
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} --allow-run-as-root
            $<TARGET_FILE:example3>
    DEPENDS example3
    COMMENT "Running example3 with 2 MPI processes (2 GPUs each)"
)

add_custom_target(run_example4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} --allow-run-as-root
            $<TARGET_FILE:example4>
    DEPENDS example4
    COMMENT "Running example4 with 4 MPI processes"
)

add_custom_target(run_reduce_verify
    COMMAND $<TARGET_FILE:example_reduce_verify>
    DEPENDS example_reduce_verify
    COMMENT "Running reduce operations verification (4 GPUs)"
)

# 显示配置信息
message(STATUS "")
message(STATUS "=== NCCL Study Build Configuration ===")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "MPI Implementation: ${MPI_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=======================================")

