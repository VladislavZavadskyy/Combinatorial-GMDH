cmake_minimum_required (VERSION 3.8)
project (CUDA-GMDH)
set (CUDA-GMDH_VERSION_MAJOR 0)
set (CUDA-GMDH_VERSION_MINOR 0)

find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30
    )

# Specify include directories
include_directories(
    include
     /opt/intel/mkl/include
    )

# Specify library paths
link_directories(
    /opt/intel/mkl/lib/intel64/
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    app
    include/dependencies.cuh
    include/ops.cuh
    include/utils.cuh
    include/matrix.cuh
    include/tests.cuh
    src/ops.cu
    src/matrix.cu
    src/utils.cu
    src/tests.cu
    src/main.cu
    )

target_link_libraries(
    app
    -lcusolver
    -lcublas
    -lmkl_intel_ilp64
    -lmkl_sequential
    -lmkl_core
    )
