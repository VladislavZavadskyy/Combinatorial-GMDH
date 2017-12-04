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
    dependencies.cuh
    ops.cuh
    utils.cuh
    matrix.cuh
    tests.cuh
    ops.cu
    matrix.cu
    utils.cu
    tests.cu
    main.cu
    )

target_link_libraries(
    app
    -lcusolver
    -lcublas
    -lmkl_intel_ilp64
    -lmkl_sequential
    -lmkl_core
    )
