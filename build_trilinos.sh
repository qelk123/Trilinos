#!/bin/bash

BUILD_TYPE=RELEASE

# set following paths according to the env！！

TRILINOS_SOURCE=/home/v-yinuoliu/code/Trilinos
TRILINOS_BUILD=/home/v-yinuoliu/code/Trilinos/build
TRILINOS_INSTALL=/home/v-yinuoliu/code/Trilinos/install
# MPI support
MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
# BLAS lib
TPL_BLAS_LIBRARIES_PATH=/usr/lib/x86_64-linux-gnu/libblas.so.3
# LAPACK lib
TPL_LAPACK_LIBRARIES_PATH=/usr/lib/x86_64-linux-gnu/liblapack.so.3

# set Kokkos_ARCH_XXX to you gpu compute capacity ！！

cmake \
  -D CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
  -D CMAKE_INSTALL_PREFIX:STRING=${TRILINOS_INSTALL} \
  -D CMAKE_CXX_FLAGS=" -O3 -g -Wall -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-inline -Wshadow -I${MPI_ROOT}/include" \
  -D CMAKE_CXX_STANDARD:STRING=17 \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
  \
  -D Trilinos_ASSERT_MISSING_PACKAGES:BOOL=ON \
  -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
  -D Trilinos_ENABLE_Fortran:BOOL=OFF \
  -D Trilinos_ENABLE_OpenMP:BOOL=ON \
  -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
  -D Trilinos_EXTRA_LINK_FLAGS:STRING="-L${MPI_ROOT}/lib -lmpi -fopenmp" \
  \
  -D Trilinos_ENABLE_Belos:BOOL=ON \
  -D Trilinos_ENABLE_Galeri:BOOL=ON \
  -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
  -D Trilinos_ENABLE_Epetra:BOOL=ON \
  -D Trilinos_ENABLE_Xpetra:BOOL=OFF \
  -D Trilinos_ENABLE_Triutils=ON \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
  -D Trilinos_ENABLE_Kokkos:BOOL=ON \
    -D Kokkos_ENABLE_OPENMP:BOOL=ON \
    -D Kokkos_ENABLE_THREADS:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE:BOOL=ON \
    -D Kokkos_ENABLE_TESTS:BOOL=OFF \
    -D Kokkos_ARCH_AMPERE80=ON \
  \
  -D Trilinos_ENABLE_Tpetra:BOOL=ON \
    -D Tpetra_INST_SERIAL:BOOL=ON \
    -D Tpetra_INST_CUDA:BOOL=ON \
    -D Tpetra_INST_OPENMP:BOOL=ON \
    -D Tpetra_ASSUME_GPU_AWARE_MPI=ON \
  \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
    -D TPL_BLAS_LIBRARIES:FILEPATH=${TPL_BLAS_LIBRARIES_PATH} \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
    -D TPL_LAPACK_LIBRARIES:FILEPATH=${TPL_LAPACK_LIBRARIES_PATH} \
  \
  \
  -D MPI_BASE_DIR="${MPI_ROOT}" \
  \
  -D CMAKE_CXX_COMPILER=${TRILINOS_SOURCE}/packages/kokkos/bin/nvcc_wrapper \
  -D CMAKE_C_COMPILER=$(which mpicc) \
  -D TPL_ENABLE_CUDA:BOOL=ON \
  -D TPL_ENABLE_CUSPARSE:BOOL=ON \
  \
  ${TRILINOS_SOURCE}
