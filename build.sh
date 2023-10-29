#!/bin/bash

BUILD_TYPE=RELEASE

TRILINOS_SOURCE=/home/v-yinuoliu/code/Trilinos
TRILINOS_BUILD=/home/v-yinuoliu/code/Trilinos/build
TRILINOS_INSTALL=/home/v-yinuoliu/code/Trilinos/install
export MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi

/home/v-yinuoliu/ENV/cmake-3.27.1-linux-x86_64/bin/cmake \
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
  \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D TPL_ENABLE_BLAS:BOOL=ON \
    -D TPL_BLAS_LIBRARIES:FILEPATH=/usr/lib/x86_64-linux-gnu/libblas.so.3 \
  -D TPL_ENABLE_LAPACK:BOOL=ON \
    -D TPL_LAPACK_LIBRARIES:FILEPATH=/usr/lib/x86_64-linux-gnu/liblapack.so.3 \
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


#-DHAVE_BELOS_TRIUTILS
# -D Teuchos_ORDINAL_TYPE:STRING="long long" \
# -D KokkosKernels_INST_ORDINAL_INT64_T=ON \
# LIST (APPEND Tpetra_ETI_LORDS "long long")
