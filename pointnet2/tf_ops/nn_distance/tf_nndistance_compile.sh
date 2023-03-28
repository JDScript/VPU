#!/usr/bin/env bash

TF_LINK_FLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_ROOT=$1

nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64 -shared -fPIC -I ${TF_INC} --std=c++17 ${TF_LINK_FLAGS} -O2 -D_GLIBCXX_USE_CXX11_ABI=1