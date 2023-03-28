#!/usr/bin/env bash

TF_LINK_FLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_ROOT=$1

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64 -shared -fPIC -I ${TF_INC} --std=c++17 ${TF_LINK_FLAGS} -O2 -D_GLIBCXX_USE_CXX11_ABI=1