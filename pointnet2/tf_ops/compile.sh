#!/usr/bin/env bash

CUDA_ROOT=/usr/local/cuda

cd nn_distance
source tf_nndistance_compile.sh $CUDA_ROOT
cd ../approxmatch/
source tf_approxmatch_compile.sh $CUDA_ROOT
cd ../grouping/
source tf_grouping_compile.sh $CUDA_ROOT
cd ../interpolation/
source tf_interpolate_compile.sh $CUDA_ROOT
cd ../renderball
source compile_render_balls_so.sh
cd ../sampling
source tf_sampling_compile.sh $CUDA_ROOT
cd ..

