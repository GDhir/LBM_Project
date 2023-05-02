#!/usr/bin/bash

rm -f *.pydat
rm -f *.png

if [ ! -d "cmake-build-release" ]; then
    mkdir "cmake-build-release"
fi

source ./venv/bin/activate
cmake -DCMAKE_BUILD_TYPE=Release -G  "Unix Makefiles" -S ./ -B ./cmake-build-release -DSINGLE_PRECISION=1
cmake --build ./cmake-build-release --target clean -j 4
cmake --build ./cmake-build-release --target all -j 4

echo ""
echo "Running Lid-driven Cavity cuda code:"

./cmake-build-release/cudaLidDrivenCavity-128
if [ -f "cuda_lbm_lid_cavity_128x128.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_128x128.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-256
if [ -f "cuda_lbm_lid_cavity_256x256.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_256x256.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-512
if [ -f "cuda_lbm_lid_cavity_512x512.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_512x512.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-1024
if [ -f "cuda_lbm_lid_cavity_1024x1024.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_1024x1024.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-2048
if [ -f "cuda_lbm_lid_cavity_2048x2048.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_2048x2048.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-4096
if [ -f "cuda_lbm_lid_cavity_4096x4096.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_4096x4096.pydat
fi

./cmake-build-release/cudaLidDrivenCavity-8192
if [ -f "cuda_lbm_lid_cavity_8192x8192.pydat" ]; then
  python3 display.py cuda_lbm_lid_cavity_8192x8192.pydat
fi


echo ""
echo "Running Lid-driven Cavity cuda code using Shared Memory:"

./cmake-build-release/cudaSmLidDrivenCavity-128
if [ -f "cuda_sm_lbm_lid_cavity_128x128.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_128x128.pydat
fi

./cmake-build-release/cudaSmLidDrivenCavity-256
if [ -f "cuda_sm_lbm_lid_cavity_256x256.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_256x256.pydat
fi

./cmake-build-release/cudaSmLidDrivenCavity-512
if [ -f "cuda_sm_lbm_lid_cavity_512x512.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_512x512.pydat
fi

./cmake-build-release/cudaSmLidDrivenCavity-1024
if [ -f "cuda_sm_lbm_lid_cavity_1024x1024.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_1024x1024.pydat
fi

./cmake-build-release/cudaSmLidDrivenCavity-2048
if [ -f "cuda_sm_lbm_lid_cavity_2048x2048.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_2048x2048.pydat
fi

./cmake-build-release/cudaSmLidDrivenCavity-4096
if [ -f "cuda_sm_lbm_lid_cavity_4096x4096.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_4096x4096.pydat
fi


./cmake-build-release/cudaSmLidDrivenCavity-8192
if [ -f "cuda_sm_lbm_lid_cavity_8192x8192.pydat" ]; then
  python3 display.py cuda_sm_lbm_lid_cavity_8192x8192.pydat
fi


echo ""
echo "Running Lid-driven Cavity sycl code:"

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=128 -DNY=128 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-128
./syclLidDrivenCavity-128
python3 display.py sycl_lbm_lid_cavity_128x128.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=256 -DNY=256 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-256
./syclLidDrivenCavity-256
python3 display.py sycl_lbm_lid_cavity_256x256.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=512 -DNY=512 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-512
./syclLidDrivenCavity-512
python3 display.py sycl_lbm_lid_cavity_512x512.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=1024 -DNY=1024 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-1024
./syclLidDrivenCavity-1024
python3 display.py sycl_lbm_lid_cavity_1024x1024.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=2048 -DNY=2048 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-2048
./syclLidDrivenCavity-2048
python3 display.py sycl_lbm_lid_cavity_2048x2048.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=4096 -DNY=4096 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-4096
./syclLidDrivenCavity-4096
python3 display.py sycl_lbm_lid_cavity_4096x4096.pydat

clang++ syclLidDrivenCavity.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DNX=8192 -DNY=8192 -DSINGLE_PRECISION=1 -Wno-unknown-cuda-version -o syclLidDrivenCavity-8192
./syclLidDrivenCavity-8192
python3 display.py sycl_lbm_lid_cavity_8192x8192.pydat


echo ""
echo "Running Lid-driven Cavity serial code:"

./cmake-build-release/serialLidDrivenCavity
python3 display.py serial_lbm_lid_cavity.pydat
