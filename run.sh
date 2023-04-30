#!/usr/bin/bash

rm -f *.pydat
rm -f *.png

if [ ! -d "cmake-build-release" ]; then
    mkdir "cmake-build-release"
fi

source ./venv/bin/activate
cmake -DCMAKE_BUILD_TYPE=Release -G  "Unix Makefiles" -S ./ -B ./cmake-build-release
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

echo ""
echo "Running Lid-driven Cavity serial code:"

./cmake-build-release/serialLidDrivenCavity
python3 display.py serial_lbm_lid_cavity.pydat
