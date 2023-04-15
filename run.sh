#!/usr/bin/bash

rm -f lbm_lid_cavity_*.pydat

if [ ! -d "cmake-build-release" ]; then
    mkdir "cmake-build-release"
fi

source ./venv/bin/activate
cmake -DCMAKE_BUILD_TYPE=Release -G  "Unix Makefiles" -S ./ -B ./cmake-build-release
cmake --build ./cmake-build-release --target clean -j 3
cmake --build ./cmake-build-release --target serialLidDrivenCavity -j 3

echo ""
echo "Running Lid-driven Cavity serial code:"

./cmake-build-release/serialLidDrivenCavity
python3 display.py ./lbm_lid_cavity_0.pydat
