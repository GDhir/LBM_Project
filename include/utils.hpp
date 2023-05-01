#ifndef utils
#define utils "utils"

#include <string>
#include<cmath>
#include<iostream>

#define Q9 9
#define dim 2

#define BLOCKSIZE 256

constexpr int Ny = 256;
constexpr int Nx = 8192;

constexpr double ex[] = {   0,    1,    0,   -1,    0,     1,    -1,    -1, 1};
constexpr double ey[] = {   0,    0,    1,    0,   -1,     1,     1,    -1, -1};

constexpr int szf = Ny * Nx * Q9;
constexpr int sz = Nx * Ny;

const std::string root_dir = "/uufs/chpc.utah.edu/common/home/u1444601/CS6235/LBM_Project/";

double calcVelError( double* ux, double* uy, double* uxprev, double* uyprev, double tol );

void accuracyTest(double *ux, double *uy, double *uxd, double *uyd, int sz);

#endif