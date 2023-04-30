#ifndef utils
#define utils "utils"

#include <string>

#define Q9 9
#define dim 2

constexpr int Ny = 256;
constexpr int Nx = 8192;

const std::string root_dir = "/uufs/chpc.utah.edu/common/home/u1444601/CS6235/LBM_Project/";

double calcVelError( double* ux, double* uy, double* uxprev, double* uyprev, double tol );

#endif