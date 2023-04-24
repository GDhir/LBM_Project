#ifndef utils
#define utils "utils"

#define Q9 9
#define dim 2

constexpr int Ny = 256;
constexpr int Nx = 4096;

double calcVelError( double* ux, double* uy, double* uxprev, double* uyprev, double tol );

#endif