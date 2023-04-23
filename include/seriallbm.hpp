#ifndef seriallbm
#define seriallbm "seriallbm"

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <math.h>

#define Q9 9
#define dim 2

constexpr int Ny = 12;
constexpr int Nx = 100;

void calcMacroscopic(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey);

void performStreamPushOut(double *fvals, double *ftemp, double *ex, double *ey);

void performLBMStepsPullIn(double *fvals, double *ex, double *ey, double tau, double g);

void performLBMPullIn(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey, double g, double tau, int szf, int Niter);

void calcEqDis(double *feq, double *rho, double *ux, double *uy, double g, double tau);

void collide(double *f, double *ftemp, double *feq, double tau);

void applyBC(double *f, double *ftemp);

double calcError(double *val, double *valprev);

void performLBMPushOut(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey, double g, double tau, int szf, int Niter);

void printu(double *ux, double *uy, std::string fname);

void printval(double *val, std::string fname);

void printf(double *val, std::string fname);

void setInitialVelocity(double *ux, double *uy, double U);

#endif