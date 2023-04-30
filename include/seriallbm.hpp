#ifndef seriallbm
#define seriallbm "seriallbm"

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <math.h>
#include "utils.hpp"

void calcMacroscopic_AOS(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey);

void calcMacroscopic_SOA(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey);

void performStreamPushOut(double *fvals, double *ftemp, double *ex, double *ey);

void performLBMStepsPullIn_AOS(double *fvals, double *fvalsprev, double *feq, double *ex, double *ey, double tau, double g);

void performLBMStepsPullIn_SOA(double *fvals, double *fvalsprev, double *feq, double *ex, double *ey, double tau, double g);

void performLBMPullIn_AOS(double *fvals,  double *fvalsprev, double *feq, double *rho, double *ux, double *uy, double* uxprev, double* uyprev, double *ex, double *ey, double g, double tau, int szf, int Niter, double tol);

void calcEqDis_AOS(double *feq, double *rho, double *ux, double *uy, double g, double tau);

void calcEqDis_SOA(double *feq, double *rho, double *ux, double *uy, double g, double tau);

void collide(double *f, double *ftemp, double *feq, double tau);

void applyBC(double *f, double *ftemp);

double calcError(double *val, double *valprev);

void performLBMPushOut(double *fvals, double *rho, double *ux, double *uy, double *ex, double *ey, double g, double tau, int szf, int Niter);

void printu(double *ux, double *uy, std::string fname);

void printval(double *val, std::string fname);

void printf(double *val, std::string fname);

void setInitialVelocity(double *ux, double *uy, double U);

#endif