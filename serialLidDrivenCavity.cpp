#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

/**
 * Lattice Boltzmann Method for simulating fluid dynamics at mesoscopic level
 *
 * Problem        : Lid-driven Cavity
 * Grid           : [128, 128]
 * Model          : D2Q9
 * Fluid density  : 1 kg/m^3
 * Reynolds number: 1000.
 * Lid velocity   : 0.1m/s
 */

// Lid Cavity
//
//  (1, NY)   .1m/s   (NX, NY)
//        >>>>>>>>>>>>>
//        |           |
//        |           |
//        |           |
//        |           |
//        |           |
//        -------------
//   (1, 1)           (NX, 1)
//

//  D2Q9 model indexing:
//
//   6    2    5
//    \   |   /
//     \  |  /
//      \ | /
//   3--- 0 ---1
//      / | \
//     /  |  \
//    /   |   \
//   7    4    8
//

#ifdef SINGLE_PRECISION
using Float = float;
#else
using Float = double;
#endif

// Add ghost cells to the grid, hence the +2
template<int32_t DimX, int32_t DimY, typename T = Float>
using Grid = T[DimX+2][DimY+2];

struct Vec2 {
    Float x;
    Float y;
};

#ifndef NX
constexpr int32_t NX            = 128;
#endif
#ifndef NY
constexpr int32_t NY            = 128;
#endif
constexpr int32_t D             = 2;
constexpr int32_t Q             = 9;
constexpr Float WEIGHTS[]       = {4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36};
constexpr int8_t CX[]           = {   0,    1,    0,   -1,    0,     1,    -1,    -1, 1};
constexpr int8_t CY[]           = {   0,    0,    1,    0,   -1,     1,     1,    -1, -1};
constexpr Float RHO0            = 1.;
constexpr Float LID_VELOCITY    = .1;
constexpr Float REYNOLDS_NUMBER = 1000.;
constexpr int32_t STEPS         = 40000;

template<typename T>
constexpr T sqr(const T val) { return val*val; }

#define xStart (1)
#define xEnd   (NX+1)
#define yStart (1)
#define yEnd   (NY+1)

Float                  tau_       = 0.;
Grid<NX, NY, Float[Q]> df1_       = {};//in
Grid<NX, NY, Float[Q]> df2_       = {};//out
Grid<NX, NY, Vec2>     velocity1_ = {};
Grid<NX, NY, Vec2>     velocity2_ = {};
Grid<NX, NY, Float>    density_   = {};

/**
 * Output tecplot file or 2D matrix
 * @param step
 */
void save() {
#ifdef TECPLOT
    FILE *pDatFile = fopen(("lbm_lid_cavity_" + std::to_string(step) + ".dat").c_str(), "w");
    fprintf(pDatFile, "Title= \"LBM Lid Driven Flow\"\n");
    fprintf(pDatFile, "VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"UV\", \"rho\"\n");
    fprintf(pDatFile, "ZONE T=\"ZONE 1\", I=%d, J=%d, F=POINT\n", NX, NY);
    for(int32_t y = yStart; y < yEnd; y++) {
        for(int32_t x = xStart; x < xEnd; x++) {
            fprintf(pDatFile, "%d %d %e %e %e %e\n", x-xStart, y-yStart, velocity1_[x][y].x, velocity1_[x][y].y, std::sqrt(sqr(velocity1_[x][y].x) + sqr(velocity1_[x][y].y)), density_[x][y]);
        }
    }
    fflush(pDatFile);
    fclose(pDatFile);
#else
    FILE *pDatFile = fopen("serial_lbm_lid_cavity.pydat", "w");
    fprintf(pDatFile, "%d %d\n", NX, NY);
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            fprintf(
                    pDatFile,
                    "%e ",
                    std::sqrt(sqr(velocity1_[x][y].x) + sqr(velocity1_[x][y].y))
            );
        }
        fprintf(pDatFile, "\n");
    }
    fflush(pDatFile);
    fclose(pDatFile);
#endif
}

Float feq(const int32_t q, const Float density, const Vec2 vel) {
    Float cv = Float(CX[q])*vel.x + Float(CY[q])*vel.y;
    Float vv = sqr(vel.x) + sqr(vel.y);
    return WEIGHTS[q] * density * (1.f + 3.f*cv + 4.5f*sqr(cv) - 1.5f*vv);
}

Float error(const Grid<NX, NY, Vec2> &newVel, const Grid<NX, NY, Vec2> &oldVel) {
    Float l2Norm = 0;
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            l2Norm += (sqr(newVel[x][y].x - oldVel[x][y].x) + sqr(newVel[x][y].y - oldVel[x][y].y));
        }
    }
    l2Norm = std::sqrt(l2Norm);
    return l2Norm;
}

void init() {
    Float dx  = 1.;
//    Float dy  = 1.;
    Float Lx  = dx*Float(NX);
//    Float Ly  = dy*Float(NY);
//    Float dt  = dx;
//    Float c   = dx/dt; //1.0
    Float nu  = LID_VELOCITY*Lx/REYNOLDS_NUMBER;
    tau_      = 3.f*nu+.5f;

    std::memset(df1_, 0, sizeof(df1_));
    std::memset(df2_, 0, sizeof(df2_));
    std::memset(density_, 0, sizeof(density_));
    std::memset(velocity1_, 0, sizeof(velocity1_));
    std::memset(velocity2_, 0, sizeof(velocity2_));

    for(int32_t x = 0; x < NX+2; ++x) {
        velocity1_[x][NY+1].x = LID_VELOCITY;
        for(int32_t y = 0; y < NY+2; ++y) {
            density_[x][y] = RHO0;
            for(int32_t q = 0; q < Q; q++) {
                df1_[x][y][q] = feq(q, density_[x][y], velocity1_[x][y]);
            }
        }
    }
}

/**
 * Streams (pull) and apply collision operator
 * @param src grid distribution function
 * @param dst grid distribution function
 * @param density grid fluid density
 * @param velocity grid fluid velocity
 */
void collisionStep(const Grid<NX, NY, Float[Q]> &src, Grid<NX, NY, Float[Q]> &dst, const Grid<NX, NY, Float> &density, const Grid<NX, NY, Vec2> &velocity) {
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            for(int32_t q = 0; q < Q; ++q) {
                int32_t nx   = x - CX[q];
                int32_t ny   = y - CY[q];
                dst[x][y][q] = src[nx][ny][q] + (feq(q, density[nx][ny], velocity[nx][ny])-src[nx][ny][q])/tau_;
            }
        }
    }
}

/**
 * Calculate Macroscopic Properties like density and velocity
 * @param df grid distribution function
 * @param density grid fluid density
 * @param velocity grid fluid velocity
 */
void calculateMacroscopicProperties(const Grid<NX, NY, Float[Q]> &df, Grid<NX, NY, Float> &density, Grid<NX, NY, Vec2> &velocity) {
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            Float rho        = 0;
            velocity[x][y].x = 0;
            velocity[x][y].y = 0;

            for(int32_t q = 0; q < Q; ++q) {
                rho              += df[x][y][q];
                velocity[x][y].x += Float(CX[q])*df[x][y][q];
                velocity[x][y].y += Float(CY[q])*df[x][y][q];
            }

            density[x][y]     = rho;
            velocity[x][y].x /= rho;
            velocity[x][y].y /= rho;
        }
    }
}

/**
 * Apply boundary conditions specific to lid driven cavity problem.
 * @param df grid distribution function
 * @param density grid fluid density
 * @param velocity grid fluid velocity
 */
void applyBoundaryCondition(Grid<NX, NY, Float[Q]> &df, Grid<NX, NY, Float> &density, Grid<NX, NY, Vec2> &velocity) {
    // left and right boundaries
    for(int32_t y = yStart; y < yEnd; ++y) {
        density[NX+1][y] = density[NX][y];
        density[0][y]    = density_[1][y];

        for(int32_t q = 0; q < Q; ++q) {
            df[NX+1][y][q] = feq(q, density[NX+1][y], velocity[NX+1][y]) + df[NX][y][q] - feq(q, density[NX][y], velocity[NX][y]);
            df[0][y][q]    = feq(q, density[0][y], velocity[0][y]) + df[1][y][q] - feq(q, density[1][y], velocity[1][y]);
        }
    }

    // top and bottom boundaries
    for(int32_t x = 0; x < NX+2; ++x) {
        density[x][0]       = density[x][1];
        density[x][NY+1]    = density[x][NY];
        velocity[x][NY+1].x = LID_VELOCITY;

        for(int32_t q = 0; q < Q; ++q) {
            df[x][0][q]    = feq(q, density[x][0], velocity[x][0]) + df[x][1][q] - feq(q, density[x][1], velocity[x][1]);
            df[x][NY+1][q] = feq(q, density[x][NY+1], velocity[x][NY+1]) + df[x][NY][q] - feq(q, density[x][NY], velocity[x][NY]);
        }
    }
}

/**
 * Lattice Boltzmann Method
 */
void latticeBoltzmannMethod() {
    init();
    auto start = std::chrono::high_resolution_clock::now();
    for(int32_t step = 0; step < STEPS; ++step) {
        collisionStep(df1_, df2_, density_, velocity1_);
        calculateMacroscopicProperties(df2_, density_, velocity2_);
        applyBoundaryCondition(df2_, density_, velocity2_);
        std::swap(df1_, df2_);
        std::swap(velocity1_, velocity2_);
        if(step%1000 == 0) {
            printf("\r[STEP %d]", step);
            fflush(stdout);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    printf("\r\nError %e\n", error(velocity2_, velocity1_));
    auto time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())/1.e9;
    printf("MLUPS %f, %fs\n", double(STEPS*NX*NY)/(time*1.e6), time);
    fflush(stdout);
    save();
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    latticeBoltzmannMethod();

    return 0;
}
