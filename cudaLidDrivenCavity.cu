#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
//#include <cuda/std/cmath>
//#include <vector_types.h>

//  D2Q9 model indexing:
//
//   6    2    5
//    \   |   /
//     \  |  /
//      \ | /
//   7--- 0 ---1
//      / | \
//     /  |  \
//    /   |   \
//   8    4    9
//

using Float = double;
using Vec2 = double2;
//using Float = float;
//using Vec2  = float2;

constexpr int32_t NX            = 128;
constexpr int32_t NY            = 128;
//constexpr int32_t D             = 2;
constexpr int32_t Q             = 9;
__constant__ Float WEIGHTS[]    = {4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36};
__constant__ int8_t CX[]        = {   0,    1,    0,   -1,    0,     1,    -1,    -1, 1};
__constant__ int8_t CY[]        = {   0,    0,    1,    0,   -1,     1,     1,    -1, -1};
constexpr Float RHO0            = 1.;
constexpr Float LID_VELOCITY    = .1;
constexpr Float REYNOLDS_NUMBER = 1000.;
constexpr Float tau_            = 3.f*(LID_VELOCITY*1.*Float(NX)/REYNOLDS_NUMBER)+.5f;
constexpr int32_t ITERS         = 40000;

template<typename T>
constexpr T sqr(const T val) { return val*val; }

#define xStart   (1)
#define xEnd     (NX+1)
#define yStart   (1)
#define yEnd     (NY+1)
#define NX_SIZE  (NX+2)
#define NY_SIZE  (NY+2)

//Float   tau_       = 0.;
Float *pDf1_       = nullptr;
Float *pDf2_       = nullptr;
Float *pDensity_   = nullptr;
Vec2  *pVelocity1_ = nullptr;
Vec2  *pVelocity2_ = nullptr;

__device__ Float feq(const int32_t q, const Float density, const Vec2 vel) {
    Float cv = Float(CX[q])*vel.x + Float(CY[q])*vel.y;
    Float vv = vel.x*vel.x + vel.y*vel.y;
    return WEIGHTS[q] * density * (1. + 3.f*cv + 4.5f*cv*cv - 1.5f*vv);
}

__global__ void init1(Float *pDensity, Vec2 *pVelocity1, Vec2 *pVelocity2) {
    uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if(NX_SIZE <= x or NY_SIZE <= y) return;

    uint32_t xy = x*NY_SIZE + y;

    // intialize values
    pDensity[xy]                   = RHO0;

    pVelocity1[xy].x               = 0;
    pVelocity1[xy].y               = 0;
    pVelocity1[x*NY_SIZE + yEnd].x = LID_VELOCITY;

    pVelocity2[xy].x               = 0;
    pVelocity2[xy].y               = 0;
}

__global__ void init2(Float *pDf, Float *pDensity, Vec2 *pVelocity) {
    uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if(NX_SIZE <= x or NY_SIZE <= y) return;

    uint32_t xy = x*NY_SIZE + y;

    for(int q = 0; q < Q; q++) {
        pDf[q*NX_SIZE*NY_SIZE + xy] = feq(q, pDensity[xy], pVelocity[xy]);
    }
}

/**
 * Streams (pull) and apply collision operator
 * @param pSrc grid distribution function
 * @param pDst grid distribution function
 * @param pDensity grid fluid density
 * @param pVelocity grid fluid velocity
 */
__global__ void collisionKernel(Float *pSrc /**[9][NX+2][NY+2]**/, Float *pDst /**[9][NX+2][NY+2]**/, Float *pDensity /**[NX+2][NY+2]**/, Vec2 *pVelocity /**[NX+2][NY+2]**/) {
    uint32_t x = xStart + blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t y = yStart + blockIdx.x * blockDim.x + threadIdx.x;
    if(xEnd <= x  or yEnd <= y) return;

    uint32_t xy = x*NY_SIZE + y;

    for(int32_t q = 0; q < Q; ++q) {
        uint32_t nx  = x - CX[q];
        uint32_t ny  = y - CY[q];
        uint32_t nxy = nx*NY_SIZE + ny;

        Float f                      = pSrc[q*NX_SIZE*NY_SIZE + nxy];
        pDst[q*NX_SIZE*NY_SIZE + xy] = f + (feq(q, pDensity[nxy], pVelocity[nxy]) - f)/tau_;
    }
}

/**
 * Calculate Macroscopic Properties like density and velocity
 * @param pDf grid distribution function
 * @param pDensity grid fluid density
 * @param pVelocity grid fluid velocity
 */
__global__ void calculateMacroscopicPropertiesKernel(Float *pDf /**[9][NX+2][NY+2]**/, Float *pDensity /**[NX+2][NY+2]**/, Vec2 *pVelocity /**[NX+2][NY+2]**/) {
    uint32_t x = xStart + blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t y = yStart + blockIdx.x * blockDim.x + threadIdx.x;
    if(xEnd <= x  or yEnd <= y) return;

    uint32_t xy = x*NY_SIZE + y;

    Float rho        = 0;
    Vec2  vel        = {0, 0};
    for(int32_t q = 0; q < Q; ++q) {
        Float df = pDf[q*NX_SIZE*NY_SIZE + xy];
        rho     += df;
        vel.x   += Float(CX[q])*df;
        vel.y   += Float(CY[q])*df;
    }

    pDensity[xy]    = rho;
    pVelocity[xy].x = vel.x/rho;
    pVelocity[xy].y = vel.y/rho;
}

/**
 * Apply left and right boundary conditions specific to the lid driven cavity problem.
 * @param pDf grid distribution function
 * @param pDensity grid fluid density
 * @param pVelocity grid fluid velocity
 */
__global__ void boundaryConditionKernel1(Float *pDf /**[9][NX+2][NY+2]**/, Float *pDensity /**[NX+2][NY+2]**/, Vec2 *pVelocity /**[NX+2][NY+2]**/) {
    uint32_t y = yStart + blockIdx.x * blockDim.x + threadIdx.x;
    if(yEnd <= y) return;

    uint32_t rightGhostXy    = (NX+1)*NY_SIZE + y;
    uint32_t rightBoundaryXy = rightGhostXy - NY_SIZE;
    uint32_t leftGhostXy     = 0*NY_SIZE + y;
    uint32_t leftBoundaryXy  = leftGhostXy + NY_SIZE;

    pDensity[rightGhostXy] = pDensity[rightBoundaryXy];
    pDensity[leftGhostXy]  = pDensity[leftBoundaryXy];
    for(int32_t q = 0; q < Q; ++q) {
        // right
        pDf[q*NX_SIZE*NY_SIZE + rightGhostXy] =
                feq(q, pDensity[rightGhostXy], pVelocity[rightGhostXy]) +
                pDf[q*NX_SIZE*NY_SIZE + rightBoundaryXy] - feq(q, pDensity[rightBoundaryXy], pVelocity[rightBoundaryXy]);

        // left
        pDf[q*NX_SIZE*NY_SIZE + leftGhostXy] =
                feq(q, pDensity[leftGhostXy], pVelocity[leftGhostXy]) +
                pDf[q*NX_SIZE*NY_SIZE + leftBoundaryXy] - feq(q, pDensity[leftBoundaryXy], pVelocity[leftBoundaryXy]);
    }
}

/**
 * Apply top and bottom boundary conditions specific to the lid driven cavity problem.
 * @param pDf grid distribution function
 * @param pDensity grid fluid density
 * @param pVelocity grid fluid velocity
 */
__global__ void boundaryConditionKernel2(Float *pDf /**[9][NX+2][NY+2]**/, Float *pDensity /**[NX+2][NY+2]**/, Vec2 *pVelocity /**[NX+2][NY+2]**/) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if(NX_SIZE <= x) return;

    uint32_t bottomGhostXy     = x*NY_SIZE + 0;
    uint32_t bottomBoundaryXy  = bottomGhostXy + 1;
    uint32_t topGhostXy        = x*NY_SIZE + NY+1;
    uint32_t topBoundaryXy     = topGhostXy - 1;

    pDensity[bottomGhostXy] = pDensity[bottomBoundaryXy];
    pDensity[topGhostXy]    = pDensity[topBoundaryXy];
    pVelocity[topGhostXy].x = LID_VELOCITY;
    for(int32_t q = 0; q < Q; ++q) {
        // bottom
        pDf[q*NX_SIZE*NY_SIZE + bottomGhostXy] =
                feq(q, pDensity[bottomGhostXy], pVelocity[bottomGhostXy]) +
                pDf[q*NX_SIZE*NY_SIZE + bottomBoundaryXy] - feq(q, pDensity[bottomBoundaryXy], pVelocity[bottomBoundaryXy]);

        // top
        pDf[q*NX_SIZE*NY_SIZE + topGhostXy] =
                feq(q, pDensity[topGhostXy], pVelocity[topGhostXy]) +
                pDf[q*NX_SIZE*NY_SIZE + topBoundaryXy] - feq(q, pDensity[topBoundaryXy], pVelocity[topBoundaryXy]);
    }
}

void cudaError_(cudaError_t error, const char *pFile, const int32_t lineNo) {
    if(error == cudaError::cudaSuccess) return;
    printf("[ERROR][CUDA] %s:%d %s\n", pFile, lineNo, cudaGetErrorString(error));
}
#define cudaError(x) cudaError_(x, __FILE__, __LINE__)

void save(Vec2 velocity[NX_SIZE][NY_SIZE]) {
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
    FILE *pDatFile = fopen("cuda_lbm_lid_cavity.pydat", "w");
    fprintf(pDatFile, "%d %d\n", NX, NY);
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            fprintf(
                    pDatFile,
                    "%e ",
                    std::sqrt(sqr(velocity[x][y].x) + sqr(velocity[x][y].y))
            );
        }
        fprintf(pDatFile, "\n");
    }
    fflush(pDatFile);
    fclose(pDatFile);
#endif
}

void latticeBoltzmannMethod() {
    cudaError(cudaMalloc(&pDf1_, Q*NX_SIZE*NY_SIZE*sizeof(Float)));
    cudaError(cudaMalloc(&pDf2_, Q*NX_SIZE*NY_SIZE*sizeof(Float)));
    cudaError(cudaMalloc(&pDensity_, NX_SIZE*NY_SIZE*sizeof(Float)));
    cudaError(cudaMalloc(&pVelocity1_, NX_SIZE*NY_SIZE*sizeof(Vec2)));
    cudaError(cudaMalloc(&pVelocity2_, NX_SIZE*NY_SIZE*sizeof(Vec2)));

    dim3 block(32, 32);
    dim3 grid(std::ceil(Float(NX_SIZE)/Float(block.x)), std::ceil(Float(NY_SIZE)/Float(block.y)));
    init1<<<grid, block>>>(pDensity_, pVelocity1_, pVelocity2_);
#ifndef NDEBUG
    Float density[NX_SIZE][NY_SIZE];
    Vec2  velocity[NX_SIZE][NY_SIZE];
    cudaMemcpy(density, pDensity_, NX_SIZE*NY_SIZE*sizeof(Float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(velocity, pVelocity1_, NX_SIZE*NY_SIZE*sizeof(Vec2), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for(int32_t x = 0; x < NX_SIZE; ++x) {
        for(int32_t y = 0; y < NY_SIZE; ++y) {
            printf("%2.1f ", density[x][y]);
        }
        printf("\n");
    }

    for(int32_t x = 0; x < NX_SIZE; ++x) {
        for(int32_t y = 0; y < NY_SIZE; ++y) {
            printf("(%2.1f, %2.1f) ", velocity[x][y].x, velocity[x][y].y);
        }
        printf("\n");
    }
#endif

    init2<<<grid, block>>>(pDf1_, pDensity_, pVelocity1_);
#ifndef NDEBUG
    Float df[9][NX_SIZE][NY_SIZE];
    cudaMemcpy(df, pDf1_, Q*NX_SIZE*NY_SIZE*sizeof(Float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    for(int32_t x = 0; x < NX_SIZE; ++x) {
        for(int32_t y = 0; y < NY_SIZE; ++y) {
            printf("(");
            for(int32_t q = 0; q < Q; ++q) {
                printf("%4.2f, ", df[q][x][y]);
            }
            printf("\b\b) ");
        }
        printf("\n");
    }
#endif

    const dim3 gridXY   = {uint32_t(std::ceil(Float(NX)/Float(block.x))), uint32_t(std::ceil(Float(NY)/Float(block.y))), 1};
    const dim3 gridXSYS = {uint32_t(std::ceil(Float(NX_SIZE)/Float(block.x))), uint32_t(std::ceil(Float(NY_SIZE)/Float(block.y))), 1};

    auto start = std::chrono::high_resolution_clock::now();
    for(int32_t step = 0; step < ITERS ; ++step) {
        collisionKernel<<<gridXY, block>>>(pDf1_, pDf2_, pDensity_, pVelocity1_);
        calculateMacroscopicPropertiesKernel<<<gridXY, block>>>(pDf2_, pDensity_, pVelocity2_);
        boundaryConditionKernel1<<<gridXY.y, block.y>>>(pDf2_, pDensity_, pVelocity2_);
        boundaryConditionKernel2<<<gridXSYS.x, block.x>>>(pDf2_, pDensity_, pVelocity2_);
        std::swap(pDf1_, pDf2_);
        std::swap(pVelocity1_, pVelocity2_);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())/1.e9;
    printf("MLUPS %f, %fs\n", double(ITERS*NX*NY)/(time*1.e6), time);

    Vec2 vel[NX_SIZE][NY_SIZE];
    cudaMemcpy(vel, pVelocity2_, NX_SIZE*NY_SIZE*sizeof(Vec2), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    save(vel);

    cudaError(cudaFree((void*)pDf1_));
    cudaError(cudaFree((void*)pDf2_));
    cudaError(cudaFree((void*)pDensity_));
    cudaError(cudaFree((void*)pVelocity1_));
    cudaError(cudaFree((void*)pVelocity2_));
}



int main(int argc, char *argv[]) {
    cudaSetDevice(3);
    latticeBoltzmannMethod();

    return 0;
}