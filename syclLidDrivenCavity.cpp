#include <sycl/sycl.hpp>

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
constexpr Float tau_            = 3.f*(LID_VELOCITY*1.*Float(NX)/REYNOLDS_NUMBER)+.5f;
constexpr int32_t STEPS         = 40000;

template<typename T>
constexpr T sqr(const T val) { return val*val; }

#define xStart (1)
#define xEnd   (NX+1)
#define yStart (1)
#define yEnd   (NY+1)
#define NX_SIZE  (NX+2)
#define NY_SIZE  (NY+2)

Grid<NX, NY, Float[Q]> df1_       = {};//in
Grid<NX, NY, Float[Q]> df2_       = {};//out
Grid<NX, NY, Vec2>     velocity1_ = {};
Grid<NX, NY, Vec2>     velocity2_ = {};
Grid<NX, NY, Float>    density_   = {};

int nvidia_gpu_selector(const sycl::device &dev) {
    if (dev.has(sycl::aspect::gpu)) {
        auto vendorName = dev.get_info<sycl::info::device::vendor>();
        if (vendorName.find("NVIDIA") != std::string::npos) {
            return 1;
        }
    }
    return -1;
}

void save(sycl::buffer<Vec2, D> velocityBuffer) {
    sycl::host_accessor hostAcc{velocityBuffer};
    char fileName[256];
    memset(fileName, 0, sizeof(fileName));
    sprintf(fileName, "sycl_lbm_lid_cavity_%dx%d.pydat", NX, NY);
    FILE *pDatFile = fopen(fileName, "w");
    fprintf(pDatFile, "%d %d\n", NX, NY);
    for(int32_t x = xStart; x < xEnd; ++x) {
        for(int32_t y = yStart; y < yEnd; ++y) {
            fprintf(
                pDatFile,
                "%e ",
                std::sqrt(sqr(hostAcc[x][y].x) + sqr(hostAcc[x][y].y))
            );
        }
        fprintf(pDatFile, "\n");
    }
    fflush(pDatFile);
    fclose(pDatFile);
}

Float feq(const int32_t q, const Float density, const Vec2 vel) {
    Float cv = Float(CX[q])*vel.x + Float(CY[q])*vel.y;
    Float vv = sqr(vel.x) + sqr(vel.y);
    return WEIGHTS[q] * density * (1.f + 3.f*cv + 4.5f*sqr(cv) - 1.5f*vv);
}

double calculateMLUPS(double nx, double ny, double steps, double time) {
    return (nx*ny*steps)/(time*1.e6);
}

void latticeBoltzmannMethod() {
    // sycl::queue queue{nvidia_gpu_selector, sycl::property::queue::in_order()};
    sycl::queue queue{sycl::property::queue::in_order()};
    sycl::buffer<Float, D> densityBuffer {&density_[0][0], sycl::range{NX_SIZE, NY_SIZE}};
    sycl::buffer<Vec2, D>  velocity1Buffer {&velocity1_[0][0], sycl::range{NX_SIZE, NY_SIZE}};
    sycl::buffer<Vec2, D>  velocity2Buffer {&velocity2_[0][0], sycl::range{NX_SIZE, NY_SIZE}};
    sycl::buffer<Float, 3> df1Buffer {&df1_[0][0][0], sycl::range{Q, NX_SIZE, NY_SIZE}};
    sycl::buffer<Float, 3> df2Buffer {&df2_[0][0][0], sycl::range{Q, NX_SIZE, NY_SIZE}};

    // init
    queue.submit([&](sycl::handler &cgh) {
        // init
        sycl::accessor density{densityBuffer, cgh};
        sycl::accessor velocity1{velocity1Buffer, cgh};
        sycl::accessor velocity2{velocity2Buffer, cgh};
        cgh.parallel_for(sycl::range{NX_SIZE, NY_SIZE}, [density, velocity1, velocity2](auto item) {
            auto x = item[0], y = item[1];

            density[x][y]        = RHO0;

            velocity1[x][y].x    = 0;
            velocity1[x][y].y    = 0;
            velocity1[x][NY+1].x = LID_VELOCITY;

            velocity2[x][y].x    = 0;
            velocity2[x][y].y    = 0;
        });
    });

    // init
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor df{df1Buffer, cgh};
        sycl::accessor density{densityBuffer, cgh};
        sycl::accessor velocity{velocity1Buffer, cgh};
        cgh.parallel_for(sycl::range{NX_SIZE, NY_SIZE}, [df, density, velocity](auto item) {
            auto x = item[0], y = item[1];

            for(int q = 0; q < Q; q++) {
                df[q][x][y] = feq(q, density[x][y], velocity[x][y]);
            }
        });
    });

    auto start = std::chrono::high_resolution_clock::now();
    for(int32_t step = 0; step < STEPS; ++step) {
        // collision step
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor density{densityBuffer, cgh};
            sycl::accessor velocity{velocity1Buffer, cgh};
            sycl::accessor src{df1Buffer, cgh};
            sycl::accessor dst{df2Buffer, cgh};

            cgh.parallel_for(sycl::range{NX, NY}, [src, dst, density, velocity](auto item) {
                auto x = xStart + item[0], y = yStart + item[1];
                for(int32_t q = 0; q < Q; ++q) {
                    int32_t nx   = x - CX[q];
                    int32_t ny   = y - CY[q];
                    dst[q][x][y] = src[q][nx][ny] + (feq(q, density[nx][ny], velocity[nx][ny])-src[q][nx][ny])/tau_;
                }
            });
        });

        // macroscopic step
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor df{df2Buffer, cgh};
            sycl::accessor density{densityBuffer, cgh};
            sycl::accessor velocity{velocity2Buffer, cgh};

            cgh.parallel_for(sycl::range{NX, NY}, [df, density, velocity](auto item) {
                auto x = xStart + item[0], y = yStart + item[1];

                Float rho        = 0;
                Vec2  vel        = {0, 0};
                for(int32_t q = 0; q < Q; ++q) {
                    Float f  = df[q][x][y];
                    rho     += f;
                    vel.x   += Float(CX[q])*f;
                    vel.y   += Float(CY[q])*f;
                }

                density[x][y]     = rho;
                velocity[x][y].x = vel.x/rho;
                velocity[x][y].y = vel.y/rho;
            });
        });

        // boundary condition
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor df{df2Buffer, cgh};
            sycl::accessor density{densityBuffer, cgh};
            sycl::accessor velocity{velocity2Buffer, cgh};

            cgh.parallel_for(sycl::range{NY}, [df, density, velocity](auto gid) {
                int32_t y = yStart + gid;

                density[0][y]    = density[0][y];
                density[NX+1][y] = density[NX][y];

                for(int32_t q = 0; q < Q; ++q) {
                    df[q][NX+1][y] = feq(q, density[NX+1][y], velocity[NX+1][y]) + df[q][NX][y] - feq(q, density[NX][y], velocity[NX][y]);
                    df[q][0][y]    = feq(q, density[0][y], velocity[0][y]) + df[q][1][y] - feq(q, density[1][y], velocity[1][y]);
                }
            });
        });

        // boundary condition
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor df{df2Buffer, cgh};
            sycl::accessor density{densityBuffer, cgh};
            sycl::accessor velocity{velocity2Buffer, cgh};

            cgh.parallel_for(sycl::range{NX_SIZE}, [df, density, velocity](auto x) {
                density[x][0]        = density[x][1];
                density[x][NY+1]     = density[x][NY];
                velocity[x][NY+1].x  = LID_VELOCITY;

                for(int32_t q = 0; q < Q; ++q) {
                    df[q][x][0]    = feq(q, density[x][0], velocity[x][0]) + df[q][x][1] - feq(q, density[x][1], velocity[x][1]);
                    df[q][x][NY+1] = feq(q, density[x][NY+1], velocity[x][NY+1]) + df[q][x][NY] - feq(q, density[x][NY], velocity[x][NY]);
                }
            });
        });

        std::swap(velocity1Buffer, velocity2Buffer);
        std::swap(df1Buffer, df2Buffer);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())/1.e9;
    printf("MLUPS %f, %fs\n", calculateMLUPS(NX, NY, STEPS, time), time);
    fflush(stdout);
    save(velocity2Buffer);
}

int main(int argc, char *argv[]) {
    latticeBoltzmannMethod();

    return 0;
}