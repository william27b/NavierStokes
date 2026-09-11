__device__ const int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__device__ const int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__device__ const double W[9] = {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
__device__ const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__device__ inline double equilibrium(
    int q, double density, double ux, double uy)
{
    double cu = CX[q]*ux + CY[q]*uy;
    double speed2 = ux*ux + uy*uy;

    return W[q] * density *
        (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed2);
}

// Shared by time stepping and read-only state export, so boundary labels use
// the same reconstruction as the next collision.
__device__ inline void prepare_cell(
    const double* f, const double* u_in, int cell, int x, int y,
    int nx, int ny, double* local_f, double& total, double& ux, double& uy)
{
    int N = nx * ny;
    for (int q = 0; q < 9; q++) {
        local_f[q] = f[q*N + cell];
    }

    if (x == 0) {
        double inlet_ux = u_in[y];
        double inlet_uy = u_in[ny + y];

        double inlet_rho = (
            local_f[0] + local_f[2] + local_f[4]
            + 2.0 * (local_f[3] + local_f[6] + local_f[7])
        ) / (1.0 - inlet_ux);

        local_f[1] = local_f[3]
            + (2.0 / 3.0) * inlet_rho * inlet_ux;

        local_f[5] = local_f[7]
            + 0.5 * (local_f[4] - local_f[2])
            + inlet_rho * inlet_ux / 6.0
            + 0.5 * inlet_rho * inlet_uy;

        local_f[8] = local_f[6]
            + 0.5 * (local_f[2] - local_f[4])
            + inlet_rho * inlet_ux / 6.0
            - 0.5 * inlet_rho * inlet_uy;
    }

    if (x == nx - 1) {
        double outlet_rho = 1.0;  // Sets outlet pressure to 1/3.
        double outlet_uy = 0.0;

        double outlet_ux = (
            local_f[0] + local_f[2] + local_f[4]
            + 2.0 * (local_f[1] + local_f[5] + local_f[8])
        ) / outlet_rho - 1.0;

        local_f[3] = local_f[1]
            - (2.0 / 3.0) * outlet_rho * outlet_ux;

        local_f[6] = local_f[8]
            + 0.5 * (local_f[4] - local_f[2])
            - outlet_rho * outlet_ux / 6.0
            + 0.5 * outlet_rho * outlet_uy;

        local_f[7] = local_f[5]
            + 0.5 * (local_f[2] - local_f[4])
            - outlet_rho * outlet_ux / 6.0
            - 0.5 * outlet_rho * outlet_uy;
    }

    total = 0.0;
    double mx = 0.0;
    double my = 0.0;

    for (int q = 0; q < 9; q++) {
        double fq = local_f[q];

        total += fq;
        mx += CX[q] * fq;
        my += CY[q] * fq;
    }

    ux = mx / total;
    uy = my / total;

    if (x == 0 || x == nx - 1) {
        // Retain the non-equilibrium second moment (viscous stress), removing
        // higher kinetic moments at open boundaries. Bulk collision stays BGK.
        // Latt & Chopard, arXiv:physics/0506157, Eq. (10), cs^2 = 1/3.
        double pxx = 0.0, pxy = 0.0, pyy = 0.0;
        for (int q = 0; q < 9; q++) {
            double neq = local_f[q] - equilibrium(q, total, ux, uy);
            pxx += CX[q] * CX[q] * neq;
            pxy += CX[q] * CY[q] * neq;
            pyy += CY[q] * CY[q] * neq;
        }
        for (int q = 0; q < 9; q++) {
            double reg = 4.5 * W[q] * ((CX[q]*CX[q] - 1.0/3.0)*pxx
                + 2.0*CX[q]*CY[q]*pxy + (CY[q]*CY[q] - 1.0/3.0)*pyy);
            local_f[q] = equilibrium(q, total, ux, uy) + reg;
        }
    }
}

extern "C" __global__
void fused(
    const double* f, double* f_out, double* rho, double* u,
    double* u_in, const unsigned char* solid, int nx, int ny, double tau)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int N = nx * ny;
    int cell = y * nx + x;
    if (solid[cell]) return;

    double local_f[9], total, ux, uy;
    prepare_cell(f, u_in, cell, x, y, nx, ny, local_f, total, ux, uy);

    rho[cell] = total;
    u[cell] = ux;
    u[N + cell] = uy;

    for (int q = 0; q < 9; q++) {
        double fq = local_f[q];
        double feq = equilibrium(q, rho[cell], ux, uy);

        double post = fq - (fq - feq) / tau;

        int source_x = x - CX[q];

        if (source_x < 0 || source_x >= nx)
            f_out[q*N + cell] = post;

        int dest_x = x + CX[q];
        int dest_y = y + CY[q];

        if (dest_x < 0 || dest_x >= nx) continue;

        int dest_cell = dest_y * nx + dest_x;

        if (dest_y < 0 || dest_y >= ny || solid[dest_cell])
            f_out[OPP[q]*N + cell] = post;
        else
            f_out[q*N + dest_cell] = post;
    }
}

// Reconstruct a separate snapshot. Reading fields must never advance the flow
// or modify the populations used by the following time step.
extern "C" __global__
void reconstruct_state(
    const double* f, double* corrected_f, double* rho, double* u,
    const double* u_in, const unsigned char* solid, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int N = nx * ny;
    int cell = y * nx + x;
    if (solid[cell]) {
        rho[cell] = 1.0;
        u[cell] = 0.0;
        u[N + cell] = 0.0;
        for (int q = 0; q < 9; q++) corrected_f[q*N + cell] = W[q];
        return;
    }
    double local_f[9], total, ux, uy;
    prepare_cell(f, u_in, cell, x, y, nx, ny, local_f, total, ux, uy);
    rho[cell] = total;
    u[cell] = ux;
    u[N + cell] = uy;
    for (int q = 0; q < 9; q++) corrected_f[q*N + cell] = local_f[q];
}
