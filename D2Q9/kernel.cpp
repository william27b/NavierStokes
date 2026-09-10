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

extern "C" __global__
void fused(
    const double* f,
    double* f_out,
    double* rho,
    double* u,
    double* u_in,
    const unsigned char* solid,
    int nx,
    int ny,
    double tau)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    int N = nx * ny;
    int cell = y * nx + x;

    if (solid[cell]) return;

    double local_f[9];
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

    double total = 0.0;
    double mx = 0.0;
    double my = 0.0;

    for (int q = 0; q < 9; q++) {
        double fq = local_f[q];

        total += fq;
        mx += CX[q] * fq;
        my += CY[q] * fq;
    }

    double ux = mx / total;
    double uy = my / total;

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

        if (solid[dest_cell])
            f_out[OPP[q]*N + cell] = post;
        else
            f_out[q*N + dest_cell] = post;
    }
}