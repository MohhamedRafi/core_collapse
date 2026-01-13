// navier_stokes_2d_hllc_poisson.cpp
// 2D compressible Navier–Stokes (explicit viscosity) + self-gravity via periodic Poisson solve
// Finite-volume: HLLC for inviscid fluxes + explicit viscous stress fluxes + gravity source terms
//
// IMPORTANT: Output format matches euler2d_gravity_hllc.cpp, PLUS extra fields so diagnostics plots work.
// Single HDF5 file: out_ns2d_poisson.h5 containing
//   /x    (Nx) cell centers
//   /y    (Ny) cell centers
//   /t    (Nt) snapshot times
//   /rho  (Nt,Ny,Nx)
//   /P    (Nt,Ny,Nx)
//   /Mach (Nt,Ny,Nx)
//
// Additional datasets (to enable momentum/energy diagnostics in viz_h5.py):
//   /u    (Nt,Ny,Nx)  primitive velocity u
//   /v    (Nt,Ny,Nx)  primitive velocity v
//   /momx (Nt,Ny,Nx)  conserved momentum rho*u
//   /momy (Nt,Ny,Nx)  conserved momentum rho*v
//   /Phi  (Nt,Ny,Nx)  gravitational potential (periodic Poisson, mean mode removed)
//
// Poisson (periodic):
//   ∇^2 Phi = 4πG (ρ - <ρ>)
// solved spectrally with 2D FFT:
//   Phi_k = -(4πG ρ_k)/|k|^2 , Phi_{k=0}=0
// g = -∇Phi via centered differences
//
// Build:
//   clang++ -O3 -std=c++17 navier_stokes_2d_hllc_poisson.cpp -lhdf5 -o navier_stokes_2d_hllc_poisson
//
// Notes:
// - Nx,Ny must be powers of two for the FFT in this file.
// - No heat conduction (only viscosity).
// - Periodic boundaries assumed (for Poisson + finite-volume indexing).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <complex>
#include <sstream>

#include "hdf5.h"

static inline double sqr(double x){ return x*x; }
static inline double clamp_min(double x, double lo){ return (x < lo) ? lo : x; }
static inline bool is_pow2(int n){ return (n > 0) && ((n & (n-1)) == 0); }

struct Params {
    double gamma = 1.4;
    double cfl   = 0.35;

    // domain
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;

    int Nx = 128;
    int Ny = 128;

    // simulation control
    double t_end    = 10;
    double dt_store = 5e-3;

    unsigned seed = 42;

    // floors
    double floor_rho = 1e-10;
    double floor_P   = 1e-10;

    // viscosity (Navier–Stokes)
    bool viscosity_on = true;
    double mu = 1e-6;        // dynamic viscosity
    double cfl_visc = 0.05;  // explicit diffusion stability

    // gravity (Poisson)
    bool gravity_on = true;
    double G = 10.0;

    // output
    std::string out_h5 = "out_ns2d_poisson.h5";
};

struct U4 { // conserved: [rho, rho*u, rho*v, E]
    double d, mx, my, E;
};
struct W4 { // primitive: [rho, u, v, P]
    double d, u, v, P;
};

struct Grid2D {
    int Nx, Ny;
    double x0, x1, y0, y1;
    double dx, dy;
    double Lx, Ly;
    std::vector<double> x, y; // centers
    Grid2D(int Nx_, int Ny_, double x0_, double x1_, double y0_, double y1_)
    : Nx(Nx_), Ny(Ny_), x0(x0_), x1(x1_), y0(y0_), y1(y1_) {
        Lx = x1 - x0; Ly = y1 - y0;
        dx = Lx / Nx;
        dy = Ly / Ny;
        x.resize(Nx);
        y.resize(Ny);
        for(int i=0;i<Nx;i++) x[i] = x0 + (i + 0.5)*dx;
        for(int j=0;j<Ny;j++) y[j] = y0 + (j + 0.5)*dy;
    }
};

static inline int imod(int a, int n){
    int r = a % n;
    return (r < 0) ? (r + n) : r;
}
static inline int idx(int i, int j, int Nx){ return i + Nx*j; }

static inline W4 cons_to_prim(const U4& U, const Params& p){
    W4 W;
    W.d = clamp_min(U.d, p.floor_rho);
    W.u = U.mx / W.d;
    W.v = U.my / W.d;

    const double ke = 0.5 * W.d * (W.u*W.u + W.v*W.v);
    double eint = U.E - ke;

    double P = (p.gamma - 1.0) * eint;
    W.P = clamp_min(P, p.floor_P);

    return W;
}

static inline U4 prim_to_cons(const W4& W, const Params& p){
    U4 U;
    const double d = clamp_min(W.d, p.floor_rho);
    const double P = clamp_min(W.P, p.floor_P);
    U.d  = d;
    U.mx = d * W.u;
    U.my = d * W.v;
    const double eint = P/(p.gamma - 1.0);
    const double ke   = 0.5 * d * (W.u*W.u + W.v*W.v);
    U.E  = eint + ke;
    return U;
}

static inline double sound_speed(const W4& W, const Params& p){
    return std::sqrt(p.gamma * W.P / clamp_min(W.d, p.floor_rho));
}

// ----------------------------- FFT (power-of-two) -----------------------------
static void fft1d(std::vector<std::complex<double>>& a, bool inverse){
    const int n = (int)a.size();
    for (int i=1, j=0; i<n; i++){
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len=2; len<=n; len<<=1){
        const double ang = 2.0*M_PI/len * (inverse ? +1.0 : -1.0);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i=0; i<n; i+=len){
            std::complex<double> w(1.0, 0.0);
            for (int j=0; j<len/2; j++){
                std::complex<double> u = a[i+j];
                std::complex<double> v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse){
        for (int i=0;i<n;i++) a[i] /= (double)n;
    }
}

static void fft2d(std::vector<std::complex<double>>& A, int Nx, int Ny, bool inverse){
    std::vector<std::complex<double>> line(std::max(Nx, Ny));

    for(int j=0;j<Ny;j++){
        line.assign(Nx, {});
        for(int i=0;i<Nx;i++) line[i] = A[i + Nx*j];
        fft1d(line, inverse);
        for(int i=0;i<Nx;i++) A[i + Nx*j] = line[i];
    }

    for(int i=0;i<Nx;i++){
        line.assign(Ny, {});
        for(int j=0;j<Ny;j++) line[j] = A[i + Nx*j];
        fft1d(line, inverse);
        for(int j=0;j<Ny;j++) A[i + Nx*j] = line[j];
    }
}

// ----------------------------- HLLC -----------------------------
static U4 flux_x_from_prim(const W4& W, const Params& p){
    U4 F;
    const double d = W.d;
    const double u = W.u;
    const double v = W.v;
    const double P = W.P;
    const double E = P/(p.gamma - 1.0) + 0.5*d*(u*u + v*v);
    F.d  = d*u;
    F.mx = d*u*u + P;
    F.my = d*u*v;
    F.E  = (E + P)*u;
    return F;
}

static U4 hllc_x(const W4& WL, const W4& WR, const Params& p){
    const double aL = sound_speed(WL, p);
    const double aR = sound_speed(WR, p);

    const double uL = WL.u, vL = WL.v, dL = WL.d, pL = WL.P;
    const double uR = WR.u, vR = WR.v, dR = WR.d, pR = WR.P;

    const double EL = pL/(p.gamma-1.0) + 0.5*dL*(uL*uL + vL*vL);
    const double ER = pR/(p.gamma-1.0) + 0.5*dR*(uR*uR + vR*vR);

    const double SL = std::min(uL - aL, uR - aR);
    const double SR = std::max(uL + aL, uR + aR);

    const double num = (pR - pL) + dL*uL*(SL - uL) - dR*uR*(SR - uR);
    const double den = dL*(SL - uL) - dR*(SR - uR);
    const double SM  = (std::abs(den) > 0.0) ? (num/den) : 0.0;

    const U4 FL = flux_x_from_prim(WL, p);
    const U4 FR = flux_x_from_prim(WR, p);

    if (0.0 <= SL) return FL;
    if (0.0 >= SR) return FR;

    auto Ustar = [&](const W4& W, double S, double E)->U4{
        U4 Us;
        const double d = W.d;
        const double u = W.u;
        const double v = W.v;
        const double P = W.P;

        const double fac = d * (S - u) / (S - SM);
        Us.d  = fac;
        Us.mx = fac * SM;
        Us.my = fac * v;
        const double pstar = P + d*(S - u)*(SM - u);
        Us.E  = ( (S - u)*E - P*u + pstar*SM ) / (S - SM);
        return Us;
    };

    const U4 ULs = Ustar(WL, SL, EL);
    const U4 URs = Ustar(WR, SR, ER);

    if (0.0 < SM){
        U4 F;
        F.d  = FL.d  + SL*(ULs.d  - dL);
        F.mx = FL.mx + SL*(ULs.mx - dL*uL);
        F.my = FL.my + SL*(ULs.my - dL*vL);
        F.E  = FL.E  + SL*(ULs.E  - EL);
        return F;
    } else {
        U4 F;
        F.d  = FR.d  + SR*(URs.d  - dR);
        F.mx = FR.mx + SR*(URs.mx - dR*uR);
        F.my = FR.my + SR*(URs.my - dR*vR);
        F.E  = FR.E  + SR*(URs.E  - ER);
        return F;
    }
}

static U4 hllc_y(const W4& WL, const W4& WR, const Params& p){
    W4 AL = WL; W4 AR = WR;
    std::swap(AL.u, AL.v);
    std::swap(AR.u, AR.v);
    U4 Fx = hllc_x(AL, AR, p);
    U4 G;
    G.d  = Fx.d;
    G.mx = Fx.my; // flux of rho*u in y
    G.my = Fx.mx; // flux of rho*v in y
    G.E  = Fx.E;
    return G;
}

// ----------------------------- Poisson gravity (FFT) -----------------------------
static void solve_poisson_periodic_fft(
    const std::vector<double>& rho,
    std::vector<double>& phi,
    const Grid2D& g,
    const Params& p
){
    const int Nx = g.Nx, Ny = g.Ny;
    const int N = Nx*Ny;

    double mean = 0.0;
    for(int q=0;q<N;q++) mean += rho[q];
    mean /= (double)N;

    std::vector<std::complex<double>> A(N);
    for(int q=0;q<N;q++){
        A[q] = std::complex<double>(rho[q] - mean, 0.0);
    }

    fft2d(A, Nx, Ny, false);

    const double Lx = g.Lx, Ly = g.Ly;
    const double two_pi = 2.0 * M_PI;

    for(int j=0;j<Ny;j++){
        int nj = (j <= Ny/2) ? j : j - Ny;
        const double ky = two_pi * (double)nj / Ly;
        for(int i=0;i<Nx;i++){
            int ni = (i <= Nx/2) ? i : i - Nx;
            const double kx = two_pi * (double)ni / Lx;
            const double k2 = kx*kx + ky*ky;
            const int q = i + Nx*j;
            if (k2 == 0.0){
                A[q] = std::complex<double>(0.0, 0.0);
            } else {
                A[q] *= (-(4.0*M_PI*p.G) / k2);
            }
        }
    }

    fft2d(A, Nx, Ny, true);

    phi.assign(N, 0.0);
    for(int q=0;q<N;q++) phi[q] = A[q].real();
}

static void compute_gravity_accel(
    const std::vector<double>& phi,
    std::vector<double>& gx,
    std::vector<double>& gy,
    const Grid2D& g
){
    const int Nx = g.Nx, Ny = g.Ny;
    gx.assign(Nx*Ny, 0.0);
    gy.assign(Nx*Ny, 0.0);

    const double idx2 = 1.0/(2.0*g.dx);
    const double idy2 = 1.0/(2.0*g.dy);

    for(int j=0;j<Ny;j++){
        int jp = imod(j+1, Ny);
        int jm = imod(j-1, Ny);
        for(int i=0;i<Nx;i++){
            int ip = imod(i+1, Nx);
            int im = imod(i-1, Nx);

            const int q  = i  + Nx*j;
            const int qip= ip + Nx*j;
            const int qim= im + Nx*j;
            const int qjp= i  + Nx*jp;
            const int qjm= i  + Nx*jm;

            gx[q] = -(phi[qip] - phi[qim]) * idx2;
            gy[q] = -(phi[qjp] - phi[qjm]) * idy2;
        }
    }
}

// ----------------------------- RHS assembly -----------------------------
static void compute_rhs(
    const std::vector<U4>& U,
    std::vector<U4>& dU,
    const Grid2D& g,
    const Params& p
){
    const int Nx = g.Nx, Ny = g.Ny;
    const int N = Nx*Ny;

    std::vector<W4> W(N);
    std::vector<double> rho(N);
    for(int q=0;q<N;q++){
        W[q] = cons_to_prim(U[q], p);
        rho[q] = W[q].d;
    }

    std::vector<double> phi, gx, gy;
    if (p.gravity_on){
        solve_poisson_periodic_fft(rho, phi, g, p);
        compute_gravity_accel(phi, gx, gy, g);
    }

    std::vector<U4> Fx(N), Fy(N);

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            int ip = imod(i+1, Nx);
            Fx[i + Nx*j] = hllc_x(W[i + Nx*j], W[ip + Nx*j], p);
        }
    }
    for(int j=0;j<Ny;j++){
        int jp = imod(j+1, Ny);
        for(int i=0;i<Nx;i++){
            Fy[i + Nx*j] = hllc_y(W[i + Nx*j], W[i + Nx*jp], p);
        }
    }

    if (p.viscosity_on && p.mu > 0.0){
        std::vector<double> du_dx(N), du_dy(N), dv_dx(N), dv_dy(N);

        const double idx2 = 1.0/(2.0*g.dx);
        const double idy2 = 1.0/(2.0*g.dy);

        for(int j=0;j<Ny;j++){
            int jp = imod(j+1, Ny);
            int jm = imod(j-1, Ny);
            for(int i=0;i<Nx;i++){
                int ip = imod(i+1, Nx);
                int im = imod(i-1, Nx);

                const int q  = i  + Nx*j;
                const int qip= ip + Nx*j;
                const int qim= im + Nx*j;
                const int qjp= i  + Nx*jp;
                const int qjm= i  + Nx*jm;

                du_dx[q] = (W[qip].u - W[qim].u) * idx2;
                dv_dx[q] = (W[qip].v - W[qim].v) * idx2;
                du_dy[q] = (W[qjp].u - W[qjm].u) * idy2;
                dv_dy[q] = (W[qjp].v - W[qjm].v) * idy2;
            }
        }

        // x-faces: normal = x
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                int ip = imod(i+1, Nx);
                const int qL = i  + Nx*j;
                const int qR = ip + Nx*j;

                const double dudx = 0.5*(du_dx[qL] + du_dx[qR]);
                const double dudy = 0.5*(du_dy[qL] + du_dy[qR]);
                const double dvdx = 0.5*(dv_dx[qL] + dv_dx[qR]);
                const double dvdy = 0.5*(dv_dy[qL] + dv_dy[qR]);

                const double divv = dudx + dvdy;
                const double mu = p.mu;

                const double tau_xx = mu * (2.0*dudx - (2.0/3.0)*divv);
                const double tau_xy = mu * (dudy + dvdx);

                const double uf = 0.5*(W[qL].u + W[qR].u);
                const double vf = 0.5*(W[qL].v + W[qR].v);

                U4& F = Fx[i + Nx*j];
                F.mx += -tau_xx;
                F.my += -tau_xy;
                F.E  += -(uf*tau_xx + vf*tau_xy);
            }
        }

        // y-faces: normal = y
        for(int j=0;j<Ny;j++){
            int jp = imod(j+1, Ny);
            for(int i=0;i<Nx;i++){
                const int qL = i + Nx*j;
                const int qR = i + Nx*jp;

                const double dudx = 0.5*(du_dx[qL] + du_dx[qR]);
                const double dudy = 0.5*(du_dy[qL] + du_dy[qR]);
                const double dvdx = 0.5*(dv_dx[qL] + dv_dx[qR]);
                const double dvdy = 0.5*(dv_dy[qL] + dv_dy[qR]);

                const double divv = dudx + dvdy;
                const double mu = p.mu;

                const double tau_yy = mu * (2.0*dvdy - (2.0/3.0)*divv);
                const double tau_xy = mu * (dudy + dvdx);

                const double uf = 0.5*(W[qL].u + W[qR].u);
                const double vf = 0.5*(W[qL].v + W[qR].v);

                U4& G = Fy[i + Nx*j];
                G.mx += -tau_xy;   // tau_yx = tau_xy
                G.my += -tau_yy;
                G.E  += -(uf*tau_xy + vf*tau_yy);
            }
        }
    }

    dU.assign(N, U4{0,0,0,0});

    const double idx1 = 1.0/g.dx;
    const double idy1 = 1.0/g.dy;

    for(int j=0;j<Ny;j++){
        int jm = imod(j-1, Ny);
        for(int i=0;i<Nx;i++){
            int im = imod(i-1, Nx);
            const int q  = i  + Nx*j;

            const U4& Fp = Fx[i  + Nx*j];
            const U4& Fm = Fx[im + Nx*j];

            const U4& Gp = Fy[i + Nx*j ];
            const U4& Gm = Fy[i + Nx*jm];

            U4 R;
            R.d  = -(Fp.d  - Fm.d )*idx1 - (Gp.d  - Gm.d )*idy1;
            R.mx = -(Fp.mx - Fm.mx)*idx1 - (Gp.mx - Gm.mx)*idy1;
            R.my = -(Fp.my - Fm.my)*idx1 - (Gp.my - Gm.my)*idy1;
            R.E  = -(Fp.E  - Fm.E )*idx1 - (Gp.E  - Gm.E )*idy1;

            if (p.gravity_on){
                const double d = W[q].d;
                const double u = W[q].u;
                const double v = W[q].v;
                R.mx += d * gx[q];
                R.my += d * gy[q];
                R.E  += d * (u*gx[q] + v*gy[q]);
            }

            dU[q] = R;
        }
    }
}

// ----------------------------- time step -----------------------------
static double compute_dt(const std::vector<U4>& U, const Grid2D& g, const Params& p){
    const int N = g.Nx*g.Ny;

    double smax = 0.0;
    double rho_min = 1e300;

    for(int q=0;q<N;q++){
        W4 W = cons_to_prim(U[q], p);
        const double a = sound_speed(W, p);
        smax = std::max(smax, std::abs(W.u) + a);
        smax = std::max(smax, std::abs(W.v) + a);
        rho_min = std::min(rho_min, W.d);
    }

    const double h = std::min(g.dx, g.dy);
    double dt_hyp = (smax > 0.0) ? (p.cfl * h / smax) : 1e-6;
    double dt = dt_hyp;

    if (p.viscosity_on && p.mu > 0.0){
        const double nu_max = p.mu / clamp_min(rho_min, p.floor_rho);
        const double dt_visc = (nu_max > 0.0) ? (p.cfl_visc * h*h / nu_max) : dt_hyp;
        dt = std::min(dt, dt_visc);
    }

    return dt;
}

// ----------------------------- HDF5 output -----------------------------
static void write_1d(hid_t file, const char* name, const std::vector<double>& v){
    hsize_t dims[1] = { (hsize_t)v.size() };
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(file, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Dclose(ds); H5Sclose(sp);
}

// data layout: a[t*(Ny*Nx) + j*Nx + i]
static void write_3d_rowmajor(
    hid_t file, const char* name,
    const std::vector<double>& a,
    hsize_t Nt, hsize_t Ny, hsize_t Nx
){
    hsize_t dims[3] = { Nt, Ny, Nx };
    hid_t sp = H5Screate_simple(3, dims, nullptr);
    hid_t ds = H5Dcreate2(file, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data());
    H5Dclose(ds); H5Sclose(sp);
}

static void write_history_h5(
    const std::string& path,
    const Grid2D& g,
    const std::vector<double>& times,
    const std::vector<double>& rho_hist,
    const std::vector<double>& P_hist,
    const std::vector<double>& Mach_hist,
    const std::vector<double>& u_hist,
    const std::vector<double>& v_hist,
    const std::vector<double>& momx_hist,
    const std::vector<double>& momy_hist,
    const std::vector<double>& Phi_hist
){
    const hsize_t Nx = (hsize_t)g.Nx;
    const hsize_t Ny = (hsize_t)g.Ny;
    const hsize_t Nt = (hsize_t)times.size();

    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0){
        std::cerr << "H5Fcreate failed: " << path << "\n";
        std::exit(1);
    }

    write_1d(f, "/x", g.x);
    write_1d(f, "/y", g.y);
    write_1d(f, "/t", times);

    write_3d_rowmajor(f, "/rho",  rho_hist,  Nt, Ny, Nx);
    write_3d_rowmajor(f, "/P",    P_hist,    Nt, Ny, Nx);
    write_3d_rowmajor(f, "/Mach", Mach_hist, Nt, Ny, Nx);

    write_3d_rowmajor(f, "/u",    u_hist,    Nt, Ny, Nx);
    write_3d_rowmajor(f, "/v",    v_hist,    Nt, Ny, Nx);
    write_3d_rowmajor(f, "/momx", momx_hist, Nt, Ny, Nx);
    write_3d_rowmajor(f, "/momy", momy_hist, Nt, Ny, Nx);
    write_3d_rowmajor(f, "/Phi",  Phi_hist,  Nt, Ny, Nx);

    H5Fclose(f);
}

// ----------------------------- main -----------------------------
int main(){
    Params p;

    if (!is_pow2(p.Nx) || !is_pow2(p.Ny)){
        std::cerr << "Nx and Ny must be powers of two for the FFT Poisson solver.\n";
        return 1;
    }

    Grid2D g(p.Nx, p.Ny, p.x0, p.x1, p.y0, p.y1);

    const int Nx = g.Nx, Ny = g.Ny;
    const int N  = Nx*Ny;

    std::vector<U4> U(N);


    // -------------------- Collapse IC: mild central overdensity + small subsonic perturbations --------------------
    // Goal: form a central condensation + accretion shock (gravity + shock capturing test)
    //
    // Suggested companion parameters (good starting point):
    //   p.G   = 4.0;         // strengthen gravity
    //   P0    = 0.2;         // reduce pressure support (see below)
    //   p.mu  = 0.0;         // start inviscid; later try 1e-5
    //   p.t_end ~ 0.5..1.0   // allow time for shock to form
    //
    // Replace your existing IC block in main() with this.

    std::mt19937 rng(p.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // Background state
    const double rho0 = 1.0;
    const double P0   = 0.05;   // lower P -> easier collapse

    // Central overdensity: rho = rho0 [1 + A exp(-r^2/2sigma^2)]
    const double A     = 1;                      // peak overdensity ~ 80% above background
    const double sigma = 0.10 * std::min(g.Lx, g.Ly);

    // Small random density noise (optional)
    const double Arho  = 0.01;   // 1% noise

    // Small subsonic velocity perturbation (optional, helps break symmetry)
    const double a0    = std::sqrt(p.gamma * P0 / rho0);
    const double Mach0 = 0.05;   // very small (0.02..0.10)
    const double vamp  = Mach0 * a0;

    // Domain center
    const double xc = 0.5*(p.x0 + p.x1);
    const double yc = 0.5*(p.y0 + p.y1);

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            const int q = i + Nx*j;
            const double x = g.x[i];
            const double y = g.y[j];

            const double rx = x - xc;
            const double ry = y - yc;
            const double r2 = rx*rx + ry*ry;

            // Density: background + Gaussian bump + small noise
            const double bump  = A * std::exp(-0.5 * r2 / (sigma*sigma));
            const double noise = Arho * gauss(rng);

            W4 W;
            W.d = rho0 * (1.0 + bump + noise);
            W.d = clamp_min(W.d, p.floor_rho);

            // Pressure: uniform (start near hydro; collapse is gravity-driven)
            W.P = clamp_min(P0, p.floor_P);

            // Velocity: tiny random subsonic perturbation
            W.u = vamp * gauss(rng);
            W.v = vamp * gauss(rng);

            U[q] = prim_to_cons(W, p);
        }
    }

    // History buffers (row-major planes appended in time)
    std::vector<double> times;
    std::vector<double> rho_hist, P_hist, Mach_hist;
    std::vector<double> u_hist, v_hist, momx_hist, momy_hist, Phi_hist;

    // scratch planes for Phi solve at store time (so /Phi exists for diagnostics)
    std::vector<double> rho_plane(N), phi_plane;

    auto store = [&](double tcur){
        times.push_back(tcur);
        const size_t Nt = times.size();
        const size_t plane = (size_t)Nx * (size_t)Ny;

        rho_hist.resize(Nt * plane);
        P_hist.resize(Nt * plane);
        Mach_hist.resize(Nt * plane);
        u_hist.resize(Nt * plane);
        v_hist.resize(Nt * plane);
        momx_hist.resize(Nt * plane);
        momy_hist.resize(Nt * plane);
        Phi_hist.resize(Nt * plane);

        const size_t base = (Nt - 1) * plane;

        // fill primitives + momenta, and build rho_plane for Poisson
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                const int q = i + Nx*j;
                W4 W = cons_to_prim(U[q], p);

                rho_hist[base + (size_t)q] = W.d;
                P_hist[base + (size_t)q]   = W.P;

                const double a = sound_speed(W, p);
                const double vmag = std::sqrt(W.u*W.u + W.v*W.v);
                Mach_hist[base + (size_t)q] = (a > 0.0) ? (vmag / a) : 0.0;

                u_hist[base + (size_t)q] = W.u;
                v_hist[base + (size_t)q] = W.v;

                momx_hist[base + (size_t)q] = W.d * W.u;
                momy_hist[base + (size_t)q] = W.d * W.v;

                rho_plane[q] = W.d;
            }
        }

        // compute Phi for this snapshot (needed for energy budget plots)
        if (p.gravity_on){
            solve_poisson_periodic_fft(rho_plane, phi_plane, g, p);
        } else {
            phi_plane.assign(N, 0.0);
        }
        for(int q=0;q<N;q++){
            Phi_hist[base + (size_t)q] = phi_plane[q];
        }
    };

    double t = 0.0;
    double next_store = 0.0;

    store(t);
    next_store += p.dt_store;

    // SSP-RK2 (Heun)
    std::vector<U4> k1, k2, U1;
    k1.reserve(N); k2.reserve(N); U1.reserve(N);

    while (t < p.t_end - 1e-14){
        double dt = compute_dt(U, g, p);
        if (t + dt > p.t_end) dt = p.t_end - t;

        compute_rhs(U, k1, g, p);

        U1.resize(N);
        for(int q=0;q<N;q++){
            U1[q].d  = U[q].d  + dt*k1[q].d;
            U1[q].mx = U[q].mx + dt*k1[q].mx;
            U1[q].my = U[q].my + dt*k1[q].my;
            U1[q].E  = U[q].E  + dt*k1[q].E;
            if (U1[q].d < p.floor_rho) U1[q].d = p.floor_rho;
        }

        compute_rhs(U1, k2, g, p);

        for(int q=0;q<N;q++){
            U[q].d  = U[q].d  + 0.5*dt*(k1[q].d  + k2[q].d );
            U[q].mx = U[q].mx + 0.5*dt*(k1[q].mx + k2[q].mx);
            U[q].my = U[q].my + 0.5*dt*(k1[q].my + k2[q].my);
            U[q].E  = U[q].E  + 0.5*dt*(k1[q].E  + k2[q].E );
            if (U[q].d < p.floor_rho) U[q].d = p.floor_rho;
        }

        t += dt;

        if (t + 1e-14 >= next_store || t >= p.t_end - 1e-14){
            store(t);
            next_store += p.dt_store;
            std::cerr << "t=" << std::setprecision(10) << t
                      << " dt=" << dt
                      << " snapshots=" << times.size() << "\n";
        }
    }

    write_history_h5(
        p.out_h5, g, times,
        rho_hist, P_hist, Mach_hist,
        u_hist, v_hist, momx_hist, momy_hist, Phi_hist
    );

    std::cout << "Wrote " << p.out_h5
              << " with datasets /x /y /t /rho /P /Mach /u /v /momx /momy /Phi\n";
    return 0;
}
