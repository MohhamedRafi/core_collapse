// euler2d_gravity_hllc.cpp
// 2D Euler finite-volume solver with central enclosed-mass gravity.
// Method: MUSCL (MC limiter) + HLLC Riemann + SSP-RK2 time stepping.
// Output: ONE HDF5 file "out2d_hllc.h5" with datasets exactly:
//   /x (Nx), /y (Ny), /t (Nt), /rho (Nt,Ny,Nx), /P (Nt,Ny,Nx), /Mach (Nt,Ny,Nx)
// Storage is row-major with q = idx(i,j,Nx) and snapshot offset base=(n*Ny*Nx).
//
// Build:
//   clang++ -O3 -std=c++17 euler2d_gravity_hllc.cpp -lhdf5 -o euler2d_gravity_hllc
//
// Run:
//   ./euler2d_gravity_hllc

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "hdf5.h"

static inline size_t idx(int i, int j, int Nx) { return (size_t)j * (size_t)Nx + (size_t)i; }

struct Params {
    double gamma = 1.2;
    double cfl   = 0.45;

    // domain
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;

    int Nx = 50;
    int Ny = 50;

    // simulation control
    double t_end    = 5;
    double dt_store = 1e-3;

    // floors
    double floor_rho = 1e-12;
    double floor_P   = 1e-12;

    // boundary conditions
    bool periodic_x = true;
    bool periodic_y = true;

    // gravity
    bool gravity_on = true;
    double G = 1.0;
    double softening = 1e-4;
};

struct State {
    std::vector<double> rho, mx, my, E;
};

static inline void enforce_floors(const Params& p, double& rho, double& mx, double& my, double& E) {
    rho = std::max(rho, p.floor_rho);
    double u = mx / rho;
    double v = my / rho;
    double kin = 0.5 * rho * (u*u + v*v);
    double P = (p.gamma - 1.0) * (E - kin);
    if (P < p.floor_P) {
        double eint = p.floor_P / (p.gamma - 1.0);
        E = kin + eint;
    }
}

static inline void cons_to_prim_full(const Params& p, double rho, double mx, double my, double E,
                                     double& u, double& v, double& P, double& a) {
    rho = std::max(rho, p.floor_rho);
    u = mx / rho;
    v = my / rho;
    double kin = 0.5 * rho * (u*u + v*v);
    P = (p.gamma - 1.0) * (E - kin);
    if (P < p.floor_P) P = p.floor_P;
    a = std::sqrt(p.gamma * P / rho);
}

static inline double minmod(double a, double b) {
    if (a*b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

static inline double mc_limiter(double dm, double dp) {
    return minmod( 0.5*(dm+dp), minmod(2.0*dm, 2.0*dp) );
}

struct Prim { double rho, u, v, P; };

static inline Prim cons_to_prim(const Params& p, double rho, double mx, double my, double E) {
    double u, v, P, a;
    cons_to_prim_full(p, rho, mx, my, E, u, v, P, a);
    return {std::max(rho, p.floor_rho), u, v, P};
}

static inline void prim_to_cons(const Params& p, const Prim& W, double& rho, double& mx, double& my, double& E) {
    rho = std::max(W.rho, p.floor_rho);
    mx = rho * W.u;
    my = rho * W.v;
    double kin = 0.5 * rho * (W.u*W.u + W.v*W.v);
    double eint = W.P / (p.gamma - 1.0);
    E = kin + eint;
}

struct Flux { double f1, f2, f3, f4; };

static inline Flux euler_flux_x(const Params& p, const Prim& W) {
    double rho = W.rho, u = W.u, v = W.v, P = W.P;
    double E = P/(p.gamma-1.0) + 0.5*rho*(u*u+v*v);
    return { rho*u, rho*u*u + P, rho*u*v, (E+P)*u };
}

static inline Flux euler_flux_y(const Params& p, const Prim& W) {
    double rho = W.rho, u = W.u, v = W.v, P = W.P;
    double E = P/(p.gamma-1.0) + 0.5*rho*(u*u+v*v);
    return { rho*v, rho*u*v, rho*v*v + P, (E+P)*v };
}

static inline Flux hllc_x(const Params& p, const Prim& WL, const Prim& WR) {
    double aL = std::sqrt(p.gamma * WL.P / std::max(WL.rho, p.floor_rho));
    double aR = std::sqrt(p.gamma * WR.P / std::max(WR.rho, p.floor_rho));

    double SL = std::min(WL.u - aL, WR.u - aR);
    double SR = std::max(WL.u + aL, WR.u + aR);

    double rL, mLx, mLy, EL;
    double rR, mRx, mRy, ER;
    prim_to_cons(p, WL, rL, mLx, mLy, EL);
    prim_to_cons(p, WR, rR, mRx, mRy, ER);

    Flux FL = euler_flux_x(p, WL);
    Flux FR = euler_flux_x(p, WR);

    double num = (WR.P - WL.P) + rL*WL.u*(SL - WL.u) - rR*WR.u*(SR - WR.u);
    double den = rL*(SL - WL.u) - rR*(SR - WR.u);
    double SM = (std::abs(den) > 1e-14) ? (num / den) : 0.0;

    if (0.0 <= SL) return FL;
    if (0.0 >= SR) return FR;

    auto star_cons = [&](const Prim& W, double r, double E, double S) {
        double u = W.u;
        double v = W.v;
        double P = W.P;
        double rS  = r * (S - u) / (S - SM);
        double mxS = rS * SM;
        double myS = rS * v;
        double ES  = ( (S - u)*E - P*u + P*SM ) / (S - SM);
        return std::array<double,4>{rS, mxS, myS, ES};
    };

    if (0.0 < SM) {
        auto ULs = star_cons(WL, rL, EL, SL);
        return { FL.f1 + SL*(ULs[0] - rL),
                 FL.f2 + SL*(ULs[1] - mLx),
                 FL.f3 + SL*(ULs[2] - mLy),
                 FL.f4 + SL*(ULs[3] - EL) };
    } else {
        auto URs = star_cons(WR, rR, ER, SR);
        return { FR.f1 + SR*(URs[0] - rR),
                 FR.f2 + SR*(URs[1] - mRx),
                 FR.f3 + SR*(URs[2] - mRy),
                 FR.f4 + SR*(URs[3] - ER) };
    }
}

static inline Flux hllc_y(const Params& p, const Prim& WL, const Prim& WR) {
    double aL = std::sqrt(p.gamma * WL.P / std::max(WL.rho, p.floor_rho));
    double aR = std::sqrt(p.gamma * WR.P / std::max(WR.rho, p.floor_rho));

    double SL = std::min(WL.v - aL, WR.v - aR);
    double SR = std::max(WL.v + aL, WR.v + aR);

    double rL, mLx, mLy, EL;
    double rR, mRx, mRy, ER;
    prim_to_cons(p, WL, rL, mLx, mLy, EL);
    prim_to_cons(p, WR, rR, mRx, mRy, ER);

    Flux FL = euler_flux_y(p, WL);
    Flux FR = euler_flux_y(p, WR);

    double num = (WR.P - WL.P) + rL*WL.v*(SL - WL.v) - rR*WR.v*(SR - WR.v);
    double den = rL*(SL - WL.v) - rR*(SR - WR.v);
    double SM = (std::abs(den) > 1e-14) ? (num / den) : 0.0;

    if (0.0 <= SL) return FL;
    if (0.0 >= SR) return FR;

    auto star_cons = [&](const Prim& W, double r, double E, double S) {
        double u = W.u;
        double v = W.v;
        double P = W.P;
        double rS  = r * (S - v) / (S - SM);
        double myS = rS * SM;
        double mxS = rS * u;
        double ES  = ( (S - v)*E - P*v + P*SM ) / (S - SM);
        return std::array<double,4>{rS, mxS, myS, ES};
    };

    if (0.0 < SM) {
        auto ULs = star_cons(WL, rL, EL, SL);
        return { FL.f1 + SL*(ULs[0] - rL),
                 FL.f2 + SL*(ULs[1] - mLx),
                 FL.f3 + SL*(ULs[2] - mLy),
                 FL.f4 + SL*(ULs[3] - EL) };
    } else {
        auto URs = star_cons(WR, rR, ER, SR);
        return { FR.f1 + SR*(URs[0] - rR),
                 FR.f2 + SR*(URs[1] - mRx),
                 FR.f3 + SR*(URs[2] - mRy),
                 FR.f4 + SR*(URs[3] - ER) };
    }
}

struct GravField { std::vector<double> gx, gy; };

static inline int wrap(int i, int N) { int r = i % N; if (r < 0) r += N; return r; }


static inline GravField compute_enclosed_mass_gravity(const Params& p, const State& U) {
    GravField g;
    int Nx = p.Nx, Ny = p.Ny;
    g.gx.assign((size_t)Nx*Ny, 0.0);
    g.gy.assign((size_t)Nx*Ny, 0.0);
    if (!p.gravity_on) return g;

    double dx = (p.x1 - p.x0) / Nx;
    double dy = (p.y1 - p.y0) / Ny;

    // gravity center: domain center (matches your earlier code)
    double cx = 0.5*(p.x0 + p.x1);
    double cy = 0.5*(p.y0 + p.y1);

    struct CellR { double r; size_t k; double m; double x, y; };
    std::vector<CellR> cells;
    cells.reserve((size_t)Nx*Ny);

    for (int j=0;j<Ny;++j) {
        double y = p.y0 + (j + 0.5)*dy;
        for (int i=0;i<Nx;++i) {
            double x = p.x0 + (i + 0.5)*dx;
            size_t k = idx(i,j,Nx);
            double rx = x - cx;
            double ry = y - cy;
            double r  = std::sqrt(rx*rx + ry*ry);
            double rho = std::max(U.rho[k], p.floor_rho);
            double m = rho * dx * dy;
            cells.push_back({r, k, m, x, y});
        }
    }

    std::sort(cells.begin(), cells.end(),
              [](const CellR& a, const CellR& b){ return a.r < b.r; });

    double Menc = 0.0;
    size_t s = 0;
    while (s < cells.size()) {
        double r0 = cells[s].r;
        size_t e = s+1;
        while (e < cells.size() && cells[e].r <= r0 + 1e-12) ++e;

        for (size_t t=s;t<e;++t) Menc += cells[t].m;

        for (size_t t=s;t<e;++t) {
            size_t k = cells[t].k;
            double rx = cells[t].x - cx;
            double ry = cells[t].y - cy;
            double r2 = rx*rx + ry*ry + p.softening*p.softening;
            double r  = std::sqrt(r2);
            if (r > 0.0) {
                double gmag = -p.G * Menc / r2;
                g.gx[k] = gmag * (rx / r);
                g.gy[k] = gmag * (ry / r);
            }
        }
        s = e;
    }

    return g;
}

static inline GravField compute_poisson_gravity_periodic(const Params& p, const State& U) {
    GravField g;
    const int Nx = p.Nx, Ny = p.Ny;
    g.gx.assign((size_t)Nx*Ny, 0.0);
    g.gy.assign((size_t)Nx*Ny, 0.0);
    if (!p.gravity_on) return g;

    const double dx = (p.x1 - p.x0) / Nx;
    const double dy = (p.y1 - p.y0) / Ny;
    const double idx2 = 1.0/(dx*dx);
    const double idy2 = 1.0/(dy*dy);

    auto widx = [&](int i, int j)->size_t {
        i = wrap(i, Nx);
        j = wrap(j, Ny);
        return idx(i,j,Nx);
    };

    // Build source term: rhs = 4*pi*G*(rho - mean(rho))
    std::vector<double> rho((size_t)Nx*Ny, 0.0);
    double rhobar = 0.0;
    for (int j=0;j<Ny;++j){
        for (int i=0;i<Nx;++i){
            size_t k = idx(i,j,Nx);
            double r = std::max(U.rho[k], p.floor_rho);
            rho[k] = r;
            rhobar += r;
        }
    }
    rhobar /= (double)(Nx*Ny);

    const double FOURPI = 4.0 * M_PI;
    std::vector<double> rhs((size_t)Nx*Ny, 0.0);
    for (size_t k=0;k<rhs.size();++k){
        rhs[k] = FOURPI * p.G * (rho[k] - rhobar);
    }

    // Solve Poisson with periodic BC via SOR:
    // (phiE-2phi+phiW)/dx^2 + (phiN-2phi+phiS)/dy^2 = rhs
    std::vector<double> phi((size_t)Nx*Ny, 0.0);

    const int    max_iter = 50000;
    const double tol      = 1e-10;     // residual tolerance (can relax to 1e-8)
    const double omega    = 1.7;       // SOR factor (1<omega<2)

    const double denom = 2.0*(idx2 + idy2);

    for (int it=0; it<max_iter; ++it) {
        double max_res = 0.0;

        for (int j=0;j<Ny;++j){
            for (int i=0;i<Nx;++i){
                size_t k  = idx(i,j,Nx);

                double phiE = phi[widx(i+1,j)];
                double phiW = phi[widx(i-1,j)];
                double phiN = phi[widx(i,j+1)];
                double phiS = phi[widx(i,j-1)];

                double phi_new = ( (phiE + phiW)*idx2 + (phiN + phiS)*idy2 - rhs[k] ) / denom;

                // SOR update
                double delta = phi_new - phi[k];
                phi[k] += omega * delta;

                // residual (discrete Laplacian - rhs)
                double lap = (phiE - 2.0*phi[k] + phiW)*idx2 + (phiN - 2.0*phi[k] + phiS)*idy2;
                double res = lap - rhs[k];
                max_res = std::max(max_res, std::abs(res));
            }
        }

        // Remove arbitrary constant mode (periodic Poisson has gauge freedom)
        // enforce mean(phi)=0
        double phibar = 0.0;
        for (double v : phi) phibar += v;
        phibar /= (double)phi.size();
        for (double& v : phi) v -= phibar;

        if (max_res < tol) break;
    }

    // g = -grad(phi) with central differences
    for (int j=0;j<Ny;++j){
        for (int i=0;i<Nx;++i){
            size_t k = idx(i,j,Nx);

            double phiE = phi[widx(i+1,j)];
            double phiW = phi[widx(i-1,j)];
            double phiN = phi[widx(i,j+1)];
            double phiS = phi[widx(i,j-1)];

            g.gx[k] = -(phiE - phiW) / (2.0*dx);
            g.gy[k] = -(phiN - phiS) / (2.0*dy);
        }
    }

    return g;
}



static inline double compute_dt(const Params& p, const State& U) {
    int Nx = p.Nx, Ny = p.Ny;
    double dx = (p.x1 - p.x0) / Nx;
    double dy = (p.y1 - p.y0) / Ny;

    double max_speed = 1e-30;
    for (int j=0;j<Ny;++j) for (int i=0;i<Nx;++i) {
        size_t k = idx(i,j,Nx);
        double u, v, P, a;
        cons_to_prim_full(p, U.rho[k], U.mx[k], U.my[k], U.E[k], u, v, P, a);
        max_speed = std::max(max_speed, std::abs(u) + a);
        max_speed = std::max(max_speed, std::abs(v) + a);
    }
    return p.cfl * std::min(dx, dy) / max_speed;
}

static inline State rhs(const Params& p, const State& U) {
    int Nx = p.Nx, Ny = p.Ny;
    double dx = (p.x1 - p.x0) / Nx;
    double dy = (p.y1 - p.y0) / Ny;

    GravField g = compute_poisson_gravity_periodic(p, U); //compute_enclosed_mass_gravity(p, U);

    State dU;
    dU.rho.assign((size_t)Nx*Ny, 0.0);
    dU.mx .assign((size_t)Nx*Ny, 0.0);
    dU.my .assign((size_t)Nx*Ny, 0.0);
    dU.E  .assign((size_t)Nx*Ny, 0.0);

    std::vector<Prim> W((size_t)Nx*Ny);
    for (int j=0;j<Ny;++j) for (int i=0;i<Nx;++i) {
        size_t k = idx(i,j,Nx);
        double rho = U.rho[k], mx=U.mx[k], my=U.my[k], E=U.E[k];
        enforce_floors(p, rho, mx, my, E);
        W[k] = cons_to_prim(p, rho, mx, my, E);
    }

    std::vector<Flux> Fx((size_t)(Nx+1)*Ny);
    auto fidx_x = [&](int i, int j){ return (size_t)j*(size_t)(Nx+1) + (size_t)i; };

    for (int j=0;j<Ny;++j) {
        for (int i=0;i<=Nx;++i) {
            int iL = i-1;
            int iR = i;
            if (p.periodic_x) { iL = wrap(iL, Nx); iR = wrap(iR, Nx); }
            else { iL = std::clamp(iL, 0, Nx-1); iR = std::clamp(iR, 0, Nx-1); }

            auto recon = [&](auto get)->std::pair<double,double> {
                auto at = [&](int ii)->double {
                    int iw = p.periodic_x ? wrap(ii, Nx) : std::clamp(ii, 0, Nx-1);
                    return get(W[idx(iw,j,Nx)]);
                };
                double wm = at(iL-1);
                double w0 = at(iL);
                double wp = at(iR);
                double wpp= at(iR+1);

                double sL = mc_limiter(w0 - wm, wp - w0);
                double sR = mc_limiter(wp - w0, wpp - wp);

                return std::pair<double,double>(w0 + 0.5*sL, wp - 0.5*sR);
            };

            auto [rhoL, rhoR] = recon([](const Prim& q){ return q.rho; });
            auto [uL,   uR  ] = recon([](const Prim& q){ return q.u;   });
            auto [vL,   vR  ] = recon([](const Prim& q){ return q.v;   });
            auto [PL,   PR  ] = recon([](const Prim& q){ return q.P;   });

            Prim WLf{ std::max(rhoL, p.floor_rho), uL, vL, std::max(PL, p.floor_P) };
            Prim WRf{ std::max(rhoR, p.floor_rho), uR, vR, std::max(PR, p.floor_P) };

            Fx[fidx_x(i,j)] = hllc_x(p, WLf, WRf);
        }
    }

    std::vector<Flux> Fy((size_t)Nx*(Ny+1));
    auto fidx_y = [&](int i, int j){ return (size_t)j*(size_t)Nx + (size_t)i; };

    for (int j=0;j<=Ny;++j) {
        for (int i=0;i<Nx;++i) {
            int jL = j-1;
            int jR = j;
            if (p.periodic_y) { jL = wrap(jL, Ny); jR = wrap(jR, Ny); }
            else { jL = std::clamp(jL, 0, Ny-1); jR = std::clamp(jR, 0, Ny-1); }

            auto recon = [&](auto get)->std::pair<double,double> {
                auto at = [&](int jj)->double {
                    int jw = p.periodic_y ? wrap(jj, Ny) : std::clamp(jj, 0, Ny-1);
                    return get(W[idx(i,jw,Nx)]);
                };
                double wm = at(jL-1);
                double w0 = at(jL);
                double wp = at(jR);
                double wpp= at(jR+1);

                double sL = mc_limiter(w0 - wm, wp - w0);
                double sR = mc_limiter(wp - w0, wpp - wp);

                return std::pair<double,double>(w0 + 0.5*sL, wp - 0.5*sR);
            };

            auto [rhoL, rhoR] = recon([](const Prim& q){ return q.rho; });
            auto [uL,   uR  ] = recon([](const Prim& q){ return q.u;   });
            auto [vL,   vR  ] = recon([](const Prim& q){ return q.v;   });
            auto [PL,   PR  ] = recon([](const Prim& q){ return q.P;   });

            Prim WLf{ std::max(rhoL, p.floor_rho), uL, vL, std::max(PL, p.floor_P) };
            Prim WRf{ std::max(rhoR, p.floor_rho), uR, vR, std::max(PR, p.floor_P) };

            Fy[fidx_y(i,j)] = hllc_y(p, WLf, WRf);
        }
    }

    for (int j=0;j<Ny;++j) for (int i=0;i<Nx;++i) {
        size_t k = idx(i,j,Nx);

        Flux fxR = Fx[(size_t)j*(size_t)(Nx+1) + (size_t)(i+1)];
        Flux fxL = Fx[(size_t)j*(size_t)(Nx+1) + (size_t)i];
        Flux fyU = Fy[(size_t)(j+1)*(size_t)Nx + (size_t)i];
        Flux fyD = Fy[(size_t)j*(size_t)Nx + (size_t)i];

        dU.rho[k] = -(fxR.f1 - fxL.f1)/dx - (fyU.f1 - fyD.f1)/dy;
        dU.mx[k]  = -(fxR.f2 - fxL.f2)/dx - (fyU.f2 - fyD.f2)/dy;
        dU.my[k]  = -(fxR.f3 - fxL.f3)/dx - (fyU.f3 - fyD.f3)/dy;
        dU.E[k]   = -(fxR.f4 - fxL.f4)/dx - (fyU.f4 - fyD.f4)/dy;

        if (p.gravity_on) {
            double rho = std::max(U.rho[k], p.floor_rho);
            double u = U.mx[k] / rho;
            double v = U.my[k] / rho;
            dU.mx[k] += rho * g.gx[k];
            dU.my[k] += rho * g.gy[k];
            dU.E[k]  += rho * (u*g.gx[k] + v*g.gy[k]);
        }
    }

    return dU;
}

static inline void rk2_step(const Params& p, State& U, double dt) {
    State k1 = rhs(p, U);
    State U1 = U;

    size_t N = U.rho.size();
    for (size_t q=0;q<N;++q) {
        U1.rho[q] += dt * k1.rho[q];
        U1.mx[q]  += dt * k1.mx[q];
        U1.my[q]  += dt * k1.my[q];
        U1.E[q]   += dt * k1.E[q];
        enforce_floors(p, U1.rho[q], U1.mx[q], U1.my[q], U1.E[q]);
    }

    State k2 = rhs(p, U1);
    for (size_t q=0;q<N;++q) {
        U.rho[q] = 0.5*U.rho[q] + 0.5*(U1.rho[q] + dt*k2.rho[q]);
        U.mx[q]  = 0.5*U.mx[q]  + 0.5*(U1.mx[q]  + dt*k2.mx[q]);
        U.my[q]  = 0.5*U.my[q]  + 0.5*(U1.my[q]  + dt*k2.my[q]);
        U.E[q]   = 0.5*U.E[q]   + 0.5*(U1.E[q]   + dt*k2.E[q]);
        enforce_floors(p, U.rho[q], U.mx[q], U.my[q], U.E[q]);
    }
}

// ---------------- HDF5 helpers (match your out2d.h5 layout) ----------------
static void h5_check(herr_t s, const char* what){
    if(s < 0){ std::cerr << "HDF5 error in " << what << "\n"; std::exit(1); }
}

static void write_1d(hid_t file, const char* name, const std::vector<double>& a){
    hsize_t dims[1] = {(hsize_t)a.size()};
    hid_t sp = H5Screate_simple(1,dims,nullptr);
    hid_t ds = H5Dcreate2(file,name,H5T_IEEE_F64LE,sp,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    h5_check(H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,a.data()), "write_1d");
    H5Dclose(ds); H5Sclose(sp);
}

static void write_3d_rowmajor(hid_t file, const char* name, const std::vector<double>& a,
                             hsize_t Nt, hsize_t Ny, hsize_t Nx){
    hsize_t dims[3] = {Nt, Ny, Nx};
    hid_t sp = H5Screate_simple(3,dims,nullptr);
    hid_t ds = H5Dcreate2(file,name,H5T_IEEE_F64LE,sp,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    h5_check(H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,a.data()), "write_3d");
    H5Dclose(ds); H5Sclose(sp);
}

static State init_random_blob(const Params& p) {
    const int Nx = p.Nx, Ny = p.Ny;
    const double dx = (p.x1 - p.x0) / Nx;
    const double dy = (p.y1 - p.y0) / Ny;

    State U;
    U.rho.assign((size_t)Nx*Ny, 0.0);
    U.mx .assign((size_t)Nx*Ny, 0.0);
    U.my .assign((size_t)Nx*Ny, 0.0);
    U.E  .assign((size_t)Nx*Ny, 0.0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uniV(-1.0, 1.0);

    // Background state
    const double rho0 = 1.0;
    const double P0   = 1.0;

    // Small random fluctuations
    const double amp_rho = 0.10;
    const double amp_P   = 0.10;
    const double vmax    = 0.05;

    // Strong central overdensity (Gaussian bump)
    const double xc = 0.5*(p.x0 + p.x1);
    const double yc = 0.5*(p.y0 + p.y1);
    const double sigma = 0.08;      // width in box units (try 0.03–0.08)
    const double bump_amp = 0.5;   // peak adds ~15*rho0 at center (make bigger if you want)

    for (int j=0;j<Ny;++j) {
        const double y = p.y0 + (j + 0.5)*dy;
        for (int i=0;i<Nx;++i) {
            const double x = p.x0 + (i + 0.5)*dx;
            const size_t k = idx(i,j,Nx);

            const double fr = 2.0*uni01(rng) - 1.0;
            const double fp = 2.0*uni01(rng) - 1.0;

            // random background
            double rho = rho0 * (1.0 + amp_rho * fr);
            double P   = P0   * (1.0 + amp_P   * fp);

            // add big central Gaussian bump to rho
            const double rx = x - xc;
            const double ry = y - yc;
            const double r2 = rx*rx + ry*ry;
            const double bump = bump_amp * rho0 * std::exp(-r2 / (2.0*sigma*sigma));
            rho += bump;

            // velocities
            const double u = vmax * uniV(rng);
            const double v = vmax * uniV(rng);

            // floors and conservative conversion
            rho = std::max(rho, p.floor_rho);
            P   = std::max(P,   p.floor_P);

            U.rho[k] = rho;
            U.mx[k]  = rho * u;
            U.my[k]  = rho * v;
            U.E[k]   = P/(p.gamma - 1.0) + 0.5 * rho * (u*u + v*v);

            enforce_floors(p, U.rho[k], U.mx[k], U.my[k], U.E[k]);
        }
    }

    return U;
}


int main() {
    Params p;

    // build x,y (cell centers)
    const double dx = (p.x1 - p.x0) / p.Nx;
    const double dy = (p.y1 - p.y0) / p.Ny;
    std::vector<double> x((size_t)p.Nx), y((size_t)p.Ny);
    for(int i=0;i<p.Nx;i++) x[(size_t)i] = p.x0 + (i+0.5)*dx;
    for(int j=0;j<p.Ny;j++) y[(size_t)j] = p.y0 + (j+0.5)*dy;

    State U = init_random_blob(p);

    // history buffers (match your out2d.h5 layout)
    std::vector<double> times;
    std::vector<double> rho_hist, P_hist, Mach_hist;

    auto store = [&](double tcur){
        times.push_back(tcur);
        size_t Nt = times.size();
        size_t plane = (size_t)p.Nx * (size_t)p.Ny;

        rho_hist.resize(Nt * plane);
        P_hist.resize(Nt * plane);
        Mach_hist.resize(Nt * plane);

        size_t base = (Nt - 1) * plane;

        for(int j=0;j<p.Ny;j++){
            for(int i=0;i<p.Nx;i++){
                size_t q = idx(i,j,p.Nx);

                double rho = U.rho[q], mx = U.mx[q], my = U.my[q], E = U.E[q];
                enforce_floors(p, rho, mx, my, E);

                double u,v,P,a;
                cons_to_prim_full(p, rho, mx, my, E, u, v, P, a);

                rho_hist[base + q] = rho;
                P_hist[base + q]   = P;

                double vmag = std::sqrt(u*u + v*v);
                Mach_hist[base + q] = (a > 0.0) ? (vmag / a) : 0.0;
            }
        }
    };

    double t = 0.0;
    double next_store = 0.0;

    store(t);
    next_store += p.dt_store;

    while (t < p.t_end - 1e-14) {
        double dt = compute_dt(p, U);
        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "Bad dt=" << dt << " at t=" << t << "\n";
            return 1;
        }
        if (t + dt > p.t_end) dt = p.t_end - t;

        rk2_step(p, U, dt);
        t += dt;

        if (t + 1e-14 >= next_store || t >= p.t_end - 1e-14) {
            store(t);
            next_store += p.dt_store;
            std::cerr << "t=" << std::setprecision(10) << t
                      << " dt=" << dt
                      << " snapshots=" << times.size() << "\n";
        }
    }

    // write ONE file with the exact dataset names/shapes you use elsewhere
    hid_t file = H5Fcreate("out2d_hllc.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(file < 0){ std::cerr << "Failed to create out2d_hllc.h5\n"; return 1; }

    write_1d(file, "/x", x);
    write_1d(file, "/y", y);
    write_1d(file, "/t", times);

    hsize_t Nt = (hsize_t)times.size();
    hsize_t Ny = (hsize_t)p.Ny;
    hsize_t Nx = (hsize_t)p.Nx;

    write_3d_rowmajor(file, "/rho",  rho_hist,  Nt, Ny, Nx);
    write_3d_rowmajor(file, "/P",    P_hist,    Nt, Ny, Nx);
    write_3d_rowmajor(file, "/Mach", Mach_hist, Nt, Ny, Nx);

    H5Fclose(file);
    std::cout << "Wrote out2d_hllc.h5 datasets: /x /y /t /rho /P /Mach\n";
    return 0;
}
