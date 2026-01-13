#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

// Requires these to exist in the including translation unit:
//   struct Params { ... gravity_on, G, softening, Nr_bins, xc, yc, floor_rho, gamma, floor_P ... };
//   struct Grid2D { int Nx, Ny; double dx, dy; std::vector<double> x,y; };
//   struct U2D { std::vector<double> rho,mx,my,E; };
//   static inline size_t idx(int i,int j,int Nx);

// Computes cell-centered gravitational acceleration (gx, gy) using a spherical/enclosed-mass approximation:
//   M(<r) built by binning cell masses into radial shells around (xc,yc).
//
// NOTE: This is an approximation suitable for "toy CCSN-like" demos: 2D hydro with a radial gravity profile
// derived from a spherical average. It is NOT a true 2D Poisson solve.
static inline void gravity_accel_from_enclosed_mass(
    const U2D& U, const Grid2D& g, const Params& p,
    std::vector<double>& gx, std::vector<double>& gy
){
    const size_t N = (size_t)g.Nx * (size_t)g.Ny;
    gx.assign(N, 0.0);
    gy.assign(N, 0.0);

    if (!p.gravity_on) return;

    // r_max over the domain
    double rmax = 0.0;
    for (int j = 0; j < g.Ny; ++j){
        for (int i = 0; i < g.Nx; ++i){
            const double dx = g.x[i] - p.xc;
            const double dy = g.y[j] - p.yc;
            rmax = std::max(rmax, std::sqrt(dx*dx + dy*dy));
        }
    }
    rmax = std::max(rmax, 1e-12);

    const int Nr = std::max(8, p.Nr_bins);
    const double dr = rmax / (double)Nr;

    std::vector<double> m_shell((size_t)Nr, 0.0);

    // deposit cell masses into shells
    const double dA = g.dx * g.dy;
    for (int j = 0; j < g.Ny; ++j){
        for (int i = 0; i < g.Nx; ++i){
            const size_t q = idx(i,j,g.Nx);
            const double rho = std::max(U.rho[q], p.floor_rho);

            const double dx = g.x[i] - p.xc;
            const double dy = g.y[j] - p.yc;
            const double r  = std::sqrt(dx*dx + dy*dy);

            int b = (int)std::floor(r / dr);
            if (b < 0) b = 0;
            if (b >= Nr) b = Nr - 1;

            m_shell[(size_t)b] += rho * dA;
        }
    }

    // enclosed mass prefix sum
    std::vector<double> Menc((size_t)Nr, 0.0);
    double run = 0.0;
    for (int b = 0; b < Nr; ++b){
        run += m_shell[(size_t)b];
        Menc[(size_t)b] = run;
    }

    // acceleration: g = -G M(<r) r_vec / (r^2 + eps^2)^(3/2)
    const double eps2 = p.softening * p.softening;

    for (int j = 0; j < g.Ny; ++j){
        for (int i = 0; i < g.Nx; ++i){
            const size_t q = idx(i,j,g.Nx);

            const double dx = g.x[i] - p.xc;
            const double dy = g.y[j] - p.yc;
            const double r2 = dx*dx + dy*dy;

            const double r  = std::sqrt(r2);
            int b = (int)std::floor(r / dr);
            if (b < 0) b = 0;
            if (b >= Nr) b = Nr - 1;

            const double denom = std::pow(r2 + eps2, 1.5);   // (r^2 + eps^2)^(3/2)
            if (denom <= 0.0) { gx[q] = 0.0; gy[q] = 0.0; continue; }

            const double gcoef = -p.G * Menc[(size_t)b] / denom;

            gx[q] = gcoef * dx;
            gy[q] = gcoef * dy;
        }
    }
}

// Adds Newtonian gravity source terms to a RHS accumulator dU (conservative form):
//   d(mx)/dt += rho * gx
//   d(my)/dt += rho * gy
//   d(E)/dt  += rho * v · g
// and d(rho)/dt unchanged.
static inline void add_gravity_source(
    const U2D& U, const Grid2D& /*g*/, const Params& p,
    const std::vector<double>& gx, const std::vector<double>& gy,
    U2D& dU
){
    if (!p.gravity_on) return;

    for (size_t q = 0; q < U.rho.size(); ++q){
        const double rho = std::max(U.rho[q], p.floor_rho);
        const double mx  = U.mx[q];
        const double my  = U.my[q];

        const double ux = mx / rho;
        const double uy = my / rho;

        dU.mx[q] += rho * gx[q];
        dU.my[q] += rho * gy[q];
        dU.E[q]  += rho * (ux * gx[q] + uy * gy[q]);
    }
}
