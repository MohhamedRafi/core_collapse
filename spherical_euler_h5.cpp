#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "hdf5.h"

// -------------------- Params --------------------
struct Params {
    double gamma = 1.4;
    double cfl = 0.4;
    double rmin = 1.0;
    double rmax = 10.0;
    int    N = 400;
    double t_end = 10.0;
    int    store_every = 5;
    unsigned seed = 42;

    double floor_rho = 1e-12;
    double floor_P   = 1e-12;
};

struct State {
    std::vector<double> rho;
    std::vector<double> mom;
    std::vector<double> E;
};

static inline double sqr(double x) { return x * x; }

// -------------------- EOS / conversions --------------------
static inline void cons_to_prim_cell(
    double rho_c, double mom_c, double E_c,
    double gamma, double floor_rho, double floor_P,
    double &rho, double &v, double &P
){
    rho = std::max(rho_c, floor_rho);
    v = mom_c / rho;
    double kinetic = 0.5 * rho * v * v;
    double eint = (E_c - kinetic) / rho;
    P = (gamma - 1.0) * rho * eint;
    P = std::max(P, floor_P);
}

static inline double sound_speed(double rho, double P, double gamma) {
    return std::sqrt(gamma * P / rho);
}

static inline void flux_cell(
    double rho_c, double mom_c, double E_c,
    double gamma, double floor_rho, double floor_P,
    double &F0, double &F1, double &F2
){
    double rho, v, P;
    cons_to_prim_cell(rho_c, mom_c, E_c, gamma, floor_rho, floor_P, rho, v, P);
    F0 = rho * v;
    F1 = rho * v * v + P;
    F2 = (E_c + P) * v;
}

// -------------------- Riemann solver (HLL) --------------------
static inline void hll_flux(
    const double UL[3], const double UR[3],
    const Params &par,
    double Fhat[3]
){
    double rhoL, vL, PL;
    double rhoR, vR, PR;
    cons_to_prim_cell(UL[0], UL[1], UL[2], par.gamma, par.floor_rho, par.floor_P, rhoL, vL, PL);
    cons_to_prim_cell(UR[0], UR[1], UR[2], par.gamma, par.floor_rho, par.floor_P, rhoR, vR, PR);

    double aL = sound_speed(rhoL, PL, par.gamma);
    double aR = sound_speed(rhoR, PR, par.gamma);

    double SL = std::min(vL - aL, vR - aR);
    double SR = std::max(vL + aL, vR + aR);

    double FL[3], FR[3];
    flux_cell(UL[0], UL[1], UL[2], par.gamma, par.floor_rho, par.floor_P, FL[0], FL[1], FL[2]);
    flux_cell(UR[0], UR[1], UR[2], par.gamma, par.floor_rho, par.floor_P, FR[0], FR[1], FR[2]);

    if (SL >= 0.0) {
        Fhat[0]=FL[0]; Fhat[1]=FL[1]; Fhat[2]=FL[2];
        return;
    }
    if (SR <= 0.0) {
        Fhat[0]=FR[0]; Fhat[1]=FR[1]; Fhat[2]=FR[2];
        return;
    }

    Fhat[0] = (SR*FL[0] - SL*FR[0] + SL*SR*(UR[0]-UL[0])) / (SR - SL);
    Fhat[1] = (SR*FL[1] - SL*FR[1] + SL*SR*(UR[1]-UL[1])) / (SR - SL);
    Fhat[2] = (SR*FL[2] - SL*FR[2] + SL*SR*(UR[2]-UL[2])) / (SR - SL);
}

// -------------------- Reconstruction --------------------
static inline double minmod(double a, double b) {
    if (a*b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

static void apply_bc_reflect_inner_outflow_outer(State &U) {
    const int Ntot = (int)U.rho.size();

    // inner reflect
    U.rho[0] = U.rho[3];
    U.rho[1] = U.rho[2];
    U.E[0]   = U.E[3];
    U.E[1]   = U.E[2];
    U.mom[0] = -U.mom[3];
    U.mom[1] = -U.mom[2];

    // outer outflow
    U.rho[Ntot-1] = U.rho[Ntot-4];
    U.rho[Ntot-2] = U.rho[Ntot-3];
    U.mom[Ntot-1] = U.mom[Ntot-4];
    U.mom[Ntot-2] = U.mom[Ntot-3];
    U.E[Ntot-1]   = U.E[Ntot-4];
    U.E[Ntot-2]   = U.E[Ntot-3];
}

static void reconstruct_minmod(
    const State &U,
    std::vector<double> &UL_rho, std::vector<double> &UL_mom, std::vector<double> &UL_E,
    std::vector<double> &UR_rho, std::vector<double> &UR_mom, std::vector<double> &UR_E
){
    const int Ntot = (int)U.rho.size();
    const int Nif = Ntot + 1;

    UL_rho.assign(Nif, 0.0); UL_mom.assign(Nif, 0.0); UL_E.assign(Nif, 0.0);
    UR_rho.assign(Nif, 0.0); UR_mom.assign(Nif, 0.0); UR_E.assign(Nif, 0.0);

    std::vector<double> sr(Ntot, 0.0), sm(Ntot, 0.0), sE(Ntot, 0.0);
    for (int i = 1; i < Ntot-1; ++i) {
        sr[i] = minmod(U.rho[i] - U.rho[i-1], U.rho[i+1] - U.rho[i]);
        sm[i] = minmod(U.mom[i] - U.mom[i-1], U.mom[i+1] - U.mom[i]);
        sE[i] = minmod(U.E[i]   - U.E[i-1],   U.E[i+1]   - U.E[i]);
    }

    std::vector<double> Lr(Ntot), Rr(Ntot), Lm(Ntot), Rm(Ntot), LE(Ntot), RE(Ntot);
    for (int i = 0; i < Ntot; ++i) {
        Lr[i] = U.rho[i] - 0.5 * sr[i];
        Rr[i] = U.rho[i] + 0.5 * sr[i];
        Lm[i] = U.mom[i] - 0.5 * sm[i];
        Rm[i] = U.mom[i] + 0.5 * sm[i];
        LE[i] = U.E[i]   - 0.5 * sE[i];
        RE[i] = U.E[i]   + 0.5 * sE[i];
    }

    for (int i = 0; i < Nif; ++i) {
        if (i >= 1) {
            UL_rho[i] = Rr[i-1];
            UL_mom[i] = Rm[i-1];
            UL_E[i]   = RE[i-1];
        }
        if (i <= Ntot-1) {
            UR_rho[i] = Lr[i];
            UR_mom[i] = Lm[i];
            UR_E[i]   = LE[i];
        }
    }
}

// -------------------- RHS (spherical finite-volume) --------------------
static void rhs(
    const State &U_in,
    const std::vector<double> &r_centers,
    double dr,
    const Params &par,
    State &dUdt
){
    State U = U_in;
    apply_bc_reflect_inner_outflow_outer(U);

    const int Ntot = (int)U.rho.size();
    const int Nif = Ntot + 1;

    std::vector<double> UL_rho, UL_mom, UL_E, UR_rho, UR_mom, UR_E;
    reconstruct_minmod(U, UL_rho, UL_mom, UL_E, UR_rho, UR_mom, UR_E);

    std::vector<double> r_if(Nif, 0.0);
    for (int i = 1; i < Nif-1; ++i) r_if[i] = 0.5 * (r_centers[i] + r_centers[i-1]);
    r_if[0] = r_centers[0] - 0.5 * dr;
    r_if[Nif-1] = r_centers[Ntot-1] + 0.5 * dr;

    std::vector<double> F0(Nif,0.0), F1(Nif,0.0), F2(Nif,0.0);
    for (int i = 0; i < Nif; ++i) {
        double UL[3] = {UL_rho[i], UL_mom[i], UL_E[i]};
        double UR[3] = {UR_rho[i], UR_mom[i], UR_E[i]};
        double Fhat[3];
        hll_flux(UL, UR, par, Fhat);
        F0[i]=Fhat[0]; F1[i]=Fhat[1]; F2[i]=Fhat[2];
    }

    dUdt.rho.assign(Ntot, 0.0);
    dUdt.mom.assign(Ntot, 0.0);
    dUdt.E.assign(Ntot, 0.0);

    for (int i = 0; i < Ntot; ++i) {
        double r = r_centers[i];
        double V = sqr(r) * dr; // 4π cancels
        double A_ip = sqr(r_if[i+1]);
        double A_im = sqr(r_if[i]);

        double div0 = (A_ip*F0[i+1] - A_im*F0[i]) / V;
        double div1 = (A_ip*F1[i+1] - A_im*F1[i]) / V;
        double div2 = (A_ip*F2[i+1] - A_im*F2[i]) / V;

        double rho_p, v_p, P_p;
        cons_to_prim_cell(U.rho[i], U.mom[i], U.E[i], par.gamma, par.floor_rho, par.floor_P, rho_p, v_p, P_p);
        double S1 = 2.0 * P_p / std::max(r, 1e-12);

        dUdt.rho[i] = -div0;
        dUdt.mom[i] = -div1 + S1;
        dUdt.E[i]   = -div2;
    }
}

static double compute_dt_phys(const State &U, const Params &par, double dr) {
    double smax = 0.0;
    const int N = (int)U.rho.size();
    for (int i = 0; i < N; ++i) {
        double rho, v, P;
        cons_to_prim_cell(U.rho[i], U.mom[i], U.E[i], par.gamma, par.floor_rho, par.floor_P, rho, v, P);
        double a = sound_speed(rho, P, par.gamma);
        smax = std::max(smax, std::abs(v) + a);
    }
    smax = std::max(smax, 1e-12);
    return par.cfl * dr / smax;
}

// -------------------- Random IC --------------------
static std::vector<double> smooth_random_field(const std::vector<double> &r, std::mt19937 &gen, int n_modes) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 2.0*M_PI);

    double rmin = r.front();
    double rmax = r.back();

    std::vector<double> f(r.size(), 0.0);
    for (int k = 1; k <= n_modes; ++k) {
        double amp = normal(gen) / (double)(k*k);
        double phase = uni(gen);
        for (size_t i = 0; i < r.size(); ++i) {
            double x = (r[i] - rmin) / (rmax - rmin);
            f[i] += amp * std::sin(2.0*M_PI*k*x + phase);
        }
    }
    double mean = 0.0;
    for (double v : f) mean += v;
    mean /= (double)f.size();
    for (double &v : f) v -= mean;

    double mx = 0.0;
    for (double v : f) mx = std::max(mx, std::abs(v));
    mx = std::max(mx, 1e-12);
    for (double &v : f) v /= mx;

    return f;
}

static State make_initial_condition(const std::vector<double> &r_phys, const Params &par) {
    std::mt19937 gen(par.seed);

    const size_t N = r_phys.size();
    auto fr = smooth_random_field(r_phys, gen, 10);
    auto fp = smooth_random_field(r_phys, gen, 10);
    auto fv = smooth_random_field(r_phys, gen, 6);

    double rho0 = 1.0;
    double P0   = 1.0;
    double v0   = 0.0;

    double amp_rho = 0.35;
    double amp_P   = 0.35;

    std::vector<double> rho(N), P(N), v(N);
    for (size_t i = 0; i < N; ++i) {
        rho[i] = rho0 * std::exp(amp_rho * fr[i]);
        P[i]   = P0   * std::exp(amp_P   * fp[i]);
    }

    double cs0 = std::sqrt(par.gamma * P0 / rho0);
    double amp_v = 0.25 * cs0;
    for (size_t i = 0; i < N; ++i) v[i] = v0 + amp_v * fv[i];

    State U;
    U.rho.resize(N);
    U.mom.resize(N);
    U.E.resize(N);

    for (size_t i = 0; i < N; ++i) {
        U.rho[i] = rho[i];
        U.mom[i] = rho[i] * v[i];
        double e_int = P[i] / ((par.gamma - 1.0) * rho[i]);
        U.E[i] = rho[i] * e_int + 0.5 * rho[i] * v[i] * v[i];
    }
    return U;
}

// -------------------- HDF5 writing helpers --------------------
static void h5_check(herr_t status, const char* what) {
    if (status < 0) {
        std::cerr << "HDF5 error in " << what << "\n";
        std::exit(1);
    }
}

static void write_1d_double(hid_t file, const char* name, const std::vector<double>& x) {
    hsize_t dims[1] = { (hsize_t)x.size() };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset  = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x.data()), "H5Dwrite 1D");
    H5Dclose(dset);
    H5Sclose(space);
}

static void write_2d_double_rowmajor(hid_t file, const char* name, const std::vector<double>& data, hsize_t n0, hsize_t n1) {
    // data is row-major of shape (n0, n1)
    hsize_t dims[2] = { n0, n1 };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset  = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()), "H5Dwrite 2D");
    H5Dclose(dset);
    H5Sclose(space);
}

int main() {
    Params par;

    const int N = par.N;
    const int Ntot = N + 4;
    const double dr = (par.rmax - par.rmin) / (double)N;

    std::vector<double> r_phys(N);
    for (int i = 0; i < N; ++i) r_phys[i] = par.rmin + (i + 0.5) * dr;

    std::vector<double> r_all(Ntot);
    for (int i = 0; i < N; ++i) r_all[i+2] = r_phys[i];
    r_all[0] = r_phys[0] - 2*dr;
    r_all[1] = r_phys[0] - 1*dr;
    r_all[Ntot-2] = r_phys[N-1] + 1*dr;
    r_all[Ntot-1] = r_phys[N-1] + 2*dr;

    State U;
    U.rho.assign(Ntot, 0.0);
    U.mom.assign(Ntot, 0.0);
    U.E.assign(Ntot, 0.0);

    State U0 = make_initial_condition(r_phys, par);
    for (int i = 0; i < N; ++i) {
        U.rho[i+2] = U0.rho[i];
        U.mom[i+2] = U0.mom[i];
        U.E[i+2]   = U0.E[i];
    }
    apply_bc_reflect_inner_outflow_outer(U);

    // storage (packed row-major)
    std::vector<double> times;
    std::vector<double> rho_hist;   // (Nt, N)
    std::vector<double> P_hist;     // (Nt, N)
    std::vector<double> Mach_hist;  // (Nt, N)

    auto store_snapshot = [&](double tcur) {
        times.push_back(tcur);
        const size_t base = (times.size()-1) * (size_t)N;
        rho_hist.resize(times.size() * (size_t)N);
        P_hist.resize(times.size() * (size_t)N);
        Mach_hist.resize(times.size() * (size_t)N);

        for (int i = 0; i < N; ++i) {
            double rr, vv, PP;
            cons_to_prim_cell(U.rho[i+2], U.mom[i+2], U.E[i+2],
                              par.gamma, par.floor_rho, par.floor_P,
                              rr, vv, PP);
            double cs = sound_speed(rr, PP, par.gamma);
            rho_hist[base + (size_t)i] = rr;
            P_hist[base + (size_t)i]   = PP;
            Mach_hist[base + (size_t)i] = std::abs(vv) / cs;
        }
    };

    double t = 0.0;
    store_snapshot(t);

    int step = 0;
    while (t < par.t_end - 1e-14) {
        // build physical-only view for dt
        State U_phys;
        U_phys.rho.resize(N);
        U_phys.mom.resize(N);
        U_phys.E.resize(N);
        for (int i = 0; i < N; ++i) {
            U_phys.rho[i] = U.rho[i+2];
            U_phys.mom[i] = U.mom[i+2];
            U_phys.E[i]   = U.E[i+2];
        }

        double dt = compute_dt_phys(U_phys, par, dr);
        if (t + dt > par.t_end) dt = par.t_end - t;

        State k1, k2, U1;
        rhs(U, r_all, dr, par, k1);

        U1.rho.resize(Ntot);
        U1.mom.resize(Ntot);
        U1.E.resize(Ntot);
        for (int i = 0; i < Ntot; ++i) {
            U1.rho[i] = U.rho[i] + dt * k1.rho[i];
            U1.mom[i] = U.mom[i] + dt * k1.mom[i];
            U1.E[i]   = U.E[i]   + dt * k1.E[i];
        }
        apply_bc_reflect_inner_outflow_outer(U1);

        rhs(U1, r_all, dr, par, k2);

        for (int i = 0; i < Ntot; ++i) {
            U.rho[i] = 0.5 * (U.rho[i] + U1.rho[i] + dt * k2.rho[i]);
            U.mom[i] = 0.5 * (U.mom[i] + U1.mom[i] + dt * k2.mom[i]);
            U.E[i]   = 0.5 * (U.E[i]   + U1.E[i]   + dt * k2.E[i]);
        }
        apply_bc_reflect_inner_outflow_outer(U);

        t += dt;
        step += 1;

        if (step % par.store_every == 0 || t >= par.t_end - 1e-14) {
            store_snapshot(t);
            std::cerr << "t=" << t << " snapshots=" << times.size() << "\n";
        }
    }

    // ---- write HDF5
    hid_t file = H5Fcreate("out.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Failed to create out.h5\n";
        return 1;
    }

    write_1d_double(file, "/r", r_phys);
    write_1d_double(file, "/t", times);

    const hsize_t Nt = (hsize_t)times.size();
    const hsize_t Nr = (hsize_t)N;

    write_2d_double_rowmajor(file, "/rho",  rho_hist,  Nt, Nr);
    write_2d_double_rowmajor(file, "/P",    P_hist,    Nt, Nr);
    write_2d_double_rowmajor(file, "/Mach", Mach_hist, Nt, Nr);

    H5Fclose(file);

    std::cout << "Wrote out.h5 with datasets: /r /t /rho /P /Mach\n";
    return 0;
}
