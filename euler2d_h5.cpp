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

#include "hdf5.h"

struct Params {
    double gamma = 1.4;
    double cfl   = 0.45;

    // domain
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;

    int Nx = 50;
    int Ny = 50;

    // simulation control
    double t_end    = 4;   // keep small for visualization; increase later
    double dt_store = 1e-3;  // store snapshots at fixed time spacing

    unsigned seed = 42;

    // floors
    double floor_rho = 1e-10;
    double floor_P   = 1e-10;

    // boundary conditions (periodic handled by modular indexing)
    bool periodic_x = true;
    bool periodic_y = true;

    // gravity (Newtonian enclosed-mass approximation)
    bool gravity_on = true;
    double G = 0.5;           // code units; increase slowly
    double softening = 2e-3;  // avoid singularity near r=0
    int Nr_bins = 512;        // radial bins for M(<r)
    double xc = 0.5;          // gravity center
    double yc = 0.5;
};

static inline size_t idx(int i, int j, int Nx) { return (size_t)j * (size_t)Nx + (size_t)i; }

struct Grid2D {
    int Nx, Ny;
    double dx, dy;
    std::vector<double> x, y; // cell centers
};

struct U2D {
    // conservative: [rho, mx, my, E] in SoA layout
    std::vector<double> rho, mx, my, E;
};

#include "gravity_enclosed.h"

static inline void cons_to_prim(
    double rho_c, double mx_c, double my_c, double E_c,
    const Params& p,
    double &rho, double &u, double &v, double &P
){
    rho = std::max(rho_c, p.floor_rho);
    u = mx_c / rho;
    v = my_c / rho;
    double kinetic = 0.5 * rho * (u*u + v*v);

    double eint = (E_c - kinetic) / rho;
    P = (p.gamma - 1.0) * rho * eint;
    P = std::max(P, p.floor_P);
}

static inline double sound_speed(double rho, double P, const Params& p){
    return std::sqrt(p.gamma * P / rho);
}

static inline void flux_x(const double U[4], const Params& p, double F[4]){
    double rho,u,v,P;
    cons_to_prim(U[0],U[1],U[2],U[3],p,rho,u,v,P);
    F[0] = rho*u;
    F[1] = rho*u*u + P;
    F[2] = rho*u*v;
    F[3] = (U[3] + P)*u;
}

static inline void flux_y(const double U[4], const Params& p, double G[4]){
    double rho,u,v,P;
    cons_to_prim(U[0],U[1],U[2],U[3],p,rho,u,v,P);
    G[0] = rho*v;
    G[1] = rho*u*v;
    G[2] = rho*v*v + P;
    G[3] = (U[3] + P)*v;
}

static inline void hll_x(const double UL[4], const double UR[4], const Params& p, double Fhat[4]){
    double rhoL,uL,vL,PL; cons_to_prim(UL[0],UL[1],UL[2],UL[3],p,rhoL,uL,vL,PL);
    double rhoR,uR,vR,PR; cons_to_prim(UR[0],UR[1],UR[2],UR[3],p,rhoR,uR,vR,PR);
    double aL = sound_speed(rhoL,PL,p);
    double aR = sound_speed(rhoR,PR,p);

    double SL = std::min(uL - aL, uR - aR);
    double SR = std::max(uL + aL, uR + aR);

    double FL[4], FR[4];
    flux_x(UL,p,FL);
    flux_x(UR,p,FR);

    if (SL >= 0.0) { for(int k=0;k<4;k++) Fhat[k]=FL[k]; return; }
    if (SR <= 0.0) { for(int k=0;k<4;k++) Fhat[k]=FR[k]; return; }

    for(int k=0;k<4;k++){
        Fhat[k] = (SR*FL[k] - SL*FR[k] + SL*SR*(UR[k]-UL[k]))/(SR-SL);
    }
}

static inline void hll_y(const double UL[4], const double UR[4], const Params& p, double Ghat[4]){
    double rhoL,uL,vL,PL; cons_to_prim(UL[0],UL[1],UL[2],UL[3],p,rhoL,uL,vL,PL);
    double rhoR,uR,vR,PR; cons_to_prim(UR[0],UR[1],UR[2],UR[3],p,rhoR,uR,vR,PR);
    double aL = sound_speed(rhoL,PL,p);
    double aR = sound_speed(rhoR,PR,p);

    double SL = std::min(vL - aL, vR - aR);
    double SR = std::max(vL + aL, vR + aR);

    double GL[4], GR[4];
    flux_y(UL,p,GL);
    flux_y(UR,p,GR);

    if (SL >= 0.0) { for(int k=0;k<4;k++) Ghat[k]=GL[k]; return; }
    if (SR <= 0.0) { for(int k=0;k<4;k++) Ghat[k]=GR[k]; return; }

    for(int k=0;k<4;k++){
        Ghat[k] = (SR*GL[k] - SL*GR[k] + SL*SR*(UR[k]-UL[k]))/(SR-SL);
    }
}

static inline double minmod(double a, double b){
    if (a*b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

static void apply_periodic_bc(U2D& U, const Params& p){
    (void)U; (void)p;
}

static void enforce_floors(U2D& U, const Params& p){
    const double gm1 = (p.gamma - 1.0);
    const double rho_floor = p.floor_rho;
    const double P_floor   = p.floor_P;

    for (size_t q = 0; q < U.rho.size(); ++q) {
        if (!std::isfinite(U.rho[q])) U.rho[q] = rho_floor;
        if (!std::isfinite(U.mx[q]))  U.mx[q]  = 0.0;
        if (!std::isfinite(U.my[q]))  U.my[q]  = 0.0;
        if (!std::isfinite(U.E[q]))   U.E[q]   = P_floor / gm1;

        if (U.rho[q] < rho_floor) U.rho[q] = rho_floor;

        const double rho = U.rho[q];
        const double mx  = U.mx[q];
        const double my  = U.my[q];
        const double K   = 0.5 * (mx*mx + my*my) / rho;

        const double Emin = K + P_floor / gm1;
        if (U.E[q] < Emin) U.E[q] = Emin;
    }
}

static double compute_dt(const U2D& U, const Grid2D& g, const Params& p){
    double smax = 0.0;

    for(int j=0;j<g.Ny;j++){
        for(int i=0;i<g.Nx;i++){
            size_t q = idx(i,j,g.Nx);

            double rho = U.rho[q];
            double mx  = U.mx[q];
            double my  = U.my[q];
            double E   = U.E[q];

            if (!std::isfinite(rho) || !std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(E)) continue;

            double rho_p,u,v,P;
            cons_to_prim(rho,mx,my,E,p,rho_p,u,v,P);
            double a = sound_speed(rho_p,P,p);
            if (!std::isfinite(a) || a < 0.0) continue;

            double sx = std::abs(u) + a;
            double sy = std::abs(v) + a;
            if (std::isfinite(sx)) smax = std::max(smax, sx);
            if (std::isfinite(sy)) smax = std::max(smax, sy);
        }
    }

    if (!std::isfinite(smax) || smax <= 0.0) return 0.0;

    double dt_x = p.cfl * g.dx / smax;
    double dt_y = p.cfl * g.dy / smax;
    double dt = std::min(dt_x, dt_y);
    if (!std::isfinite(dt) || dt <= 0.0) return 0.0;
    return dt;
}

static void init_random_ic(U2D& U, const Grid2D& g, const Params& p){
    std::mt19937 gen(p.seed);
    std::normal_distribution<double> N01(0.0,1.0);

    auto smooth = [&](double x, double y){
        double s = 0.0;
        for(int kx=1;kx<=6;kx++){
            for(int ky=1;ky<=6;ky++){
                double ax = N01(gen)/(kx*kx + ky*ky);
                double ph = 2.0*M_PI*((double)std::uniform_real_distribution<double>(0,1)(gen));
                s += ax * std::sin(2.0*M_PI*kx*x + 2.0*M_PI*ky*y + ph);
            }
        }
        return s;
    };

    const double rho0 = 1.0;
    const double P0   = 1.0;

    std::vector<double> psi((size_t)g.Nx*(size_t)g.Ny, 0.0);
    for(int j=0;j<g.Ny;j++){
        for(int i=0;i<g.Nx;i++){
            double x = (g.x[i]-p.x0)/(p.x1-p.x0);
            double y = (g.y[j]-p.y0)/(p.y1-p.y0);
            psi[idx(i,j,g.Nx)] = 0.2 * smooth(x,y);
        }
    }

    auto imod = [&](int a, int m){ a%=m; if(a<0) a+=m; return a; };

    for(int j=0;j<g.Ny;j++){
        for(int i=0;i<g.Nx;i++){
            size_t q = idx(i,j,g.Nx);
            double x = (g.x[i]-p.x0)/(p.x1-p.x0);
            double y = (g.y[j]-p.y0)/(p.y1-p.y0);

            double fr = smooth(x,y);
            double fp = smooth(x+0.17,y+0.31);

            double rho = rho0 * std::exp(0.35 * fr);
            double P   = P0   * std::exp(0.35 * fp);

            int ip = imod(i+1,g.Nx), im = imod(i-1,g.Nx);
            int jp = imod(j+1,g.Ny), jm = imod(j-1,g.Ny);

            double dpsidy = (psi[idx(i,jp,g.Nx)] - psi[idx(i,jm,g.Nx)])/(2.0*g.dy);
            double dpsidx = (psi[idx(ip,j,g.Nx)] - psi[idx(im,j,g.Nx)])/(2.0*g.dx);

            double u = dpsidy;
            double v = -dpsidx;

            U.rho[q] = rho;
            U.mx[q]  = rho*u;
            U.my[q]  = rho*v;

            double e_int = P/((p.gamma-1.0)*rho);
            U.E[q] = rho*e_int + 0.5*rho*(u*u+v*v);
        }
    }

    enforce_floors(U, p);
}

static void rhs_split_x(const U2D& U, const Grid2D& g, const Params& p, U2D& dU){
    dU.rho.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.mx.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.my.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.E.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);

    auto imod = [&](int a, int m){ a%=m; if(a<0) a+=m; return a; };

    for(int j=0;j<g.Ny;j++){
        for(int i=0;i<g.Nx;i++){
            int ip = imod(i+1, g.Nx);
            int im = imod(i-1, g.Nx);

            auto getU = [&](int ii)->std::array<double,4>{
                size_t q = idx(ii,j,g.Nx);
                return {U.rho[q],U.mx[q],U.my[q],U.E[q]};
            };

            auto Ui  = getU(i);
            auto Uim = getU(im);
            auto Uip = getU(ip);

            double slope_i[4];
            for(int k=0;k<4;k++){
                slope_i[k] = minmod(Ui[k]-Uim[k], Uip[k]-Ui[k]);
            }

            int ipp = imod(i+2,g.Nx);
            auto Ui1  = getU(ip);
            auto Ui1m = getU(i);
            auto Ui1p = getU(ipp);

            double slope_i1[4];
            for(int k=0;k<4;k++){
                slope_i1[k] = minmod(Ui1[k]-Ui1m[k], Ui1p[k]-Ui1[k]);
            }

            double UL[4], UR[4];
            for(int k=0;k<4;k++){
                UL[k] = Ui[k]  + 0.5*slope_i[k];
                UR[k] = Ui1[k] - 0.5*slope_i1[k];
            }

            double Fhat[4];
            hll_x(UL, UR, p, Fhat);

            size_t q  = idx(i,j,g.Nx);
            size_t q1 = idx(ip,j,g.Nx);

            dU.rho[q]  -= Fhat[0]/g.dx;
            dU.mx[q]   -= Fhat[1]/g.dx;
            dU.my[q]   -= Fhat[2]/g.dx;
            dU.E[q]    -= Fhat[3]/g.dx;

            dU.rho[q1] += Fhat[0]/g.dx;
            dU.mx[q1]  += Fhat[1]/g.dx;
            dU.my[q1]  += Fhat[2]/g.dx;
            dU.E[q1]   += Fhat[3]/g.dx;
        }
    }
}

static void rhs_split_y(const U2D& U, const Grid2D& g, const Params& p, U2D& dU){
    dU.rho.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.mx.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.my.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);
    dU.E.assign((size_t)g.Nx*(size_t)g.Ny, 0.0);

    auto imod = [&](int a, int m){ a%=m; if(a<0) a+=m; return a; };

    for(int j=0;j<g.Ny;j++){
        int jp = imod(j+1, g.Ny);
        int jm = imod(j-1, g.Ny);

        for(int i=0;i<g.Nx;i++){
            auto getU = [&](int jj)->std::array<double,4>{
                size_t q = idx(i,jj,g.Nx);
                return {U.rho[q],U.mx[q],U.my[q],U.E[q]};
            };

            auto Uj  = getU(j);
            auto Ujm = getU(jm);
            auto Ujp = getU(jp);

            double slope_j[4];
            for(int k=0;k<4;k++){
                slope_j[k] = minmod(Uj[k]-Ujm[k], Ujp[k]-Uj[k]);
            }

            int jpp = imod(j+2,g.Ny);
            auto Uj1  = getU(jp);
            auto Uj1m = getU(j);
            auto Uj1p = getU(jpp);

            double slope_j1[4];
            for(int k=0;k<4;k++){
                slope_j1[k] = minmod(Uj1[k]-Uj1m[k], Uj1p[k]-Uj1[k]);
            }

            double UL[4], UR[4];
            for(int k=0;k<4;k++){
                UL[k] = Uj[k]  + 0.5*slope_j[k];
                UR[k] = Uj1[k] - 0.5*slope_j1[k];
            }

            double Ghat[4];
            hll_y(UL, UR, p, Ghat);

            size_t q  = idx(i,j,g.Nx);
            size_t q1 = idx(i,jp,g.Nx);

            dU.rho[q]  -= Ghat[0]/g.dy;
            dU.mx[q]   -= Ghat[1]/g.dy;
            dU.my[q]   -= Ghat[2]/g.dy;
            dU.E[q]    -= Ghat[3]/g.dy;

            dU.rho[q1] += Ghat[0]/g.dy;
            dU.mx[q1]  += Ghat[1]/g.dy;
            dU.my[q1]  += Ghat[2]/g.dy;
            dU.E[q1]   += Ghat[3]/g.dy;
        }
    }
}

// ---------------- HDF5 helpers ----------------
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

int main(){
    Params p;

    Grid2D g;
    g.Nx = p.Nx; g.Ny = p.Ny;
    g.dx = (p.x1 - p.x0) / p.Nx;
    g.dy = (p.y1 - p.y0) / p.Ny;
    g.x.resize(g.Nx);
    g.y.resize(g.Ny);
    for(int i=0;i<g.Nx;i++) g.x[i] = p.x0 + (i+0.5)*g.dx;
    for(int j=0;j<g.Ny;j++) g.y[j] = p.y0 + (j+0.5)*g.dy;

    U2D U;
    U.rho.resize((size_t)g.Nx*(size_t)g.Ny);
    U.mx.resize((size_t)g.Nx*(size_t)g.Ny);
    U.my.resize((size_t)g.Nx*(size_t)g.Ny);
    U.E.resize((size_t)g.Nx*(size_t)g.Ny);

    init_random_ic(U,g,p);

    // storage (Nt,Ny,Nx) row-major
    std::vector<double> times;
    std::vector<double> rho_hist, P_hist, Mach_hist;

    auto store = [&](double tcur){
        times.push_back(tcur);
        size_t Nt = times.size();
        size_t plane = (size_t)g.Nx*(size_t)g.Ny;
        rho_hist.resize(Nt*plane);
        P_hist.resize(Nt*plane);
        Mach_hist.resize(Nt*plane);

        size_t base = (Nt-1)*plane;
        for(int j=0;j<g.Ny;j++){
            for(int i=0;i<g.Nx;i++){
                size_t q = idx(i,j,g.Nx);
                double rho,u,v,P;
                cons_to_prim(U.rho[q],U.mx[q],U.my[q],U.E[q],p,rho,u,v,P);
                double cs = sound_speed(rho,P,p);
                rho_hist[base + q]  = rho;
                P_hist[base + q]    = P;
                Mach_hist[base + q] = std::sqrt(u*u+v*v)/cs;
            }
        }
    };

    double t = 0.0;
    double next_store = 0.0;

    store(t);
    next_store += p.dt_store;

    U2D kx, ky, U1, kx1, ky1, kg;
    auto alloc_like = [&](U2D& A){
        A.rho.resize((size_t)g.Nx*(size_t)g.Ny);
        A.mx.resize((size_t)g.Nx*(size_t)g.Ny);
        A.my.resize((size_t)g.Nx*(size_t)g.Ny);
        A.E.resize((size_t)g.Nx*(size_t)g.Ny);
    };
    alloc_like(kx); alloc_like(ky); alloc_like(U1);
    alloc_like(kx1); alloc_like(ky1); alloc_like(kg);

    std::vector<double> gx, gy;

    int step = 0;
    while(t < p.t_end - 1e-14){
        double dt = compute_dt(U,g,p);
        // std::cerr << std::setprecision(17)
        //   << "t=" << t
        //   << " dt=" << dt
        //   << " t+dt=" << (t+dt)
        //   << "\n";


        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "Bad dt=" << dt << " at t=" << t << " (abort)\n";
            return 1;
        }
        if(t + dt > p.t_end) dt = p.t_end - t;

        // ---- stage A: RHS at U ----
        rhs_split_x(U,g,p,kx);
        rhs_split_y(U,g,p,ky);

        gravity_accel_from_enclosed_mass(U, g, p, gx, gy);
        
        /* stuff */
        double gmax = 0.0, gmean = 0.0;
        for (size_t q=0;q<gx.size();q++){
            double gm = std::sqrt(gx[q]*gx[q] + gy[q]*gy[q]);
            gmax = std::max(gmax, gm);
            gmean += gm;
        }
        gmean /= (double)gx.size();
        // std::cerr << "  gravity: gmax=" << gmax << " gmean=" << gmean << "\n";


        kg.rho.assign(U.rho.size(), 0.0);
        kg.mx.assign(U.rho.size(), 0.0);
        kg.my.assign(U.rho.size(), 0.0);
        kg.E.assign(U.rho.size(), 0.0);
        add_gravity_source(U, g, p, gx, gy, kg);

        // predictor U -> U1
        for(size_t q=0;q<U.rho.size();q++){
            U1.rho[q] = U.rho[q] + dt*(kx.rho[q] + ky.rho[q] + kg.rho[q]);
            U1.mx[q]  = U.mx[q]  + dt*(kx.mx[q]  + ky.mx[q]  + kg.mx[q]);
            U1.my[q]  = U.my[q]  + dt*(kx.my[q]  + ky.my[q]  + kg.my[q]);
            U1.E[q]   = U.E[q]   + dt*(kx.E[q]   + ky.E[q]   + kg.E[q]);
        }
        apply_periodic_bc(U1,p);
        enforce_floors(U1,p);

        // ---- stage B: RHS at U1 ----
        rhs_split_x(U1,g,p,kx1);
        rhs_split_y(U1,g,p,ky1);

        gravity_accel_from_enclosed_mass(U1, g, p, gx, gy);
        kg.rho.assign(U.rho.size(), 0.0);
        kg.mx.assign(U.rho.size(), 0.0);
        kg.my.assign(U.rho.size(), 0.0);
        kg.E.assign(U.rho.size(), 0.0);
        add_gravity_source(U1, g, p, gx, gy, kg);

        // corrector
        for(size_t q=0;q<U.rho.size();q++){
            U.rho[q] = 0.5*(U.rho[q] + U1.rho[q] + dt*(kx1.rho[q] + ky1.rho[q] + kg.rho[q]));
            U.mx[q]  = 0.5*(U.mx[q]  + U1.mx[q]  + dt*(kx1.mx[q]  + ky1.mx[q]  + kg.mx[q]));
            U.my[q]  = 0.5*(U.my[q]  + U1.my[q]  + dt*(kx1.my[q]  + ky1.my[q]  + kg.my[q]));
            U.E[q]   = 0.5*(U.E[q]   + U1.E[q]   + dt*(kx1.E[q]   + ky1.E[q]   + kg.E[q]));
        }
        apply_periodic_bc(U,p);
        enforce_floors(U,p);

        t += dt;
        step++;

        if (t + 1e-14 >= next_store || t >= p.t_end - 1e-14) {
            store(t);
            next_store += p.dt_store;
            std::cerr << "step=" << step
                      << " t=" << std::setprecision(10) << t
                      << " dt=" << dt
                      << " snapshots=" << times.size() << "\n";
        }
    }

    hid_t file = H5Fcreate("out2d.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(file < 0){ std::cerr << "Failed to create out2d.h5\n"; return 1; }

    write_1d(file, "/x", g.x);
    write_1d(file, "/y", g.y);
    write_1d(file, "/t", times);

    hsize_t Nt = (hsize_t)times.size();
    hsize_t Ny = (hsize_t)g.Ny;
    hsize_t Nx = (hsize_t)g.Nx;

    write_3d_rowmajor(file, "/rho",  rho_hist,  Nt, Ny, Nx);
    write_3d_rowmajor(file, "/P",    P_hist,    Nt, Ny, Nx);
    write_3d_rowmajor(file, "/Mach", Mach_hist, Nt, Ny, Nx);

    H5Fclose(file);
    std::cout << "Wrote out2d.h5 datasets: /x /y /t /rho /P /Mach\n";
    return 0;
}
