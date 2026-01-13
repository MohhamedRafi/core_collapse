// mhd3d_resistive_hlld_ct.cpp
//
// 3D Resistive MHD with:
//   - FV Godunov update for cell-centered conserved vars using HLLD
//   - CT (constrained transport) for face-centered Bx, By, Bz using edge-centered E
//   - Resistivity via E_res = eta * J (cell-centered curl, averaged to edges)
//   - Joule heating: dE/dt += eta |J|^2
//
// Core-collapse-style IC (toy):
//   - Central overdense core with soft edge
//   - Sub-virial pressure + inward radial velocity + optional rotation
//   - Divergence-free face fields initialized from a corner-centered vector potential Az
//
// Output: out3d_mhd.h5
//   /x (Nx), /y (Ny), /z (Nz), /t (Nt)
//   /rho, /P, /Mach, /u, /v, /w, /vabs, /Bx, /By, /Bz, /divB    all (Nt,Nz,Ny,Nx)
//
// Build:
//   clang++ -O3 -std=c++17 -Wall -Wextra -pedantic mhd3d_resistive_hlld_ct.cpp -lhdf5 -o mhd3d_resistive_hlld_ct
//
// Run:
//   ./mhd3d_resistive_hlld_ct

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

static inline size_t idx(int i, int j, int k, int Nx, int Ny) {
    return (static_cast<size_t>(k) * static_cast<size_t>(Ny) + static_cast<size_t>(j)) * static_cast<size_t>(Nx) +
           static_cast<size_t>(i);
}
static inline int imod(int i, int n){ int r=i%n; return (r<0)?r+n:r; }

struct Params {
    double gamma = 5.0/3.0;
    double cfl   = 0.4;

    double x0=0.0, x1=1.0;
    double y0=0.0, y1=1.0;
    double z0=0.0, z1=1.0;
    int Nx=48;
    int Ny=48;
    int Nz=48;

    double t_end    = 0.15;
    double dt_store = 1e-3;

    double floor_rho = 1e-10;
    double floor_P   = 1e-10;

    bool periodic_x = true;
    bool periodic_y = true;
    bool periodic_z = true;

    // resistivity (constant eta)
    double eta = 5e-4;

    // IC knobs
    double rho_bg   = 1.0;
    double rho_core = 20.0;
    double r_core   = 0.12;
    double r_edge   = 0.02;      // edge thickness
    double P0       = 0.15;      // base pressure scale
    double vin      = 1.25;      // inward speed scale (core region)
    double rot_omega = 1.0;      // rotation strength (set 0 for none)

    // magnetic seed (set via Az + uniform Bz)
    double B0x   = 0.15;  // uniform Bx target
    double B0z   = 0.05;  // uniform Bz
    double Bshear= 0.05;  // strength of div-free perturbation via Az
};

struct Ucc { // cell-centered conserved
    std::vector<double> rho, mx, my, mz, E;
};

struct Bfaces { // face-centered magnetic field
    std::vector<double> Bx_f; // (Nx+1)*Ny*Nz at (i_face, j, k)
    std::vector<double> By_f; // Nx*(Ny+1)*Nz at (i, j_face, k)
    std::vector<double> Bz_f; // Nx*Ny*(Nz+1) at (i, j, k_face)
};

struct Prim {
    double rho,u,v,w,P;
    double Bx,By,Bz;
};

static inline double minmod(double a, double b){
    if (a*b <= 0.0) return 0.0;
    return (std::abs(a)<std::abs(b))?a:b;
}
static inline double mc_limiter(double dm, double dp){
    return minmod(0.5*(dm+dp), minmod(2.0*dm, 2.0*dp));
}

static inline void enforce_floors(const Params& p, double& rho, double& mx, double& my, double& mz, double& E,
                                  double Bx, double By, double Bz){
    rho = std::max(rho, p.floor_rho);
    double u = mx/rho, v = my/rho, w = mz/rho;
    double kin = 0.5*rho*(u*u+v*v+w*w);
    double mag = 0.5*(Bx*Bx+By*By+Bz*Bz);
    double P = (p.gamma-1.0)*(E - kin - mag);
    if (P < p.floor_P){
        double eint = p.floor_P/(p.gamma-1.0);
        E = kin + mag + eint;
    }
}

static inline Prim cons_to_prim(const Params& p, double rho, double mx, double my, double mz, double E,
                                double Bx, double By, double Bz){
    rho = std::max(rho, p.floor_rho);
    double u = mx/rho, v = my/rho, w = mz/rho;
    double kin = 0.5*rho*(u*u+v*v+w*w);
    double mag = 0.5*(Bx*Bx+By*By+Bz*Bz);
    double P = (p.gamma-1.0)*(E - kin - mag);
    if (P < p.floor_P) P = p.floor_P;
    return {rho,u,v,w,P,Bx,By,Bz};
}

static inline void prim_to_cons(const Params& p, const Prim& W,
                                double& rho, double& mx, double& my, double& mz, double& E){
    rho = std::max(W.rho, p.floor_rho);
    mx = rho*W.u; my = rho*W.v; mz = rho*W.w;
    double kin = 0.5*rho*(W.u*W.u+W.v*W.v+W.w*W.w);
    double mag = 0.5*(W.Bx*W.Bx+W.By*W.By+W.Bz*W.Bz);
    double eint = W.P/(p.gamma-1.0);
    E = kin + mag + eint;
}

static inline double sound_speed2(const Params& p, const Prim& W){
    return p.gamma * W.P / std::max(W.rho, p.floor_rho);
}

static inline double fast_speed_x(const Params& p, const Prim& W){
    double a2 = sound_speed2(p,W);
    double b2 = W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz;
    double rho = std::max(W.rho, 1e-300);
    double vA2 = b2/rho;
    double vAx2 = (W.Bx*W.Bx)/rho;
    double disc = (a2+vA2)*(a2+vA2) - 4.0*a2*vAx2;
    disc = std::max(0.0, disc);
    double cf2 = 0.5*((a2+vA2) + std::sqrt(disc));
    return std::sqrt(std::max(0.0, cf2));
}

static inline double fast_speed_y(const Params& p, const Prim& W){
    double a2 = sound_speed2(p,W);
    double b2 = W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz;
    double rho = std::max(W.rho, 1e-300);
    double vA2 = b2/rho;
    double vAy2 = (W.By*W.By)/rho;
    double disc = (a2+vA2)*(a2+vA2) - 4.0*a2*vAy2;
    disc = std::max(0.0, disc);
    double cf2 = 0.5*((a2+vA2) + std::sqrt(disc));
    return std::sqrt(std::max(0.0, cf2));
}

static inline double fast_speed_z(const Params& p, const Prim& W){
    double a2 = sound_speed2(p,W);
    double b2 = W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz;
    double rho = std::max(W.rho, 1e-300);
    double vA2 = b2/rho;
    double vAz2 = (W.Bz*W.Bz)/rho;
    double disc = (a2+vA2)*(a2+vA2) - 4.0*a2*vAz2;
    disc = std::max(0.0, disc);
    double cf2 = 0.5*((a2+vA2) + std::sqrt(disc));
    return std::sqrt(std::max(0.0, cf2));
}

static inline double get_Bx_face(const Params& p, const Bfaces& B, int i_face, int j, int k){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    if (p.periodic_x) i_face=imod(i_face,Nx+1); else i_face=std::clamp(i_face,0,Nx);
    if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
    if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
    size_t f = (static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
               static_cast<size_t>(i_face);
    return B.Bx_f[f];
}

static inline double get_By_face(const Params& p, const Bfaces& B, int i, int j_face, int k){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
    if (p.periodic_y) j_face=imod(j_face,Ny+1); else j_face=std::clamp(j_face,0,Ny);
    if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
    size_t f = (static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j_face))*static_cast<size_t>(Nx) +
               static_cast<size_t>(i);
    return B.By_f[f];
}

static inline double get_Bz_face(const Params& p, const Bfaces& B, int i, int j, int k_face){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
    if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
    if (p.periodic_z) k_face=imod(k_face,Nz+1); else k_face=std::clamp(k_face,0,Nz);
    size_t f = (static_cast<size_t>(k_face)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
               static_cast<size_t>(i);
    return B.Bz_f[f];
}

static inline double bx_cc_from_faces(const Bfaces& B, int i, int j, int k, const Params& p){
    double bL = get_Bx_face(p,B,i,  j,k);
    double bR = get_Bx_face(p,B,i+1,j,k);
    return 0.5*(bL+bR);
}

static inline double by_cc_from_faces(const Bfaces& B, int i, int j, int k, const Params& p){
    double bD = get_By_face(p,B,i,j,  k);
    double bU = get_By_face(p,B,i,j+1,k);
    return 0.5*(bD+bU);
}

static inline double bz_cc_from_faces(const Bfaces& B, int i, int j, int k, const Params& p){
    double bB = get_Bz_face(p,B,i,j,k);
    double bF = get_Bz_face(p,B,i,j,k+1);
    return 0.5*(bB+bF);
}

static inline Prim get_prim_cc(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, int k){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
    if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
    if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
    size_t q=idx(i,j,k,Nx,Ny);
    double Bx = bx_cc_from_faces(B,i,j,k,p);
    double By = by_cc_from_faces(B,i,j,k,p);
    double Bz = bz_cc_from_faces(B,i,j,k,p);
    return cons_to_prim(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,Bz);
}

// ------------------------------- MUSCL reconstruction -------------------------------

static inline void recon_x(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, int k, Prim& WL, Prim& WR){
    auto Wm = get_prim_cc(p,U,B,i-1,j,k);
    auto Wc = get_prim_cc(p,U,B,i,  j,k);
    auto Wp = get_prim_cc(p,U,B,i+1,j,k);

    auto lim = [&](double wm,double wc,double wp){ return mc_limiter(wc-wm, wp-wc); };

    double sr = lim(Wm.rho,Wc.rho,Wp.rho);
    double su = lim(Wm.u,  Wc.u,  Wp.u);
    double sv = lim(Wm.v,  Wc.v,  Wp.v);
    double sw = lim(Wm.w,  Wc.w,  Wp.w);
    double sP = lim(Wm.P,  Wc.P,  Wp.P);
    double sBy= lim(Wm.By, Wc.By, Wp.By);
    double sBz= lim(Wm.Bz, Wc.Bz, Wp.Bz);

    WL = {Wc.rho-0.5*sr, Wc.u-0.5*su, Wc.v-0.5*sv, Wc.w-0.5*sw, Wc.P-0.5*sP,
          Wc.Bx, Wc.By-0.5*sBy, Wc.Bz-0.5*sBz};
    WR = {Wc.rho+0.5*sr, Wc.u+0.5*su, Wc.v+0.5*sv, Wc.w+0.5*sw, Wc.P+0.5*sP,
          Wc.Bx, Wc.By+0.5*sBy, Wc.Bz+0.5*sBz};

    WL.rho=std::max(WL.rho,p.floor_rho); WR.rho=std::max(WR.rho,p.floor_rho);
    WL.P  =std::max(WL.P,  p.floor_P);   WR.P  =std::max(WR.P,  p.floor_P);
}

static inline void recon_y(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, int k, Prim& WL, Prim& WR){
    auto Wm = get_prim_cc(p,U,B,i,j-1,k);
    auto Wc = get_prim_cc(p,U,B,i,j,  k);
    auto Wp = get_prim_cc(p,U,B,i,j+1,k);

    auto lim = [&](double wm,double wc,double wp){ return mc_limiter(wc-wm, wp-wc); };

    double sr = lim(Wm.rho,Wc.rho,Wp.rho);
    double su = lim(Wm.u,  Wc.u,  Wp.u);
    double sv = lim(Wm.v,  Wc.v,  Wp.v);
    double sw = lim(Wm.w,  Wc.w,  Wp.w);
    double sP = lim(Wm.P,  Wc.P,  Wp.P);
    double sBx= lim(Wm.Bx, Wc.Bx, Wp.Bx);
    double sBz= lim(Wm.Bz, Wc.Bz, Wp.Bz);

    WL = {Wc.rho-0.5*sr, Wc.u-0.5*su, Wc.v-0.5*sv, Wc.w-0.5*sw, Wc.P-0.5*sP,
          Wc.Bx-0.5*sBx, Wc.By, Wc.Bz-0.5*sBz};
    WR = {Wc.rho+0.5*sr, Wc.u+0.5*su, Wc.v+0.5*sv, Wc.w+0.5*sw, Wc.P+0.5*sP,
          Wc.Bx+0.5*sBx, Wc.By, Wc.Bz+0.5*sBz};

    WL.rho=std::max(WL.rho,p.floor_rho); WR.rho=std::max(WR.rho,p.floor_rho);
    WL.P  =std::max(WL.P,  p.floor_P);   WR.P  =std::max(WR.P,  p.floor_P);
}

static inline void recon_z(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, int k, Prim& WL, Prim& WR){
    auto Wm = get_prim_cc(p,U,B,i,j,k-1);
    auto Wc = get_prim_cc(p,U,B,i,j,k);
    auto Wp = get_prim_cc(p,U,B,i,j,k+1);

    auto lim = [&](double wm,double wc,double wp){ return mc_limiter(wc-wm, wp-wc); };

    double sr = lim(Wm.rho,Wc.rho,Wp.rho);
    double su = lim(Wm.u,  Wc.u,  Wp.u);
    double sv = lim(Wm.v,  Wc.v,  Wp.v);
    double sw = lim(Wm.w,  Wc.w,  Wp.w);
    double sP = lim(Wm.P,  Wc.P,  Wp.P);
    double sBx= lim(Wm.Bx, Wc.Bx, Wp.Bx);
    double sBy= lim(Wm.By, Wc.By, Wp.By);

    WL = {Wc.rho-0.5*sr, Wc.u-0.5*su, Wc.v-0.5*sv, Wc.w-0.5*sw, Wc.P-0.5*sP,
          Wc.Bx-0.5*sBx, Wc.By-0.5*sBy, Wc.Bz};
    WR = {Wc.rho+0.5*sr, Wc.u+0.5*su, Wc.v+0.5*sv, Wc.w+0.5*sw, Wc.P+0.5*sP,
          Wc.Bx+0.5*sBx, Wc.By+0.5*sBy, Wc.Bz};

    WL.rho=std::max(WL.rho,p.floor_rho); WR.rho=std::max(WR.rho,p.floor_rho);
    WL.P  =std::max(WL.P,  p.floor_P);   WR.P  =std::max(WR.P,  p.floor_P);
}

// ------------------------------- HLLD (x-normal) -------------------------------

static inline double total_pressure(const Prim& W){
    double B2 = W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz;
    return W.P + 0.5*B2;
}

struct FluxX {
    double frho, fmx, fmy, fmz, fE;
    double fBy, fBz;
};

static inline FluxX flux_ideal_x(const Params& p, const Prim& W){
    double rho=W.rho, u=W.u, v=W.v, w=W.w, P=W.P;
    double Bx=W.Bx, By=W.By, Bz=W.Bz;
    double v2=u*u+v*v+w*w;
    double B2=Bx*Bx+By*By+Bz*Bz;
    double pt=P+0.5*B2;
    double E=P/(p.gamma-1.0)+0.5*rho*v2+0.5*B2;
    double vdotB = u*Bx+v*By+w*Bz;

    FluxX F{};
    F.frho = rho*u;
    F.fmx  = rho*u*u + pt - Bx*Bx;
    F.fmy  = rho*u*v - Bx*By;
    F.fmz  = rho*u*w - Bx*Bz;
    F.fE   = (E+pt)*u - Bx*vdotB;
    F.fBy  = u*By - v*Bx;
    F.fBz  = u*Bz - w*Bx;
    return F;
}

static inline FluxX hlld_x(const Params& p, Prim WL, Prim WR, double Bn){
    WL.Bx = Bn; WR.Bx = Bn;

    double rhoL=WL.rho, rhoR=WR.rho;
    double uL=WL.u, uR=WR.u;
    double vL=WL.v, vR=WR.v;
    double wL=WL.w, wR=WR.w;
    double ByL=WL.By, ByR=WR.By;
    double BzL=WL.Bz, BzR=WR.Bz;

    auto FL = flux_ideal_x(p, WL);
    auto FR = flux_ideal_x(p, WR);

    double cfL = fast_speed_x(p, WL);
    double cfR = fast_speed_x(p, WR);
    double SL = std::min(uL - cfL, uR - cfR);
    double SR = std::max(uL + cfL, uR + cfR);

    if (0.0 <= SL) return FL;
    if (0.0 >= SR) return FR;

    auto cons7 = [&](const Prim& W){
        double rho=W.rho;
        double mx=rho*W.u, my=rho*W.v, mz=rho*W.w;
        double B2=W.Bx*W.Bx+W.By*W.By+W.Bz*W.Bz;
        double E=W.P/(p.gamma-1.0)+0.5*rho*(W.u*W.u+W.v*W.v+W.w*W.w)+0.5*B2;
        return std::array<double,7>{rho,mx,my,mz,E,W.By,W.Bz};
    };
    auto UL = cons7(WL);
    auto UR = cons7(WR);

    double ptL = total_pressure(WL);
    double ptR = total_pressure(WR);

    double denom = rhoL*(SL-uL) - rhoR*(SR-uR);
    double SM = 0.0;
    if (std::abs(denom) > 1e-14){
        SM = (ptR - ptL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / denom;
    }

    double rhoL_s = rhoL*(SL-uL)/(SL-SM);
    double rhoR_s = rhoR*(SR-uR)/(SR-SM);

    double pt_s = ptL + rhoL*(SL-uL)*(SM-uL);

    double sL = SL-uL;
    double sR = SR-uR;
    double dL = rhoL*sL*(SL-SM) - Bn*Bn;
    double dR = rhoR*sR*(SR-SM) - Bn*Bn;

    double ByL_s=ByL, BzL_s=BzL;
    double ByR_s=ByR, BzR_s=BzR;
    if (std::abs(dL) > 1e-14){
        ByL_s = ByL * (rhoL*sL*(SL-uL) - Bn*Bn) / dL;
        BzL_s = BzL * (rhoL*sL*(SL-uL) - Bn*Bn) / dL;
    }
    if (std::abs(dR) > 1e-14){
        ByR_s = ByR * (rhoR*sR*(SR-uR) - Bn*Bn) / dR;
        BzR_s = BzR * (rhoR*sR*(SR-uR) - Bn*Bn) / dR;
    }

    double vL_s=vL, wL_s=wL;
    double vR_s=vR, wR_s=wR;
    if (std::abs(dL) > 1e-14){
        vL_s = vL - Bn*(ByL_s-ByL)/(rhoL*(SL-uL)*(SL-SM));
        wL_s = wL - Bn*(BzL_s-BzL)/(rhoL*(SL-uL)*(SL-SM));
    }
    if (std::abs(dR) > 1e-14){
        vR_s = vR - Bn*(ByR_s-ByR)/(rhoR*(SR-uR)*(SR-SM));
        wR_s = wR - Bn*(BzR_s-BzR)/(rhoR*(SR-uR)*(SR-SM));
    }

    double sqrt_rhoL = std::sqrt(std::max(rhoL_s,1e-300));
    double sqrt_rhoR = std::sqrt(std::max(rhoR_s,1e-300));
    double SLL = SM - std::abs(Bn)/sqrt_rhoL;
    double SRR = SM + std::abs(Bn)/sqrt_rhoR;

    if (std::abs(Bn) < 1e-12){
        double inv = 1.0/(SR-SL);
        auto FUL = std::array<double,7>{FL.frho,FL.fmx,FL.fmy,FL.fmz,FL.fE,FL.fBy,FL.fBz};
        auto FUR = std::array<double,7>{FR.frho,FR.fmx,FR.fmy,FR.fmz,FR.fE,FR.fBy,FR.fBz};

        FluxX FH{};
        for(int k=0;k<7;k++){
            double f = (SR*FUL[k] - SL*FUR[k] + SL*SR*(UR[k]-UL[k]))*inv;
            if(k==0) FH.frho=f;
            if(k==1) FH.fmx=f;
            if(k==2) FH.fmy=f;
            if(k==3) FH.fmz=f;
            if(k==4) FH.fE=f;
            if(k==5) FH.fBy=f;
            if(k==6) FH.fBz=f;
        }
        return FH;
    }

    double signBn = (Bn >= 0.0) ? 1.0 : -1.0;

    double v_ss, w_ss, By_ss, Bz_ss;
    {
        double denom2 = sqrt_rhoL + sqrt_rhoR;
        v_ss  = (sqrt_rhoL*vL_s + sqrt_rhoR*vR_s + (ByR_s-ByL_s)*signBn) / denom2;
        w_ss  = (sqrt_rhoL*wL_s + sqrt_rhoR*wR_s + (BzR_s-BzL_s)*signBn) / denom2;
        By_ss = (sqrt_rhoL*ByR_s + sqrt_rhoR*ByL_s + sqrt_rhoL*sqrt_rhoR*(vR_s-vL_s)*signBn) / denom2;
        Bz_ss = (sqrt_rhoL*BzR_s + sqrt_rhoR*BzL_s + sqrt_rhoL*sqrt_rhoR*(wR_s-wL_s)*signBn) / denom2;
    }

    auto state7 = [&](const Prim& W){
        double B2=W.Bx*W.Bx+W.By*W.By+W.Bz*W.Bz;
        double E=W.P/(p.gamma-1.0)+0.5*W.rho*(W.u*W.u+W.v*W.v+W.w*W.w)+0.5*B2;
        return std::array<double,7>{W.rho, W.rho*W.u, W.rho*W.v, W.rho*W.w, E, W.By, W.Bz};
    };

    auto U_star = [&](const Prim& Wref, double S, double rho_s, double v_s, double w_s, double By_s, double Bz_s){
        double rho = rho_s;
        double mx  = rho_s*SM;
        double my  = rho_s*v_s;
        double mz  = rho_s*w_s;

        double Bx=Bn, By=By_s, Bz=Bz_s;

        double B2W = Wref.Bx*Wref.Bx + Wref.By*Wref.By + Wref.Bz*Wref.Bz;
        double EW = Wref.P/(p.gamma-1.0) + 0.5*Wref.rho*(Wref.u*Wref.u+Wref.v*Wref.v+Wref.w*Wref.w) + 0.5*B2W;

        double vdotB_s = SM*Bx + v_s*By + w_s*Bz;
        double vdotB_w = Wref.u*Wref.Bx + Wref.v*Wref.By + Wref.w*Wref.Bz;

        double E = ( (S - Wref.u)*EW
                     - (total_pressure(Wref)*Wref.u - Bn*vdotB_w)
                     + (pt_s*SM - Bn*vdotB_s) ) / (S - SM);

        return std::array<double,7>{rho,mx,my,mz,E,By,Bz};
    };

    auto flux_from_state = [&](const std::array<double,7>& Ust, const Prim& Wref, const FluxX& Fref, double S){
        auto U0 = state7(Wref);
        FluxX F{};
        F.frho = Fref.frho + S*(Ust[0]-U0[0]);
        F.fmx  = Fref.fmx  + S*(Ust[1]-U0[1]);
        F.fmy  = Fref.fmy  + S*(Ust[2]-U0[2]);
        F.fmz  = Fref.fmz  + S*(Ust[3]-U0[3]);
        F.fE   = Fref.fE   + S*(Ust[4]-U0[4]);
        F.fBy  = Fref.fBy  + S*(Ust[5]-U0[5]);
        F.fBz  = Fref.fBz  + S*(Ust[6]-U0[6]);
        return F;
    };

    auto ULs  = U_star(WL, SL, rhoL_s, vL_s, wL_s, ByL_s, BzL_s);
    auto URs  = U_star(WR, SR, rhoR_s, vR_s, wR_s, ByR_s, BzR_s);
    auto ULss = U_star(WL, SL, rhoL_s, v_ss, w_ss, By_ss, Bz_ss);
    auto URss = U_star(WR, SR, rhoR_s, v_ss, w_ss, By_ss, Bz_ss);

    if (0.0 < SLL) return flux_from_state(ULs,  WL, FL, SL);
    if (0.0 < SM ) return flux_from_state(ULss, WL, FL, SL);
    if (0.0 < SRR) return flux_from_state(URss, WR, FR, SR);
    return flux_from_state(URs,  WR, FR, SR);
}

// ------------------------------- HLLD (y-normal via mapping) -------------------------------

struct FluxY {
    double frho, fmx, fmy, fmz, fE;
    double fBx, fBz;
};

static inline FluxY hlld_y(const Params& p, Prim WL, Prim WR, double Bn){
    WL.By = Bn;
    WR.By = Bn;

    Prim aL = {WL.rho, WL.v, WL.u, WL.w, WL.P, WL.By, WL.Bx, WL.Bz};
    Prim aR = {WR.rho, WR.v, WR.u, WR.w, WR.P, WR.By, WR.Bx, WR.Bz};

    FluxX Fx = hlld_x(p, aL, aR, Bn);

    FluxY Gy{};
    Gy.frho = Fx.frho;
    Gy.fmx  = Fx.fmy;
    Gy.fmy  = Fx.fmx;
    Gy.fmz  = Fx.fmz;
    Gy.fE   = Fx.fE;
    Gy.fBx  = Fx.fBy;
    Gy.fBz  = Fx.fBz;
    return Gy;
}

// ------------------------------- HLLD (z-normal via mapping) -------------------------------

struct FluxZ {
    double frho, fmx, fmy, fmz, fE;
    double fBx, fBy;
};

static inline FluxZ hlld_z(const Params& p, Prim WL, Prim WR, double Bn){
    WL.Bz = Bn;
    WR.Bz = Bn;

    Prim aL = {WL.rho, WL.w, WL.u, WL.v, WL.P, WL.Bz, WL.Bx, WL.By};
    Prim aR = {WR.rho, WR.w, WR.u, WR.v, WR.P, WR.Bz, WR.Bx, WR.By};

    FluxX Fx = hlld_x(p, aL, aR, Bn);

    FluxZ Hz{};
    Hz.frho = Fx.frho;
    Hz.fmx  = Fx.fmy;
    Hz.fmy  = Fx.fmz;
    Hz.fmz  = Fx.fmx;
    Hz.fE   = Fx.fE;
    Hz.fBx  = Fx.fBy;
    Hz.fBy  = Fx.fBz;
    return Hz;
}

// ------------------------------- Time step -------------------------------

static inline double compute_dt(const Params& p, const Ucc& U, const Bfaces& B, double dx, double dy, double dz){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    double maxs=1e-14;

    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                Prim W = get_prim_cc(p,U,B,i,j,k);
                maxs = std::max(maxs, std::abs(W.u)+fast_speed_x(p,W));
                maxs = std::max(maxs, std::abs(W.v)+fast_speed_y(p,W));
                maxs = std::max(maxs, std::abs(W.w)+fast_speed_z(p,W));
            }
        }
    }
    double h = std::min({dx,dy,dz});
    double dt_h = p.cfl * h / maxs;

    double dt_r = 1e100;
    if (p.eta > 0.0){
        dt_r = 0.25*h*h/p.eta;
    }
    return std::min(dt_h, dt_r);
}

// ------------------------------- CT + sources -------------------------------

static inline void add_joule_heating(const Params& p, Ucc& U, const Bfaces& B,
                                     double dx, double dy, double dz, double dt){
    if (p.eta <= 0.0) return;
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;

    auto atB = [&](int i,int j,int k){
        double Bx = bx_cc_from_faces(B,i,j,k,p);
        double By = by_cc_from_faces(B,i,j,k,p);
        double Bz = bz_cc_from_faces(B,i,j,k,p);
        return std::array<double,3>{Bx,By,Bz};
    };

    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t q=idx(i,j,k,Nx,Ny);

                auto Bxp = atB(i+1,j,k);
                auto Bxm = atB(i-1,j,k);
                auto Byp = atB(i,j+1,k);
                auto Bym = atB(i,j-1,k);
                auto Bzp = atB(i,j,k+1);
                auto Bzm = atB(i,j,k-1);

                double dBzdy = (Byp[2]-Bym[2])/(2*dy);
                double dBydz = (Bzp[1]-Bzm[1])/(2*dz);
                double dBxdz = (Bzp[0]-Bzm[0])/(2*dz);
                double dBzdx = (Bxp[2]-Bxm[2])/(2*dx);
                double dBydx = (Bxp[1]-Bxm[1])/(2*dx);
                double dBxdy = (Byp[0]-Bym[0])/(2*dy);

                double Jx = dBzdy - dBydz;
                double Jy = dBxdz - dBzdx;
                double Jz = dBydx - dBxdy;

                double J2 = Jx*Jx + Jy*Jy + Jz*Jz;
                U.E[q] += dt * p.eta * J2;
            }
        }
    }
}

static inline void advance_stage(const Params& p, const Ucc& U0, const Bfaces& B0,
                                 Ucc& U1, Bfaces& B1,
                                 double dx, double dy, double dz, double dt){
    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    size_t N=(size_t)Nx*(size_t)Ny*(size_t)Nz;

    U1 = U0;
    B1 = B0;

    std::vector<std::array<double,5>> Fx((size_t)(Nx+1)*(size_t)Ny*(size_t)Nz);
    std::vector<std::array<double,5>> Gy((size_t)Nx*(size_t)(Ny+1)*(size_t)Nz);
    std::vector<std::array<double,5>> Hz((size_t)Nx*(size_t)Ny*(size_t)(Nz+1));

    std::vector<double> Ey_x((size_t)(Nx+1)*(size_t)Ny*(size_t)Nz, 0.0);
    std::vector<double> Ez_x((size_t)(Nx+1)*(size_t)Ny*(size_t)Nz, 0.0);
    std::vector<double> Ex_y((size_t)Nx*(size_t)(Ny+1)*(size_t)Nz, 0.0);
    std::vector<double> Ez_y((size_t)Nx*(size_t)(Ny+1)*(size_t)Nz, 0.0);
    std::vector<double> Ex_z((size_t)Nx*(size_t)Ny*(size_t)(Nz+1), 0.0);
    std::vector<double> Ey_z((size_t)Nx*(size_t)Ny*(size_t)(Nz+1), 0.0);

    auto fidx_x = [&](int i,int j,int k){
        return (static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
               static_cast<size_t>(i);
    };
    auto fidx_y = [&](int i,int j,int k){
        return (static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
               static_cast<size_t>(i);
    };
    auto fidx_z = [&](int i,int j,int k){
        return (static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
               static_cast<size_t>(i);
    };

    // x-faces
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<=Nx;i++){
                int iL=i-1, iR=i;
                if (p.periodic_x){ iL=imod(iL,Nx); iR=imod(iR,Nx); }
                else { iL=std::clamp(iL,0,Nx-1); iR=std::clamp(iR,0,Nx-1); }

                Prim WLL,WLR,WRL,WRR;
                recon_x(p,U0,B0,iL,j,k,WLL,WLR);
                recon_x(p,U0,B0,iR,j,k,WRL,WRR);

                Prim WL = WLR;
                Prim WR = WRL;

                double Bn = get_Bx_face(p,B0,i,j,k);
                FluxX F = hlld_x(p, WL, WR, Bn);

                Fx[fidx_x(i,j,k)] = {F.frho,F.fmx,F.fmy,F.fmz,F.fE};
                Ey_x[fidx_x(i,j,k)] = F.fBz;
                Ez_x[fidx_x(i,j,k)] = -F.fBy;
            }
        }
    }

    // y-faces
    for(int k=0;k<Nz;k++){
        for(int j=0;j<=Ny;j++){
            for(int i=0;i<Nx;i++){
                int jD=j-1, jU=j;
                if (p.periodic_y){ jD=imod(jD,Ny); jU=imod(jU,Ny); }
                else { jD=std::clamp(jD,0,Ny-1); jU=std::clamp(jU,0,Ny-1); }

                Prim WDD,WDU,WUD,WUU;
                recon_y(p,U0,B0,i,jD,k,WDD,WDU);
                recon_y(p,U0,B0,i,jU,k,WUD,WUU);

                Prim WL = WDU;
                Prim WR = WUD;

                double Bn = get_By_face(p,B0,i,j,k);
                FluxY G = hlld_y(p, WL, WR, Bn);

                Gy[fidx_y(i,j,k)] = {G.frho,G.fmx,G.fmy,G.fmz,G.fE};
                Ez_y[fidx_y(i,j,k)] = G.fBx;
                Ex_y[fidx_y(i,j,k)] = -G.fBz;
            }
        }
    }

    // z-faces
    for(int k=0;k<=Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                int kB=k-1, kF=k;
                if (p.periodic_z){ kB=imod(kB,Nz); kF=imod(kF,Nz); }
                else { kB=std::clamp(kB,0,Nz-1); kF=std::clamp(kF,0,Nz-1); }

                Prim WBB,WBF,WFB,WFF;
                recon_z(p,U0,B0,i,j,kB,WBB,WBF);
                recon_z(p,U0,B0,i,j,kF,WFB,WFF);

                Prim WL = WBF;
                Prim WR = WFB;

                double Bn = get_Bz_face(p,B0,i,j,k);
                FluxZ H = hlld_z(p, WL, WR, Bn);

                Hz[fidx_z(i,j,k)] = {H.frho,H.fmx,H.fmy,H.fmz,H.fE};
                Ex_z[fidx_z(i,j,k)] = H.fBy;
                Ey_z[fidx_z(i,j,k)] = -H.fBx;
            }
        }
    }

    // FV update
    double invdx=1.0/dx, invdy=1.0/dy, invdz=1.0/dz;
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t q=idx(i,j,k,Nx,Ny);

                auto& FxR = Fx[fidx_x(i+1,j,k)];
                auto& FxL = Fx[fidx_x(i,  j,k)];
                auto& GyU = Gy[fidx_y(i,j+1,k)];
                auto& GyD = Gy[fidx_y(i,j,  k)];
                auto& HzF = Hz[fidx_z(i,j,k+1)];
                auto& HzB = Hz[fidx_z(i,j,k)];

                double drho = -(FxR[0]-FxL[0])*invdx - (GyU[0]-GyD[0])*invdy - (HzF[0]-HzB[0])*invdz;
                double dmx  = -(FxR[1]-FxL[1])*invdx - (GyU[1]-GyD[1])*invdy - (HzF[1]-HzB[1])*invdz;
                double dmy  = -(FxR[2]-FxL[2])*invdx - (GyU[2]-GyD[2])*invdy - (HzF[2]-HzB[2])*invdz;
                double dmz  = -(FxR[3]-FxL[3])*invdx - (GyU[3]-GyD[3])*invdy - (HzF[3]-HzB[3])*invdz;
                double dE   = -(FxR[4]-FxL[4])*invdx - (GyU[4]-GyD[4])*invdy - (HzF[4]-HzB[4])*invdz;

                U1.rho[q] += dt*drho;
                U1.mx[q]  += dt*dmx;
                U1.my[q]  += dt*dmy;
                U1.mz[q]  += dt*dmz;
                U1.E[q]   += dt*dE;

                double Bx = bx_cc_from_faces(B0,i,j,k,p);
                double By = by_cc_from_faces(B0,i,j,k,p);
                double Bz = bz_cc_from_faces(B0,i,j,k,p);
                enforce_floors(p, U1.rho[q],U1.mx[q],U1.my[q],U1.mz[q],U1.E[q],Bx,By,Bz);
            }
        }
    }

    // Edge-centered E (ideal + resistive)
    std::vector<double> Ex_edge((size_t)Nx*(size_t)(Ny+1)*(size_t)(Nz+1), 0.0);
    std::vector<double> Ey_edge((size_t)(Nx+1)*(size_t)Ny*(size_t)(Nz+1), 0.0);
    std::vector<double> Ez_edge((size_t)(Nx+1)*(size_t)(Ny+1)*(size_t)Nz, 0.0);

    auto exy = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny+1); else j=std::clamp(j,0,Ny);
        if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
        return Ex_y[fidx_y(i,j,k)];
    };
    auto ezx = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx+1); else i=std::clamp(i,0,Nx);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
        return Ez_x[fidx_x(i,j,k)];
    };
    auto eyx = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx+1); else i=std::clamp(i,0,Nx);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
        return Ey_x[fidx_x(i,j,k)];
    };
    auto exz = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        if (p.periodic_z) k=imod(k,Nz+1); else k=std::clamp(k,0,Nz);
        return Ex_z[fidx_z(i,j,k)];
    };
    auto eyz = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        if (p.periodic_z) k=imod(k,Nz+1); else k=std::clamp(k,0,Nz);
        return Ey_z[fidx_z(i,j,k)];
    };
    auto ezy = [&](int i,int j,int k){
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny+1); else j=std::clamp(j,0,Ny);
        if (p.periodic_z) k=imod(k,Nz); else k=std::clamp(k,0,Nz-1);
        return Ez_y[fidx_y(i,j,k)];
    };

    // Ex edges (x-direction)
    for(int kf=0;kf<=Nz;kf++){
        for(int jf=0;jf<=Ny;jf++){
            for(int i=0;i<Nx;i++){
                double ex_ideal = 0.25*(exy(i,jf,kf-1) + exy(i,jf,kf) + exz(i,jf-1,kf) + exz(i,jf,kf));
                double ex_res = 0.0;
                if (p.eta > 0.0){
                    auto Bzp = bz_cc_from_faces(B0,i,jf,kf,p);
                    auto Bzm = bz_cc_from_faces(B0,i,jf-1,kf,p);
                    auto Byp = by_cc_from_faces(B0,i,jf,kf,p);
                    auto Bym = by_cc_from_faces(B0,i,jf,kf-1,p);
                    double dBzdy = (Bzp - Bzm)/dy;
                    double dBydz = (Byp - Bym)/dz;
                    ex_res = p.eta * (dBzdy - dBydz);
                }
                Ex_edge[(static_cast<size_t>(kf)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf))*static_cast<size_t>(Nx) +
                        static_cast<size_t>(i)] = ex_ideal + ex_res;
            }
        }
    }

    // Ey edges (y-direction)
    for(int kf=0;kf<=Nz;kf++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<=Nx;i++){
                double ey_ideal = 0.25*(eyx(i,j,kf-1) + eyx(i,j,kf) + eyz(i-1,j,kf) + eyz(i,j,kf));
                double ey_res = 0.0;
                if (p.eta > 0.0){
                    auto Bxp = bx_cc_from_faces(B0,i,j,kf,p);
                    auto Bxm = bx_cc_from_faces(B0,i,j,kf-1,p);
                    auto Bzp = bz_cc_from_faces(B0,i,j,kf,p);
                    auto Bzm = bz_cc_from_faces(B0,i-1,j,kf,p);
                    double dBxdz = (Bxp - Bxm)/dz;
                    double dBzdx = (Bzp - Bzm)/dx;
                    ey_res = p.eta * (dBxdz - dBzdx);
                }
                Ey_edge[(static_cast<size_t>(kf)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                        static_cast<size_t>(i)] = ey_ideal + ey_res;
            }
        }
    }

    // Ez edges (z-direction)
    for(int k=0;k<Nz;k++){
        for(int jf=0;jf<=Ny;jf++){
            for(int i=0;i<=Nx;i++){
                double ez_ideal = 0.25*(ezx(i,jf-1,k) + ezx(i,jf,k) + ezy(i-1,jf,k) + ezy(i,jf,k));
                double ez_res = 0.0;
                if (p.eta > 0.0){
                    auto Byp = by_cc_from_faces(B0,i,jf,k,p);
                    auto Bym = by_cc_from_faces(B0,i-1,jf,k,p);
                    auto Bxp = bx_cc_from_faces(B0,i,jf,k,p);
                    auto Bxm = bx_cc_from_faces(B0,i,jf-1,k,p);
                    double dBydx = (Byp - Bym)/dx;
                    double dBxdy = (Bxp - Bxm)/dy;
                    ez_res = p.eta * (dBydx - dBxdy);
                }
                Ez_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf))*static_cast<size_t>(Nx+1) +
                        static_cast<size_t>(i)] = ez_ideal + ez_res;
            }
        }
    }

    // CT update faces
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<=Nx;i++){
                size_t f=fidx_x(i,j,k);
                int jf0=j, jf1=j+1;
                int kf0=k, kf1=k+1;
                if (p.periodic_y){ jf0=imod(jf0,Ny+1); jf1=imod(jf1,Ny+1); }
                else { jf0=std::clamp(jf0,0,Ny); jf1=std::clamp(jf1,0,Ny); }
                if (p.periodic_z){ kf0=imod(kf0,Nz+1); kf1=imod(kf1,Nz+1); }
                else { kf0=std::clamp(kf0,0,Nz); kf1=std::clamp(kf1,0,Nz); }

                double ezU = Ez_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf1))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(i)];
                double ezD = Ez_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf0))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(i)];
                double eyF = Ey_edge[(static_cast<size_t>(kf1)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(i)];
                double eyB = Ey_edge[(static_cast<size_t>(kf0)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(i)];

                B1.Bx_f[f] += -(dt/dy)*(ezU - ezD) + (dt/dz)*(eyF - eyB);
            }
        }
    }

    for(int k=0;k<Nz;k++){
        for(int j=0;j<=Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t f=fidx_y(i,j,k);
                int if0=i, if1=i+1;
                int kf0=k, kf1=k+1;
                if (p.periodic_x){ if0=imod(if0,Nx+1); if1=imod(if1,Nx+1); }
                else { if0=std::clamp(if0,0,Nx); if1=std::clamp(if1,0,Nx); }
                if (p.periodic_z){ kf0=imod(kf0,Nz+1); kf1=imod(kf1,Nz+1); }
                else { kf0=std::clamp(kf0,0,Nz); kf1=std::clamp(kf1,0,Nz); }

                double ezR = Ez_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(if1)];
                double ezL = Ez_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(if0)];
                double exF = Ex_edge[(static_cast<size_t>(kf1)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
                                     static_cast<size_t>(i)];
                double exB = Ex_edge[(static_cast<size_t>(kf0)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
                                     static_cast<size_t>(i)];

                B1.By_f[f] += +(dt/dx)*(ezR - ezL) - (dt/dz)*(exF - exB);
            }
        }
    }

    for(int k=0;k<=Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t f=fidx_z(i,j,k);
                int if0=i, if1=i+1;
                int jf0=j, jf1=j+1;
                if (p.periodic_x){ if0=imod(if0,Nx+1); if1=imod(if1,Nx+1); }
                else { if0=std::clamp(if0,0,Nx); if1=std::clamp(if1,0,Nx); }
                if (p.periodic_y){ jf0=imod(jf0,Ny+1); jf1=imod(jf1,Ny+1); }
                else { jf0=std::clamp(jf0,0,Ny); jf1=std::clamp(jf1,0,Ny); }

                double eyR = Ey_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(if1)];
                double eyL = Ey_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                                     static_cast<size_t>(if0)];
                double exU = Ex_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf1))*static_cast<size_t>(Nx) +
                                     static_cast<size_t>(i)];
                double exD = Ex_edge[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jf0))*static_cast<size_t>(Nx) +
                                     static_cast<size_t>(i)];

                B1.Bz_f[f] += -(dt/dx)*(eyR - eyL) + (dt/dy)*(exU - exD);
            }
        }
    }

    add_joule_heating(p, U1, B1, dx, dy, dz, dt);

    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t q=idx(i,j,k,Nx,Ny);
                double Bx = bx_cc_from_faces(B1,i,j,k,p);
                double By = by_cc_from_faces(B1,i,j,k,p);
                double Bz = bz_cc_from_faces(B1,i,j,k,p);
                enforce_floors(p, U1.rho[q],U1.mx[q],U1.my[q],U1.mz[q],U1.E[q],Bx,By,Bz);
            }
        }
    }
}

// ------------------------------- RK2 -------------------------------

static inline void rk2_step(const Params& p, Ucc& U, Bfaces& B,
                            double dx, double dy, double dz, double dt){
    Ucc U1; Bfaces B1;
    advance_stage(p, U, B, U1, B1, dx, dy, dz, dt);

    Ucc U2; Bfaces B2;
    advance_stage(p, U1, B1, U2, B2, dx, dy, dz, dt);

    size_t N=U.rho.size();
    for(size_t q=0;q<N;q++){
        U.rho[q] = 0.5*(U.rho[q] + U2.rho[q]);
        U.mx[q]  = 0.5*(U.mx[q]  + U2.mx[q]);
        U.my[q]  = 0.5*(U.my[q]  + U2.my[q]);
        U.mz[q]  = 0.5*(U.mz[q]  + U2.mz[q]);
        U.E[q]   = 0.5*(U.E[q]   + U2.E[q]);
    }
    for(size_t k=0;k<B.Bx_f.size();k++) B.Bx_f[k] = 0.5*(B.Bx_f[k] + B2.Bx_f[k]);
    for(size_t k=0;k<B.By_f.size();k++) B.By_f[k] = 0.5*(B.By_f[k] + B2.By_f[k]);
    for(size_t k=0;k<B.Bz_f.size();k++) B.Bz_f[k] = 0.5*(B.Bz_f[k] + B2.Bz_f[k]);

    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t q=idx(i,j,k,Nx,Ny);
                double Bx = bx_cc_from_faces(B,i,j,k,p);
                double By = by_cc_from_faces(B,i,j,k,p);
                double Bz = bz_cc_from_faces(B,i,j,k,p);
                enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,Bz);
            }
        }
    }
}

// ------------------------------- Diagnostics / output -------------------------------

static inline double divB_cc(const Params& p, const Bfaces& B, int i, int j, int k, double dx, double dy, double dz){
    double BxR = get_Bx_face(p,B,i+1,j,k);
    double BxL = get_Bx_face(p,B,i,  j,k);
    double ByU = get_By_face(p,B,i,j+1,k);
    double ByD = get_By_face(p,B,i,j,  k);
    double BzF = get_Bz_face(p,B,i,j,k+1);
    double BzB = get_Bz_face(p,B,i,j,k);
    return (BxR-BxL)/dx + (ByU-ByD)/dy + (BzF-BzB)/dz;
}

static void write_h5(const Params& p,
                     const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
                     const std::vector<double>& t,
                     const std::vector<double>& rho, const std::vector<double>& P,
                     const std::vector<double>& Mach,
                     const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w,
                     const std::vector<double>& vabs,
                     const std::vector<double>& Bx, const std::vector<double>& By, const std::vector<double>& Bz,
                     const std::vector<double>& divB,
                     const std::string& fname)
{
    hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    auto write1 = [&](const char* name, const std::vector<double>& a){
        hsize_t dims[1] = { (hsize_t)a.size() };
        hid_t sp = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate(file, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data());
        H5Dclose(ds); H5Sclose(sp);
    };

    auto write4 = [&](const char* name, const std::vector<double>& a){
        hsize_t dims[4] = { (hsize_t)t.size(), (hsize_t)p.Nz, (hsize_t)p.Ny, (hsize_t)p.Nx };
        hid_t sp = H5Screate_simple(4, dims, nullptr);
        hid_t ds = H5Dcreate(file, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data());
        H5Dclose(ds); H5Sclose(sp);
    };

    write1("/x", x);
    write1("/y", y);
    write1("/z", z);
    write1("/t", t);

    write4("/rho", rho);
    write4("/P",   P);
    write4("/Mach",Mach);
    write4("/u",   u);
    write4("/v",   v);
    write4("/w",   w);
    write4("/vabs",vabs);

    write4("/Bx",  Bx);
    write4("/By",  By);
    write4("/Bz",  Bz);
    write4("/divB",divB);

    H5Fclose(file);
}

// ------------------------------- Main -------------------------------

int main(){
    Params p;

    int Nx=p.Nx, Ny=p.Ny, Nz=p.Nz;
    double dx=(p.x1-p.x0)/Nx;
    double dy=(p.y1-p.y0)/Ny;
    double dz=(p.z1-p.z0)/Nz;

    std::vector<double> x(Nx), y(Ny), z(Nz);
    for(int i=0;i<Nx;i++) x[i]=p.x0+(i+0.5)*dx;
    for(int j=0;j<Ny;j++) y[j]=p.y0+(j+0.5)*dy;
    for(int k=0;k<Nz;k++) z[k]=p.z0+(k+0.5)*dz;

    size_t N=(size_t)Nx*(size_t)Ny*(size_t)Nz;

    Ucc U;
    U.rho.assign(N,0.0);
    U.mx.assign(N,0.0);
    U.my.assign(N,0.0);
    U.mz.assign(N,0.0);
    U.E.assign(N,0.0);

    Bfaces B;
    B.Bx_f.assign((size_t)(Nx+1)*(size_t)Ny*(size_t)Nz, 0.0);
    B.By_f.assign((size_t)Nx*(size_t)(Ny+1)*(size_t)Nz, 0.0);
    B.Bz_f.assign((size_t)Nx*(size_t)Ny*(size_t)(Nz+1), 0.0);

    // -------------------------
    // Divergence-free B init via vector potential Az on corners
    // -------------------------
    double cx=0.5*(p.x0+p.x1), cy=0.5*(p.y0+p.y1), cz=0.5*(p.z0+p.z1);

    auto xcorner = [&](int ic){ return p.x0 + ic*dx; };
    auto ycorner = [&](int jc){ return p.y0 + jc*dy; };
    auto zcorner = [&](int kc){ return p.z0 + kc*dz; };

    std::vector<double> Az((size_t)(Nx+1)*(size_t)(Ny+1)*(size_t)(Nz+1), 0.0);
    auto az_idx = [&](int ic,int jc,int kc)->size_t{
        if (p.periodic_x) ic = imod(ic, Nx+1); else ic = std::clamp(ic,0,Nx);
        if (p.periodic_y) jc = imod(jc, Ny+1); else jc = std::clamp(jc,0,Ny);
        if (p.periodic_z) kc = imod(kc, Nz+1); else kc = std::clamp(kc,0,Nz);
        return (static_cast<size_t>(kc)*static_cast<size_t>(Ny+1) + static_cast<size_t>(jc))*static_cast<size_t>(Nx+1) +
               static_cast<size_t>(ic);
    };

    for(int kc=0; kc<=Nz; ++kc){
        for(int jc=0; jc<=Ny; ++jc){
            for(int ic=0; ic<=Nx; ++ic){
                double X = xcorner(ic);
                double Y = ycorner(jc);
                double Z = zcorner(kc);

                double rx = X - cx;
                double ry = Y - cy;
                double rz = Z - cz;
                double r  = std::sqrt(rx*rx + ry*ry + rz*rz) + 1e-14;

                double s = (r - p.r_core)/std::max(p.r_edge, 1e-12);
                double core_shape = 0.5*(1.0 - std::tanh(s));

                double az = p.Bshear * core_shape * std::sin(2.0*M_PI*X) * std::sin(2.0*M_PI*Y);
                Az[az_idx(ic,jc,kc)] = az;
            }
        }
    }

    // Bx = dAz/dy, By = -dAz/dx
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i_face=0;i_face<=Nx;i_face++){
                double AzU = Az[az_idx(i_face, j+1, k)];
                double AzD = Az[az_idx(i_face, j,   k)];
                B.Bx_f[(static_cast<size_t>(k)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx+1) +
                       static_cast<size_t>(i_face)] = (AzU - AzD)/dy + p.B0x;
            }
        }
    }

    for(int k=0;k<Nz;k++){
        for(int j_face=0;j_face<=Ny;j_face++){
            for(int i=0;i<Nx;i++){
                double AzR = Az[az_idx(i+1, j_face, k)];
                double AzL = Az[az_idx(i,   j_face, k)];
                B.By_f[(static_cast<size_t>(k)*static_cast<size_t>(Ny+1) + static_cast<size_t>(j_face))*static_cast<size_t>(Nx) +
                       static_cast<size_t>(i)] = -(AzR - AzL)/dx;
            }
        }
    }

    for(int k_face=0;k_face<=Nz;k_face++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                B.Bz_f[(static_cast<size_t>(k_face)*static_cast<size_t>(Ny) + static_cast<size_t>(j))*static_cast<size_t>(Nx) +
                       static_cast<size_t>(i)] = p.B0z;
            }
        }
    }

    // -------------------------
    // IC for hydro/momentum
    // -------------------------
    for(int k=0;k<Nz;k++){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                double X = x[i];
                double Y = y[j];
                double Z = z[k];

                double rx = X - cx;
                double ry = Y - cy;
                double rz = Z - cz;
                double r  = std::sqrt(rx*rx + ry*ry + rz*rz) + 1e-14;

                double s = (r - p.r_core)/std::max(p.r_edge, 1e-12);
                double core_shape = 0.5*(1.0 - std::tanh(s));

                double rho = p.rho_bg + (p.rho_core - p.rho_bg)*core_shape;
                double P = p.P0 * std::pow(rho/p.rho_bg, p.gamma);

                double u=0.0, v=0.0, w=0.0;
                double vin = p.vin * core_shape;
                u -= vin * rx/r;
                v -= vin * ry/r;
                w -= vin * rz/r;

                if (p.rot_omega != 0.0){
                    u += -p.rot_omega * (Y - cy);
                    v +=  p.rot_omega * (X - cx);
                }

                double Bx = bx_cc_from_faces(B,i,j,k,p);
                double By = by_cc_from_faces(B,i,j,k,p);
                double Bz = bz_cc_from_faces(B,i,j,k,p);

                Prim W{rho,u,v,w,P,Bx,By,Bz};
                size_t q=idx(i,j,k,Nx,Ny);
                prim_to_cons(p, W, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q]);
                enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,Bz);
            }
        }
    }

    // -------------------------
    // Time integration + output
    // -------------------------
    std::vector<double> tstore;
    std::vector<double> rho_store, P_store, Mach_store, u_store, v_store, w_store, vabs_store;
    std::vector<double> Bx_store, By_store, Bz_store, divB_store;

    double t=0.0;
    double tnext=0.0;
    int it=0;

    while (t < p.t_end){
        double dt = compute_dt(p,U,B,dx,dy,dz);
        if (t + dt > p.t_end) dt = p.t_end - t;

        rk2_step(p,U,B,dx,dy,dz,dt);
        t += dt;
        it++;

        if (t >= tnext - 1e-12){
            tstore.push_back(t);

            for(int k=0;k<Nz;k++){
                for(int j=0;j<Ny;j++){
                    for(int i=0;i<Nx;i++){
                        size_t q=idx(i,j,k,Nx,Ny);
                        double Bx = bx_cc_from_faces(B,i,j,k,p);
                        double By = by_cc_from_faces(B,i,j,k,p);
                        double Bz = bz_cc_from_faces(B,i,j,k,p);
                        Prim W = cons_to_prim(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,Bz);

                        double cs = std::sqrt(sound_speed2(p,W));
                        double vabs = std::sqrt(W.u*W.u + W.v*W.v + W.w*W.w);
                        double Mach = vabs / std::max(cs,1e-12);
                        double divB = divB_cc(p,B,i,j,k,dx,dy,dz);

                        rho_store.push_back(W.rho);
                        P_store.push_back(W.P);
                        Mach_store.push_back(Mach);
                        u_store.push_back(W.u);
                        v_store.push_back(W.v);
                        w_store.push_back(W.w);
                        vabs_store.push_back(vabs);
                        Bx_store.push_back(W.Bx);
                        By_store.push_back(W.By);
                        Bz_store.push_back(W.Bz);
                        divB_store.push_back(divB);
                    }
                }
            }

            std::cout << "Stored frame " << tstore.size()-1 << " at t=" << t << "\n";
            tnext += p.dt_store;
        }
    }

    write_h5(p,x,y,z,tstore,
             rho_store,P_store,Mach_store,
             u_store,v_store,w_store,vabs_store,
             Bx_store,By_store,Bz_store,divB_store,
             "out3d_mhd.h5");

    std::cout<<"Wrote out3d_mhd.h5\n";
    return 0;
}
