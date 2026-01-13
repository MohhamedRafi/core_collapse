// mhd2d_resistive_hlld_ct_gravity.cpp
//
// 2D Resistive MHD + Self-Gravity (periodic Poisson) with:
//   - FV Godunov update for cell-centered conserved vars using HLLD
//   - CT (constrained transport) for in-plane B (Bx, By) using edge-centered Ez
//   - Resistivity via -curl(eta curl B):
//       * CT uses Ez_total = Ez_ideal + Ez_res, with Ez_res = eta * Jz at edges
//       * For constant eta, Bz resistive term reduces to eta * Laplacian(Bz) in 2D
//       * Joule heating: dE/dt += eta |J|^2
//   - Gravity via periodic Poisson solve (SOR), coupled as sources:
//       d(m)/dt = rho g,  dE/dt = rho v·g
//
// Core-collapse-style IC (toy):
//   - Central overdense core with soft edge
//   - Sub-virial pressure + inward radial velocity + optional rotation
//   - Divergence-free face fields initialized from a corner-centered vector potential Az
//
// Output: out2d_mhd.h5
//   /x (Nx), /y (Ny), /t (Nt)
//   /rho, /P, /Mach, /u, /v, /vabs, /Bx, /By, /Bz, /divB    all (Nt,Ny,Nx)
//
// Build:
//   clang++ -O3 -std=c++17 -Wall -Wextra -pedantic mhd2d_resistive_hlld_ct_gravity.cpp -lhdf5 -o mhd2d_resistive_hlld_ct_gravity
//
// Run:
//   ./mhd2d_resistive_hlld_ct_gravity

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

static inline size_t idx(int i, int j, int Nx) { return (size_t)j*(size_t)Nx + (size_t)i; }
static inline int imod(int i, int n){ int r=i%n; return (r<0)?r+n:r; }

struct Params {
    double gamma = 5.0/3.0;
    double cfl   = 0.45;

    double x0=0.0, x1=1.0;
    double y0=0.0, y1=1.0;
    int Nx=192;
    int Ny=192;

    double t_end    = 0.35;
    double dt_store = 1e-3;

    double floor_rho = 1e-10;
    double floor_P   = 1e-10;

    bool periodic_x = true;
    bool periodic_y = true;

    // resistivity (constant eta)
    double eta = 5e-4;

    // gravity
    bool gravity_on = true;
    double G = 8.0;
    double softening = 1e-4;

    // Poisson SOR
    int sor_iters = 600;
    double sor_omega = 1.8;

    // IC knobs
    double rho_bg   = 1.0;
    double rho_core = 20.0;
    double r_core   = 0.12;
    double r_edge   = 0.02;      // edge thickness
    double P0       = 0.15;      // base pressure scale
    double vin      = 1.25;      // inward speed scale (core region)
    double rot_omega = 1.0;      // rotation strength (set 0 for none)

    // magnetic seed (set via Az)
    double B0x   = 0.15;  // uniform Bx target
    double Bshear= 0.05;  // strength of div-free perturbation via Az
};

struct Ucc { // cell-centered conserved
    std::vector<double> rho, mx, my, mz, E, Bz;
};

struct Bfaces { // face-centered in-plane magnetic field
    std::vector<double> Bx_f; // (Nx+1)*Ny at (i_face, j)
    std::vector<double> By_f; // Nx*(Ny+1) at (i, j_face)
};

struct GravField {
    std::vector<double> gx, gy;
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

static inline void enforce_floors(const Params& p, double& rho, double& mx, double& my, double& mz, double& E, double Bx, double By, double Bz){
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
                                double& rho, double& mx, double& my, double& mz, double& E, double& Bz_out){
    rho = std::max(W.rho, p.floor_rho);
    mx = rho*W.u; my = rho*W.v; mz = rho*W.w;
    double kin = 0.5*rho*(W.u*W.u+W.v*W.v+W.w*W.w);
    double mag = 0.5*(W.Bx*W.Bx+W.By*W.By+W.Bz*W.Bz);
    double eint = W.P/(p.gamma-1.0);
    E = kin + mag + eint;
    Bz_out = W.Bz;
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

// ------------------------------- Gravity (periodic Poisson, SOR) -------------------------------

static inline GravField compute_poisson_gravity_periodic(const Params& p, const std::vector<double>& rho_cc,
                                                         double dx, double dy){
    GravField g;
    int Nx=p.Nx, Ny=p.Ny;
    size_t N=(size_t)Nx*(size_t)Ny;
    g.gx.assign(N,0.0);
    g.gy.assign(N,0.0);
    if (!p.gravity_on) return g;

    double rhobar=0.0;
    for(double v: rho_cc) rhobar += v;
    rhobar /= (double)rho_cc.size();

    std::vector<double> rhs(N,0.0);
    for(size_t q=0;q<N;q++){
        rhs[q] = 4.0*M_PI*p.G*(rho_cc[q] - rhobar);
    }

    auto widx = [&](int i,int j)->size_t{
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        return idx(i,j,Nx);
    };

    std::vector<double> phi(N,0.0);
    double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    double denom = 2.0*(idx2+idy2);
    double omega = p.sor_omega;

    for(int it=0; it<p.sor_iters; ++it){
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t k=widx(i,j);
                double phiE = phi[widx(i+1,j)];
                double phiW = phi[widx(i-1,j)];
                double phiN = phi[widx(i,j+1)];
                double phiS = phi[widx(i,j-1)];
                double phi_new = ((phiE+phiW)*idx2 + (phiN+phiS)*idy2 - rhs[k]) / denom;
                phi[k] += omega*(phi_new - phi[k]);
            }
        }
        double phibar=0.0;
        for(double v: phi) phibar += v;
        phibar /= (double)phi.size();
        for(double& v: phi) v -= phibar;
    }

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t k=widx(i,j);
            double phiE = phi[widx(i+1,j)];
            double phiW = phi[widx(i-1,j)];
            double phiN = phi[widx(i,j+1)];
            double phiS = phi[widx(i,j-1)];
            g.gx[k] = -(phiE - phiW)/(2.0*dx);
            g.gy[k] = -(phiN - phiS)/(2.0*dy);
        }
    }
    return g;
}

// ------------------------------- Face/Cell B helpers -------------------------------

static inline double get_Bx_face(const Params& p, const Bfaces& B, int i_face, int j){
    int Nx=p.Nx, Ny=p.Ny;
    if (p.periodic_x) i_face=imod(i_face,Nx+1); else i_face=std::clamp(i_face,0,Nx);
    if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
    return B.Bx_f[(size_t)j*(size_t)(Nx+1) + (size_t)i_face];
}
static inline double get_By_face(const Params& p, const Bfaces& B, int i, int j_face){
    int Nx=p.Nx, Ny=p.Ny;
    if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
    if (p.periodic_y) j_face=imod(j_face,Ny+1); else j_face=std::clamp(j_face,0,Ny);
    return B.By_f[(size_t)j_face*(size_t)Nx + (size_t)i];
}

static inline double bx_cc_from_faces(const Bfaces& B, int i, int j, const Params& p){
    int Nx=p.Nx, Ny=p.Ny;
    auto fidx = [&](int fi,int fj)->size_t{
        if (p.periodic_x) fi=imod(fi,Nx+1); else fi=std::clamp(fi,0,Nx);
        if (p.periodic_y) fj=imod(fj,Ny);   else fj=std::clamp(fj,0,Ny-1);
        return (size_t)fj*(size_t)(Nx+1) + (size_t)fi;
    };
    double bL = B.Bx_f[fidx(i, j)];
    double bR = B.Bx_f[fidx(i+1, j)];
    return 0.5*(bL+bR);
}

static inline double by_cc_from_faces(const Bfaces& B, int i, int j, const Params& p){
    int Nx=p.Nx, Ny=p.Ny;
    auto fidx = [&](int fi,int fj)->size_t{
        if (p.periodic_x) fi=imod(fi,Nx);   else fi=std::clamp(fi,0,Nx-1);
        if (p.periodic_y) fj=imod(fj,Ny+1); else fj=std::clamp(fj,0,Ny);
        return (size_t)fj*(size_t)Nx + (size_t)fi;
    };
    double bD = B.By_f[fidx(i, j)];
    double bU = B.By_f[fidx(i, j+1)];
    return 0.5*(bD+bU);
}

static inline Prim get_prim_cc(const Params& p, const Ucc& U, const Bfaces& B, int i, int j){
    int Nx=p.Nx, Ny=p.Ny;
    if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
    if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
    size_t q=idx(i,j,Nx);
    double Bx = bx_cc_from_faces(B,i,j,p);
    double By = by_cc_from_faces(B,i,j,p);
    return cons_to_prim(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,U.Bz[q]);
}

// ------------------------------- MUSCL reconstruction -------------------------------

static inline void recon_x(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, Prim& WL, Prim& WR){
    auto Wm = get_prim_cc(p,U,B,i-1,j);
    auto Wc = get_prim_cc(p,U,B,i,j);
    auto Wp = get_prim_cc(p,U,B,i+1,j);

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

static inline void recon_y(const Params& p, const Ucc& U, const Bfaces& B, int i, int j, Prim& WL, Prim& WR){
    auto Wm = get_prim_cc(p,U,B,i,j-1);
    auto Wc = get_prim_cc(p,U,B,i,j);
    auto Wp = get_prim_cc(p,U,B,i,j+1);

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

// ------------------------------- HLLD (x-normal) -------------------------------

static inline double total_pressure(const Prim& W){
    double B2 = W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz;
    return W.P + 0.5*B2;
}

struct FluxX {
    double frho, fmx, fmy, fmz, fE;
    double fBy, fBz;
    double Ez;
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
    F.Ez   = -(F.fBy);
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
        FH.Ez = -(FH.fBy);
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
        F.Ez   = -(F.fBy);
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
    double Ez;
};

static inline FluxY hlld_y(const Params& p, Prim WL, Prim WR, double Bn){
    WL.By = Bn;
    WR.By = Bn;

    // map y-normal -> x-normal:
    // (u',v',w') = (v,u,w), (Bx',By',Bz') = (By,Bx,Bz)
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
    Gy.Ez   = Gy.fBx; // Ez at y-face is flux(Bx)
    return Gy;
}

// ------------------------------- Resistive CT term: Jz at edge -------------------------------

static inline double Jz_edge(const Params& p, const Bfaces& B, int ie, int je, double dx, double dy){
    double ByR = get_By_face(p,B,ie,   je);
    double ByL = get_By_face(p,B,ie-1, je);
    double dBydx = (ByR - ByL)/dx;

    double BxU = get_Bx_face(p,B,ie, je);
    double BxD = get_Bx_face(p,B,ie, je-1);
    double dBxdy = (BxU - BxD)/dy;

    return dBydx - dBxdy;
}

// ------------------------------- Time step -------------------------------

static inline double compute_dt(const Params& p, const Ucc& U, const Bfaces& B, double dx, double dy){
    int Nx=p.Nx, Ny=p.Ny;
    double maxs=1e-14;

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            Prim W = get_prim_cc(p,U,B,i,j);
            maxs = std::max(maxs, std::abs(W.u)+fast_speed_x(p,W));
            maxs = std::max(maxs, std::abs(W.v)+fast_speed_y(p,W));
        }
    }
    double dt_h = p.cfl * std::min(dx,dy) / maxs;

    double dt_r = 1e100;
    if (p.eta > 0.0){
        double h = std::min(dx,dy);
        dt_r = 0.25*h*h/p.eta;
    }
    return std::min(dt_h, dt_r);
}

// ------------------------------- Sources -------------------------------

static inline void add_gravity_sources(const Params& p, Ucc& U, const Bfaces& B,
                                       const GravField& g, double dt){
    if (!p.gravity_on) return;
    int Nx=p.Nx, Ny=p.Ny;
    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);
            double rho=U.rho[q];
            double gx=g.gx[q], gy=g.gy[q];
            double mx=U.mx[q], my=U.my[q];
            double u=mx/std::max(rho,1e-300);
            double v=my/std::max(rho,1e-300);

            U.mx[q] += dt * rho * gx;
            U.my[q] += dt * rho * gy;
            U.E[q]  += dt * rho * (u*gx + v*gy);
        }
    }

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);
            double Bx=bx_cc_from_faces(B,i,j,p);
            double By=by_cc_from_faces(B,i,j,p);
            enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,U.Bz[q]);
        }
    }
}

static inline void add_joule_heating_and_Bz_resistive(const Params& p, Ucc& U, const Bfaces& B,
                                                     double dx, double dy, double dt){
    if (p.eta <= 0.0) return;
    int Nx=p.Nx, Ny=p.Ny;

    auto atBz = [&](int i,int j)->double{
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        return U.Bz[idx(i,j,Nx)];
    };

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);

            double dBzdy = (atBz(i,j+1)-atBz(i,j-1))/(2*dy);
            double dBzdx = (atBz(i+1,j)-atBz(i-1,j))/(2*dx);

            double ByR = get_By_face(p,B,i+1,j);
            double ByL = get_By_face(p,B,i,  j);
            double dBydx = (ByR - ByL)/dx;

            double BxU = get_Bx_face(p,B,i,j+1);
            double BxD = get_Bx_face(p,B,i,j);
            double dBxdy = (BxU - BxD)/dy;

            double Jx = dBzdy;
            double Jy = -dBzdx;
            double Jz = dBydx - dBxdy;

            double J2 = Jx*Jx + Jy*Jy + Jz*Jz;
            U.E[q] += dt * p.eta * J2;
        }
    }

    // Bz diffusion for constant eta: dBz/dt = eta * Laplacian(Bz)
    std::vector<double> nBz((size_t)Nx*(size_t)Ny,0.0);
    double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            double lap = (atBz(i+1,j)-2*atBz(i,j)+atBz(i-1,j))*idx2
                       + (atBz(i,j+1)-2*atBz(i,j)+atBz(i,j-1))*idy2;
            nBz[idx(i,j,Nx)] = atBz(i,j) + dt*p.eta*lap;
        }
    }
    U.Bz.swap(nBz);

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);
            double Bx=bx_cc_from_faces(B,i,j,p);
            double By=by_cc_from_faces(B,i,j,p);
            enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,U.Bz[q]);
        }
    }
}

// ------------------------------- One RK stage: FV + CT + sources -------------------------------

static inline void advance_stage(const Params& p, const Ucc& U0, const Bfaces& B0,
                                 Ucc& U1, Bfaces& B1,
                                 const GravField& g, double dx, double dy, double dt){
    int Nx=p.Nx, Ny=p.Ny;
    size_t N=(size_t)Nx*(size_t)Ny;

    U1 = U0;
    B1 = B0;

    std::vector<std::array<double,7>> Fx((size_t)(Nx+1)*(size_t)Ny);
    std::vector<std::array<double,7>> Gy((size_t)Nx*(size_t)(Ny+1));

    std::vector<double> Ez_x((size_t)(Nx+1)*(size_t)Ny, 0.0);
    std::vector<double> Ez_y((size_t)Nx*(size_t)(Ny+1), 0.0);

    // x-faces
    for(int j=0;j<Ny;j++){
        for(int i=0;i<=Nx;i++){
            int iL=i-1, iR=i;
            if (p.periodic_x){ iL=imod(iL,Nx); iR=imod(iR,Nx); }
            else { iL=std::clamp(iL,0,Nx-1); iR=std::clamp(iR,0,Nx-1); }

            Prim WLL,WLR,WRL,WRR;
            recon_x(p,U0,B0,iL,j,WLL,WLR);
            recon_x(p,U0,B0,iR,j,WRL,WRR);

            Prim WL = WLR;
            Prim WR = WRL;

            double Bn = get_Bx_face(p,B0,i,j);
            FluxX F = hlld_x(p, WL, WR, Bn);

            Fx[(size_t)j*(size_t)(Nx+1)+(size_t)i] = {F.frho,F.fmx,F.fmy,F.fmz,F.fE,F.fBy,F.fBz};
            Ez_x[(size_t)j*(size_t)(Nx+1)+(size_t)i] = F.Ez;
        }
    }

    // y-faces
    for(int j=0;j<=Ny;j++){
        for(int i=0;i<Nx;i++){
            int jD=j-1, jU=j;
            if (p.periodic_y){ jD=imod(jD,Ny); jU=imod(jU,Ny); }
            else { jD=std::clamp(jD,0,Ny-1); jU=std::clamp(jU,0,Ny-1); }

            Prim WDD,WDU,WUD,WUU;
            recon_y(p,U0,B0,i,jD,WDD,WDU);
            recon_y(p,U0,B0,i,jU,WUD,WUU);

            Prim WL = WDU;
            Prim WR = WUD;

            double Bn = get_By_face(p,B0,i,j);
            FluxY G = hlld_y(p, WL, WR, Bn);

            Gy[(size_t)j*(size_t)Nx+(size_t)i] = {G.frho,G.fmx,G.fmy,G.fmz,G.fE,G.fBx,G.fBz};
            Ez_y[(size_t)j*(size_t)Nx+(size_t)i] = G.Ez;
        }
    }

    // FV update
    double invdx=1.0/dx, invdy=1.0/dy;
    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);

            auto& FxR = Fx[(size_t)j*(size_t)(Nx+1)+(size_t)(i+1)];
            auto& FxL = Fx[(size_t)j*(size_t)(Nx+1)+(size_t)(i)];
            auto& GyU = Gy[(size_t)(j+1)*(size_t)Nx+(size_t)i];
            auto& GyD = Gy[(size_t)(j)*(size_t)Nx+(size_t)i];

            double drho = -(FxR[0]-FxL[0])*invdx - (GyU[0]-GyD[0])*invdy;
            double dmx  = -(FxR[1]-FxL[1])*invdx - (GyU[1]-GyD[1])*invdy;
            double dmy  = -(FxR[2]-FxL[2])*invdx - (GyU[2]-GyD[2])*invdy;
            double dmz  = -(FxR[3]-FxL[3])*invdx - (GyU[3]-GyD[3])*invdy;
            double dE   = -(FxR[4]-FxL[4])*invdx - (GyU[4]-GyD[4])*invdy;

            double dBz  = -(FxR[6]-FxL[6])*invdx - (GyU[6]-GyD[6])*invdy;

            U1.rho[q] += dt*drho;
            U1.mx[q]  += dt*dmx;
            U1.my[q]  += dt*dmy;
            U1.mz[q]  += dt*dmz;
            U1.E[q]   += dt*dE;
            U1.Bz[q]  += dt*dBz;

            double Bx = bx_cc_from_faces(B0,i,j,p);
            double By = by_cc_from_faces(B0,i,j,p);
            enforce_floors(p, U1.rho[q],U1.mx[q],U1.my[q],U1.mz[q],U1.E[q],Bx,By,U1.Bz[q]);
        }
    }

    // CT edge Ez
    std::vector<double> Ez_edge((size_t)(Nx+1)*(size_t)(Ny+1), 0.0);

    auto ex = [&](int i,int j)->double{
        if (p.periodic_x) i=imod(i,Nx+1); else i=std::clamp(i,0,Nx);
        if (p.periodic_y) j=imod(j,Ny); else j=std::clamp(j,0,Ny-1);
        return Ez_x[(size_t)j*(size_t)(Nx+1)+(size_t)i];
    };
    auto ey = [&](int i,int j)->double{
        if (p.periodic_x) i=imod(i,Nx); else i=std::clamp(i,0,Nx-1);
        if (p.periodic_y) j=imod(j,Ny+1); else j=std::clamp(j,0,Ny);
        return Ez_y[(size_t)j*(size_t)Nx+(size_t)i];
    };

    for(int je=0; je<=Ny; ++je){
        for(int ie=0; ie<=Nx; ++ie){
            double ez_ideal = 0.25*( ex(ie,je-1) + ex(ie,je) + ey(ie-1,je) + ey(ie,je) );
            double ez_res = 0.0;
            if (p.eta > 0.0){
                ez_res = p.eta * Jz_edge(p,B0,ie,je,dx,dy);
            }
            Ez_edge[(size_t)je*(size_t)(Nx+1)+(size_t)ie] = ez_ideal + ez_res;
        }
    }

    // CT update faces
    for(int j=0;j<Ny;j++){
        for(int i=0;i<=Nx;i++){
            int je0=j, je1=j+1;
            if (p.periodic_y){ je0=imod(je0,Ny+1); je1=imod(je1,Ny+1); }
            else { je0=std::clamp(je0,0,Ny); je1=std::clamp(je1,0,Ny); }

            size_t f=(size_t)j*(size_t)(Nx+1)+(size_t)i;
            double ezU = Ez_edge[(size_t)je1*(size_t)(Nx+1)+(size_t)i];
            double ezD = Ez_edge[(size_t)je0*(size_t)(Nx+1)+(size_t)i];
            B1.Bx_f[f] += -(dt/dy)*(ezU - ezD);
        }
    }
    for(int j=0;j<=Ny;j++){
        for(int i=0;i<Nx;i++){
            int ie0=i, ie1=i+1;
            if (p.periodic_x){ ie0=imod(ie0,Nx+1); ie1=imod(ie1,Nx+1); }
            else { ie0=std::clamp(ie0,0,Nx); ie1=std::clamp(ie1,0,Nx); }

            size_t f=(size_t)j*(size_t)Nx+(size_t)i;
            double ezR = Ez_edge[(size_t)j*(size_t)(Nx+1)+(size_t)ie1];
            double ezL = Ez_edge[(size_t)j*(size_t)(Nx+1)+(size_t)ie0];
            B1.By_f[f] += +(dt/dx)*(ezR - ezL);
        }
    }

    add_gravity_sources(p, U1, B1, g, dt);
    add_joule_heating_and_Bz_resistive(p, U1, B1, dx, dy, dt);
}

// ------------------------------- RK2 -------------------------------

static inline void rk2_step(const Params& p, Ucc& U, Bfaces& B, const GravField& g,
                            double dx, double dy, double dt){
    Ucc U1; Bfaces B1;
    advance_stage(p, U, B, U1, B1, g, dx, dy, dt);

    GravField g1 = compute_poisson_gravity_periodic(p, U1.rho, dx, dy);

    Ucc U2; Bfaces B2;
    advance_stage(p, U1, B1, U2, B2, g1, dx, dy, dt);

    size_t N=U.rho.size();
    for(size_t q=0;q<N;q++){
        U.rho[q] = 0.5*(U.rho[q] + U2.rho[q]);
        U.mx[q]  = 0.5*(U.mx[q]  + U2.mx[q]);
        U.my[q]  = 0.5*(U.my[q]  + U2.my[q]);
        U.mz[q]  = 0.5*(U.mz[q]  + U2.mz[q]);
        U.E[q]   = 0.5*(U.E[q]   + U2.E[q]);
        U.Bz[q]  = 0.5*(U.Bz[q]  + U2.Bz[q]);
    }
    for(size_t k=0;k<B.Bx_f.size();k++) B.Bx_f[k] = 0.5*(B.Bx_f[k] + B2.Bx_f[k]);
    for(size_t k=0;k<B.By_f.size();k++) B.By_f[k] = 0.5*(B.By_f[k] + B2.By_f[k]);

    int Nx=p.Nx, Ny=p.Ny;
    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            size_t q=idx(i,j,Nx);
            double Bx=bx_cc_from_faces(B,i,j,p);
            double By=by_cc_from_faces(B,i,j,p);
            enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx,By,U.Bz[q]);
        }
    }
}

// ------------------------------- Diagnostics / output -------------------------------

static inline double divB_cc(const Params& p, const Bfaces& B, int i, int j, double dx, double dy){
    double BxR = get_Bx_face(p,B,i+1,j);
    double BxL = get_Bx_face(p,B,i,  j);
    double ByU = get_By_face(p,B,i,j+1);
    double ByD = get_By_face(p,B,i,j);
    return (BxR-BxL)/dx + (ByU-ByD)/dy;
}

static void write_h5(const Params& p,
                     const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& t,
                     const std::vector<double>& rho, const std::vector<double>& P,
                     const std::vector<double>& Mach,
                     const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& vabs,
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

    auto write3 = [&](const char* name, const std::vector<double>& a){
        hsize_t dims[3] = { (hsize_t)t.size(), (hsize_t)p.Ny, (hsize_t)p.Nx };
        hid_t sp = H5Screate_simple(3, dims, nullptr);
        hid_t ds = H5Dcreate(file, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data());
        H5Dclose(ds); H5Sclose(sp);
    };

    write1("/x", x);
    write1("/y", y);
    write1("/t", t);

    write3("/rho", rho);
    write3("/P",   P);
    write3("/Mach",Mach);
    write3("/u",   u);
    write3("/v",   v);
    write3("/vabs",vabs);

    write3("/Bx",  Bx);
    write3("/By",  By);
    write3("/Bz",  Bz);
    write3("/divB",divB);

    H5Fclose(file);
}

// ------------------------------- Main -------------------------------

int main(){
    Params p;

    int Nx=p.Nx, Ny=p.Ny;
    double dx=(p.x1-p.x0)/Nx;
    double dy=(p.y1-p.y0)/Ny;

    std::vector<double> x(Nx), y(Ny);
    for(int i=0;i<Nx;i++) x[i]=p.x0+(i+0.5)*dx;
    for(int j=0;j<Ny;j++) y[j]=p.y0+(j+0.5)*dy;

    size_t N=(size_t)Nx*(size_t)Ny;

    Ucc U;
    U.rho.assign(N,0.0);
    U.mx.assign(N,0.0);
    U.my.assign(N,0.0);
    U.mz.assign(N,0.0);
    U.E.assign(N,0.0);
    U.Bz.assign(N,0.0);

    Bfaces B;
    B.Bx_f.assign((size_t)(Nx+1)*(size_t)Ny, 0.0);
    B.By_f.assign((size_t)Nx*(size_t)(Ny+1), 0.0);

    // -------------------------
    // Divergence-free B init via vector potential Az on corners
    // -------------------------
    double cx=0.5*(p.x0+p.x1), cy=0.5*(p.y0+p.y1);

    auto xcorner = [&](int ic){ return p.x0 + ic*dx; };
    auto ycorner = [&](int jc){ return p.y0 + jc*dy; };

    std::vector<double> Az((size_t)(Nx+1)*(size_t)(Ny+1), 0.0);
    auto az_idx = [&](int ic,int jc)->size_t{
        if (p.periodic_x) ic = imod(ic, Nx+1); else ic = std::clamp(ic,0,Nx);
        if (p.periodic_y) jc = imod(jc, Ny+1); else jc = std::clamp(jc,0,Ny);
        return (size_t)jc*(size_t)(Nx+1) + (size_t)ic;
    };

    for(int jc=0; jc<=Ny; ++jc){
        for(int ic=0; ic<=Nx; ++ic){
            double X = xcorner(ic);
            double Y = ycorner(jc);

            double rx = X - cx;
            double ry = Y - cy;
            double r  = std::sqrt(rx*rx + ry*ry) + 1e-14;

            double s = (r - p.r_core)/std::max(p.r_edge, 1e-12);
            double core_shape = 0.5*(1.0 - std::tanh(s));

            // uniform Bx = B0x -> Az = B0x * y
            double Az_base = p.B0x * (Y - p.y0);

            // localized, div-free perturbation to add some structure
            double Lx = (p.x1 - p.x0);
            double Az_pert = (p.Bshear * p.r_core) * core_shape * std::sin(2.0*M_PI*(X - p.x0)/Lx);

            Az[az_idx(ic,jc)] = Az_base + Az_pert;
        }
    }

    // derive face B
    for(int j=0; j<Ny; ++j){
        for(int i_face=0; i_face<=Nx; ++i_face){
            double AzU = Az[az_idx(i_face, j+1)];
            double AzD = Az[az_idx(i_face, j)];
            B.Bx_f[(size_t)j*(size_t)(Nx+1) + (size_t)i_face] = (AzU - AzD)/dy;
        }
    }
    for(int j_face=0; j_face<=Ny; ++j_face){
        for(int i=0; i<Nx; ++i){
            double AzR = Az[az_idx(i+1, j_face)];
            double AzL = Az[az_idx(i,   j_face)];
            B.By_f[(size_t)j_face*(size_t)Nx + (size_t)i] = -(AzR - AzL)/dx;
        }
    }

    // -------------------------
    // Core-collapse style hydro IC (uses face-averaged B in each cell)
    // -------------------------
    double gamma_eff = 1.05;

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            double X=x[i], Y=y[j];
            double rx = X - cx;
            double ry = Y - cy;
            double r  = std::sqrt(rx*rx + ry*ry) + 1e-14;

            double s = (r - p.r_core)/std::max(p.r_edge, 1e-12);
            double core_shape = 0.5*(1.0 - std::tanh(s)); // ~1 inside

            double rho = p.rho_bg + (p.rho_core - p.rho_bg)*core_shape;
            double P   = p.P0 * std::pow(rho/std::max(p.rho_bg,1e-12), gamma_eff);

            double taper = core_shape;

            double vr   = -p.vin * (std::min(r/p.r_core, 1.0)) * taper;
            double vphi = p.rot_omega * r * taper;

            double erx = rx/r, ery = ry/r;
            double epx = -ery, epy = erx;

            double u = vr*erx + vphi*epx;
            double v = vr*ery + vphi*epy;
            double w = 0.0;

            double Bx_cc = bx_cc_from_faces(B,i,j,p);
            double By_cc = by_cc_from_faces(B,i,j,p);
            double Bz_cc = 0.0;

            Prim W{rho,u,v,w,P,Bx_cc,By_cc,Bz_cc};

            size_t q=idx(i,j,Nx);
            prim_to_cons(p,W,U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],U.Bz[q]);
            enforce_floors(p, U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx_cc,By_cc,U.Bz[q]);
        }
    }

    // init divB diagnostic
    {
        double maxdiv0 = 0.0;
        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                maxdiv0 = std::max(maxdiv0, std::abs(divB_cc(p,B,i,j,dx,dy)));
            }
        }
        std::cout << "init max|divB|=" << std::setprecision(12) << maxdiv0 << "\n";
    }

    // Output storage
    std::vector<double> tstore;
    std::vector<double> rho_out,P_out,Mach_out,u_out,v_out,vabs_out,Bx_out,By_out,Bz_out,divB_out;

    auto store = [&](double t){
        tstore.push_back(t);
        size_t base=(tstore.size()-1)*N;

        rho_out.resize(tstore.size()*N);
        P_out.resize(tstore.size()*N);
        Mach_out.resize(tstore.size()*N);
        u_out.resize(tstore.size()*N);
        v_out.resize(tstore.size()*N);
        vabs_out.resize(tstore.size()*N);
        Bx_out.resize(tstore.size()*N);
        By_out.resize(tstore.size()*N);
        Bz_out.resize(tstore.size()*N);
        divB_out.resize(tstore.size()*N);

        for(int j=0;j<Ny;j++){
            for(int i=0;i<Nx;i++){
                size_t q=idx(i,j,Nx);

                double Bx_cc=bx_cc_from_faces(B,i,j,p);
                double By_cc=by_cc_from_faces(B,i,j,p);

                Prim W=cons_to_prim(p,U.rho[q],U.mx[q],U.my[q],U.mz[q],U.E[q],Bx_cc,By_cc,U.Bz[q]);

                double cs = std::sqrt(std::max(0.0, sound_speed2(p,W)));
                double vabs = std::sqrt(W.u*W.u + W.v*W.v + W.w*W.w);
                double Mach = vabs / std::max(cs, 1e-14);

                rho_out[base+q]=W.rho;
                P_out[base+q]=W.P;
                u_out[base+q]=W.u;
                v_out[base+q]=W.v;
                vabs_out[base+q]=vabs;
                Mach_out[base+q]=Mach;

                Bx_out[base+q]=Bx_cc;
                By_out[base+q]=By_cc;
                Bz_out[base+q]=W.Bz;
                divB_out[base+q]=divB_cc(p,B,i,j,dx,dy);
            }
        }
    };

    double t=0.0, next_store=0.0;
    store(t);

    while(t < p.t_end){
        GravField g = compute_poisson_gravity_periodic(p, U.rho, dx, dy);
        double dt = compute_dt(p,U,B,dx,dy);
        if (t+dt > p.t_end) dt = p.t_end - t;

        rk2_step(p,U,B,g,dx,dy,dt);
        t += dt;

        if (t >= next_store - 1e-15){
            double maxdiv=0.0;
            for(int j=0;j<Ny;j++){
                for(int i=0;i<Nx;i++){
                    maxdiv = std::max(maxdiv, std::abs(divB_cc(p,B,i,j,dx,dy)));
                }
            }
            std::cout<<"t="<<std::setprecision(7)<<t<<" dt="<<dt<<" max|divB|="<<std::setprecision(12)<<maxdiv<<"\n";
            store(t);
            next_store += p.dt_store;
        }
    }

    write_h5(p,x,y,tstore,
             rho_out,P_out,Mach_out,u_out,v_out,vabs_out,
             Bx_out,By_out,Bz_out,divB_out,
             "out2d_mhd.h5");

    std::cout<<"Wrote out2d_mhd.h5\n";
    return 0;
}
