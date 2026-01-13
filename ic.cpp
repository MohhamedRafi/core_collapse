
    /* Initial condition: weak perturbations + central overdensity (periodic domain)
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    const double cx = 0.5*(p.x0+p.x1);
    const double cy = 0.5*(p.y0+p.y1);

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            const int q = i + Nx*j;
            const double x = g.x[i];
            const double y = g.y[j];

            W4 W;
            W.d = 1.0 + 0.01 * uni(rng);
            W.u = 0.0;
            W.v = 0.0;
            W.P = 1.0;

            const double r2 = sqr(x-cx) + sqr(y-cy);
            W.d += 0.2 * std::exp(-r2 / sqr(0.08));

            U[q] = prim_to_cons(W, p);
        }
    } */

    // /* -------------------- Initial condition: mild central overdensity + random vortical velocity -------------------- */
    // std::mt19937 rng(p.seed);
    // std::normal_distribution<double> gauss(0.0, 1.0);

    // const double rho0   = 1.0;
    // const double P0     = 1.0;

    // // Density noise (fractional) and mild central bump
    // const double Arho   = 0.01;   // 1% random density fluctuations
    // const double Ac     = 0.02;   // 2% central overdensity (mild!)
    // const double sigma  = 0.12 * std::min(g.Lx, g.Ly);  // bump width

    // // Target RMS Mach number for initial stirring
    // const double Mach0  = 0.3;    // try 0.1 .. 0.5
    // const double a0     = std::sqrt(p.gamma * P0 / rho0);

    // // Build a random streamfunction psi on the grid, then v = curl(psi zhat):
    // //   u =  dpsi/dy,  v = -dpsi/dx
    // std::vector<double> psi(N, 0.0);
    // for(int j=0;j<Ny;j++){
    //     for(int i=0;i<Nx;i++){
    //         const int q = i + Nx*j;
    //         psi[q] = gauss(rng);
    //     }
    // }

    // // Compute velocities from psi (centered differences), periodic
    // std::vector<double> u0(N, 0.0), v0(N, 0.0);
    // const double idx2 = 1.0/(2.0*g.dx);
    // const double idy2 = 1.0/(2.0*g.dy);

    // for(int j=0;j<Ny;j++){
    //     int jp = imod(j+1, Ny);
    //     int jm = imod(j-1, Ny);
    //     for(int i=0;i<Nx;i++){
    //         int ip = imod(i+1, Nx);
    //         int im = imod(i-1, Nx);

    //         const int q   = i  + Nx*j;
    //         const int qip = ip + Nx*j;
    //         const int qim = im + Nx*j;
    //         const int qjp = i  + Nx*jp;
    //         const int qjm = i  + Nx*jm;

    //         const double dpsi_dx = (psi[qip] - psi[qim]) * idx2;
    //         const double dpsi_dy = (psi[qjp] - psi[qjm]) * idy2;

    //         u0[q] =  dpsi_dy;
    //         v0[q] = -dpsi_dx;
    //     }
    // }
    // /* ------------------------------------------------------------------------------- */

    // // Scale velocities to target RMS Mach number
    // double vrms2 = 0.0;
    // for(int q=0;q<N;q++){
    //     vrms2 += u0[q]*u0[q] + v0[q]*v0[q];
    // }
    // vrms2 /= (double)N;
    // double vrms = std::sqrt(vrms2);
    // double vscale = (vrms > 0.0) ? (Mach0 * a0 / vrms) : 0.0;

    // for(int q=0;q<N;q++){
    //     u0[q] *= vscale;
    //     v0[q] *= vscale;
    // }

    // // Now build rho, P, u, v
    // const double cx = 0.5*(p.x0 + p.x1);
    // const double cy = 0.5*(p.y0 + p.y1);

    // for(int j=0;j<Ny;j++){
    //     for(int i=0;i<Nx;i++){
    //         const int q = i + Nx*j;

    //         const double x = g.x[i];
    //         const double y = g.y[j];
    //         const double r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy);

    //         // Mild overdensity + small random noise
    //         const double bump = Ac * std::exp(-0.5 * r2 / (sigma*sigma));
    //         const double noise = Arho * gauss(rng);

    //         W4 W;
    //         W.d = rho0 * (1.0 + noise) + rho0 * bump;
    //         W.d = clamp_min(W.d, p.floor_rho);

    //         // Uniform pressure (keeps it near hydrostatic-ish initially)
    //         W.P = P0;
    //         W.P = clamp_min(W.P, p.floor_P);

    //         // Random vortical velocity field
    //         W.u = u0[q];
    //         W.v = v0[q];

    //         U[q] = prim_to_cons(W, p);
    //     }
    // }

    // // -------------------- IC: two mild overdense blobs + rotating velocity field --------------------
    std::mt19937 rng(p.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    const double rho0   = 1.0;
    const double P0     = 1.0;

    // mild density noise
    const double Arho   = 0.002;   // 0.2% noise (keep small)

    // two blobs
    const double Ac     = 1.0;    // 10% overdensity peak for each blob
    const double sigma  = 0.06 * std::min(g.Lx, g.Ly);   // blob width

    // separation between blob centers
    const double dsep   = 0.28 * std::min(g.Lx, g.Ly);   // center-to-center separation

    // rotation about midpoint (set via a target speed or target Mach)
    const double Mach0  = 0.4;     // target speed at radius r=dsep/2 in units of sound speed
    const double a0     = std::sqrt(p.gamma * P0 / rho0);

    // center of box (rotation center)
    const double xc = 0.5*(p.x0 + p.x1);
    const double yc = 0.5*(p.y0 + p.y1);

    // blob centers (placed along x-axis, symmetric)
    const double x1c = xc - 0.5*dsep;
    const double y1c = yc;
    const double x2c = xc + 0.5*dsep;
    const double y2c = yc;

    // choose angular frequency so that |v| at r0 = dsep/2 equals Mach0 * a0
    const double r0    = 0.5*dsep;
    const double v0    = Mach0 * a0;
    const double Omega = (r0 > 0.0) ? (v0 / r0) : 0.0;

    // optional: tiny radial "binding" drift toward COM (can help them approach/capture)
    const double vr_in  = 0.0;     // try 0.01*a0 if you want inspiral

    for(int j=0;j<Ny;j++){
        for(int i=0;i<Nx;i++){
            const int q = i + Nx*j;
            const double x = g.x[i];
            const double y = g.y[j];

            // distances to each blob center
            const double dx1 = x - x1c;
            const double dy1 = y - y1c;
            const double dx2 = x - x2c;
            const double dy2 = y - y2c;

            const double r1_2 = dx1*dx1 + dy1*dy1;
            const double r2_2 = dx2*dx2 + dy2*dy2;

            // density: background + two Gaussian bumps + small noise
            const double bump1 = Ac * std::exp(-0.5 * r1_2 / (sigma*sigma));
            const double bump2 = Ac * std::exp(-0.5 * r2_2 / (sigma*sigma));
            const double noise = Arho * gauss(rng);

            W4 W;
            W.d = rho0 * (1.0 + noise + bump1 + bump2);
            W.d = clamp_min(W.d, p.floor_rho);

            // pressure: uniform (keeps it simple; you can lower P0 to make gravity win faster)
            W.P = clamp_min(P0, p.floor_P);

            // rotating velocity about COM: v = Omega zhat × r
            const double rx = x - xc;
            const double ry = y - yc;

            // tangential velocity
            double u = -Omega * ry;
            double v =  Omega * rx;

            // optional inward drift toward COM (small)
            if (vr_in != 0.0){
                const double rr = std::sqrt(rx*rx + ry*ry);
                if (rr > 1e-14){
                    u += -vr_in * (rx/rr);
                    v += -vr_in * (ry/rr);
                }
            }

            // (optional) small random velocity perturbation
            const double Av = 0.0; // set 0.01*a0 if you want
            if (Av != 0.0){
                u += Av * gauss(rng);
                v += Av * gauss(rng);
            }

            W.u = u;
            W.v = v;

            U[q] = prim_to_cons(W, p);
        }
    }

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
    const double P0   = 0.2;   // lower P -> easier collapse

    // Central overdensity: rho = rho0 [1 + A exp(-r^2/2sigma^2)]
    const double A     = 0.8;                      // peak overdensity ~ 80% above background
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