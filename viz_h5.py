#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================
# HDF5 helpers
# ============================
def list_datasets(f: h5py.File):
    out = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append(name)
    f.visititems(visitor)
    return out

def pick_time_axis(arr, t):
    """
    We assume time is the first axis if arr.ndim>=2 and arr.shape[0]==len(t).
    Otherwise, we try to find an axis that matches len(t) and move it to axis=0.
    """
    if arr.ndim < 2:
        raise ValueError("Field must have ndim>=2 to animate over time.")
    if arr.shape[0] == len(t):
        return arr
    for ax in range(arr.ndim):
        if arr.shape[ax] == len(t):
            return np.moveaxis(arr, ax, 0)
    raise ValueError(f"Could not find time axis matching len(t)={len(t)} in shape {arr.shape}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _get_field(f: h5py.File, name: str, t: np.ndarray):
    if name not in f:
        return None
    arr = f[name][:]
    if not np.issubdtype(arr.dtype, np.number):
        return None
    try:
        return pick_time_axis(arr, t)
    except Exception:
        return None

def get_first_field(f: h5py.File, t: np.ndarray, names):
    """
    Return the first field (np.ndarray) among `names` that exists and is parseable.
    IMPORTANT: do NOT use `or` with arrays; use explicit None checks.
    """
    for nm in names:
        arr = _get_field(f, nm, t)
        if arr is not None:
            return arr
    return None

def get_velocity_fields(f, t, rho=None):
    """
    Return u,v on the same time axis as other fields.
    Supports:
      - stored primitive: /u,/v or /vx,/vy
      - stored conserved: /momx,/momy (or /mx,/my) with /rho
    """
    u = _get_field(f, "/u", t)
    if u is None:
        u = _get_field(f, "/vx", t)

    v = _get_field(f, "/v", t)
    if v is None:
        v = _get_field(f, "/vy", t)

    # If u/v not stored, try momenta
    if u is None or v is None:
        momx = _get_field(f, "/momx", t)
        if momx is None:
            momx = _get_field(f, "/mx", t)

        momy = _get_field(f, "/momy", t)
        if momy is None:
            momy = _get_field(f, "/my", t)

        if momx is not None and momy is not None:
            if rho is None:
                rho = _get_field(f, "/rho", t)
            if rho is None:
                raise RuntimeError("Need /rho to compute velocities from momentum fields.")
            rho_safe = np.maximum(rho, 1e-30)
            u = momx / rho_safe
            v = momy / rho_safe

    if u is None or v is None:
        raise RuntimeError("Could not find velocity fields: need (/u,/v) or (/vx,/vy) or (momx,momy)+rho")

    return u, v


# ============================
# Utility math
# ============================
def _uniform_spacing(coord):
    if coord is None or len(coord) < 2:
        return None
    d = np.diff(coord)
    return float(np.median(d))

def _cell_area_2d(x, y):
    dx = _uniform_spacing(x)
    dy = _uniform_spacing(y)
    if dx is None or dy is None:
        return None, None, None
    return dx, dy, dx * dy

def integrate_2d_time(F_tyx, x, y):
    _, _, dA = _cell_area_2d(x, y)
    if dA is None:
        raise ValueError("Need uniform /x and /y for 2D integrals.")
    return np.sum(F_tyx, axis=(1, 2)) * dA

def safe_log(x, eps=1e-300):
    return np.log(np.maximum(x, eps))

def sound_speed(P, rho, gamma, eps=1e-300):
    return np.sqrt(np.maximum(gamma * P / np.maximum(rho, eps), 0.0))

def compute_entropy_proxy(P, rho, gamma, eps=1e-300):
    # s ~ ln(P / rho^gamma)
    return safe_log(P, eps) - gamma * safe_log(rho, eps)

def vorticity_2d(u_yx, v_yx, x, y):
    # ω = dv/dx - du/dy
    dvdx = np.gradient(v_yx, x, axis=1)
    dudy = np.gradient(u_yx, y, axis=0)
    return dvdx - dudy

# ============================
# Radial profiles (azimuthal average)
# ============================
def radial_profile_2d(F_yx, x, y, center=None, nbins=128, rmax=None, weights=None):
    """
    Azimuthal average around center.
    weights: None or same shape as F_yx (e.g. rho for mass-weighted).
    Returns (r_centers, prof, counts).
    """
    X, Y = np.meshgrid(x, y)
    if center is None:
        cx = 0.5 * (x[0] + x[-1])
        cy = 0.5 * (y[0] + y[-1])
    else:
        cx, cy = center

    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    if rmax is None:
        rmax = float(np.max(R))

    bins = np.linspace(0.0, rmax, nbins + 1)
    idx = np.digitize(R.ravel(), bins) - 1
    idx = np.clip(idx, 0, nbins - 1)

    Fr = F_yx.ravel()
    if weights is None:
        w = np.ones_like(Fr)
    else:
        w = weights.ravel()

    num = np.bincount(idx, weights=Fr * w, minlength=nbins)
    den = np.bincount(idx, weights=w, minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins)

    prof = np.full(nbins, np.nan, dtype=float)
    m = den > 0
    prof[m] = num[m] / den[m]

    rcent = 0.5 * (bins[:-1] + bins[1:])
    return rcent, prof, cnt

# ============================
# Shock indicators
# ============================
def shock_indicator_1d(P_tr, r, eps=1e-30):
    lnP = np.log(np.maximum(P_tr, eps))
    dlnPdr = np.gradient(lnP, r, axis=1)
    return np.abs(dlnPdr)

def shock_indicator_2d(P_tyx, x, y, eps=1e-30):
    lnP = np.log(np.maximum(P_tyx, eps))
    dlnPdy = np.gradient(lnP, y, axis=1)
    dlnPdx = np.gradient(lnP, x, axis=2)
    return np.sqrt(dlnPdx * dlnPdx + dlnPdy * dlnPdy)

def compute_shock_fraction_2d(P_tyx, x, y, q=0.995):
    S = shock_indicator_2d(P_tyx, x, y)  # (Nt,Ny,Nx)
    thr = np.quantile(S.reshape(S.shape[0], -1), q, axis=1)
    shocked = (S >= thr[:, None, None])
    frac = np.mean(shocked.reshape(shocked.shape[0], -1), axis=1)
    return frac

# ============================
# GIF makers
# ============================
def make_shock_gif_1d(base_tr, P_tr, r, t, out_gif, stride=1, fps=20, q=0.995):
    idx = np.arange(0, base_tr.shape[0], stride)
    B = base_tr[idx]
    PP = P_tr[idx]
    tt = t[idx]

    S = shock_indicator_1d(PP, r)
    thr = np.quantile(S, q, axis=1)

    vmin = float(np.min(B))
    vmax = float(np.max(B))
    pad = 0.05 * (vmax - vmin + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 4))
    base_line, = ax.plot(r, B[0], lw=1.5)
    shock_line, = ax.plot(r, np.full_like(r, np.nan), lw=2.5)
    shock_line.set_color("red")

    ax.set_xlabel("r")
    ax.set_ylabel("base (rho or P)")
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Shock highlight (|d ln P/dr|)   t={tt[0]:.6f}")

    def update(k):
        base_line.set_ydata(B[k])
        mask = S[k] >= thr[k]
        shock_line.set_ydata(np.where(mask, B[k], np.nan))
        ax.set_title(f"Shock highlight (|d ln P/dr|)   t={tt[k]:.6f}")
        return (base_line, shock_line)

    anim = FuncAnimation(fig, update, frames=len(tt), interval=1000/fps, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

def make_shock_gif_2d(base_tyx, P_tyx, x, y, t, out_gif, stride=1, fps=20, q=0.995, cmap="viridis"):
    idx = np.arange(0, base_tyx.shape[0], stride)
    B = base_tyx[idx]
    PP = P_tyx[idx]
    tt = t[idx]

    S = shock_indicator_2d(PP, x, y)
    thr = np.quantile(S.reshape(S.shape[0], -1), q, axis=1)

    vmin = float(np.min(B))
    vmax = float(np.max(B))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        B[0],
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )

    shock0 = (S[0] >= thr[0]).astype(float)
    shock_im = ax.imshow(
        shock0,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=0.0, vmax=1.0,
        cmap="Reds",
        alpha=0.35
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Shock highlight (|grad ln P|)   t={tt[0]:.6f}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("base (rho or P)")

    def update(k):
        im.set_data(B[k])
        shock = (S[k] >= thr[k]).astype(float)
        shock_im.set_data(shock)
        ax.set_title(f"Shock highlight (|grad ln P|)   t={tt[k]:.6f}")
        return (im, shock_im)

    anim = FuncAnimation(fig, update, frames=len(tt), interval=1000/fps, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

def make_line_gif(field_tr, r, t, title, out_gif, stride=1, fps=20):
    idx = np.arange(0, field_tr.shape[0], stride)
    F = field_tr[idx]
    tt = t[idx]

    vmin = float(np.min(F))
    vmax = float(np.max(F))
    pad = 0.05 * (vmax - vmin + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(r, F[0])
    ax.set_xlabel("r")
    ax.set_ylabel(title)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}   t={tt[0]:.6f}")

    def update(k):
        line.set_ydata(F[k])
        ax.set_title(f"{title}   t={tt[k]:.6f}")
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(tt), interval=1000/fps, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

def make_heatmap_gif(field_tyx, x, y, t, title, out_gif, stride=1, fps=20, cmap="viridis"):
    idx = np.arange(0, field_tyx.shape[0], stride)
    F = field_tyx[idx]
    tt = t[idx]

    vmin = float(np.min(F))
    vmax = float(np.max(F))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        F[0],
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}   t={tt[0]:.6f}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title)

    def update(k):
        im.set_data(F[k])
        ax.set_title(f"{title}   t={tt[k]:.6f}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(tt), interval=1000/fps, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

# ============================
# Static plot writers
# ============================
def savefig(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_timeseries(t, series_dict, title, ylabel, out_png):
    cleaned = {k: v for k, v in series_dict.items() if v is not None and np.size(v) > 0}
    if len(cleaned) == 0:
        return
    plt.figure(figsize=(8, 4))
    for k, v in cleaned.items():
        plt.plot(t, v, label=k)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_png)

def plot_hist_overlay(data_list, labels, title, xlabel, out_png, bins=120, density=True, logy=False):
    ok = [(d, l) for d, l in zip(data_list, labels) if d is not None and np.size(d) > 0]
    if len(ok) == 0:
        return
    plt.figure(figsize=(7, 4.5))
    for dat, lab in ok:
        plt.hist(dat, bins=bins, density=density, histtype="step", label=lab)
    plt.xlabel(xlabel)
    plt.ylabel("PDF" if density else "counts")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_png)

def plot_joint_scatter(x, y, title, xlabel, ylabel, out_png, s=2, alpha=0.25):
    if x is None or y is None or np.size(x) == 0 or np.size(y) == 0:
        return
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig(out_png)

def plot_profile_set(r, profs, labels, title, ylabel, out_png):
    ok = [(p, l) for p, l in zip(profs, labels) if p is not None and np.size(p) > 0]
    if len(ok) == 0:
        return
    plt.figure(figsize=(7.5, 4.5))
    for p, lab in ok:
        plt.plot(r, p, label=lab)
    plt.xlabel("r")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_png)

def isotropic_power_spectrum_2d(field_yx, x, y):
    """
    Returns (k_centers, Pk) isotropic 1D spectrum from a 2D field.
    Assumes uniform spacing.
    """
    Ny, Nx = field_yx.shape
    dx = _uniform_spacing(x)
    dy = _uniform_spacing(y)
    if dx is None or dy is None:
        raise ValueError("Need uniform /x,/y for spectrum.")

    Fk = np.fft.fft2(field_yx)
    P2 = np.abs(Fk)**2 / (Nx * Ny)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).ravel()
    P = P2.ravel()

    Kmax = np.max(K)
    nbins = int(np.sqrt(Nx * Ny))
    nbins = max(nbins, 32)
    bins = np.linspace(0.0, Kmax, nbins + 1)
    idx = np.digitize(K, bins) - 1
    idx = np.clip(idx, 0, nbins - 1)

    num = np.bincount(idx, weights=P, minlength=nbins)
    den = np.bincount(idx, minlength=nbins)
    Pk = np.zeros(nbins)
    m = den > 0
    Pk[m] = num[m] / den[m]

    kcent = 0.5 * (bins[:-1] + bins[1:])
    return kcent[1:], Pk[1:]  # drop DC bin

def plot_spectrum(k, Pk_list, labels, title, out_png, loglog=True):
    ok = [(Pk, l) for Pk, l in zip(Pk_list, labels) if Pk is not None and np.size(Pk) > 0]
    if len(ok) == 0:
        return
    plt.figure(figsize=(7.0, 4.5))
    for Pk, lab in ok:
        plt.plot(k, Pk, label=lab)
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    savefig(out_png)

def core_shape_axis_ratio(rho_yx, x, y, frac=0.2):
    """
    Axis ratio b/a of region with rho >= frac * rho_max.
    Uses weighted covariance eigenvalues.
    """
    X, Y = np.meshgrid(x, y)
    rmax = np.max(rho_yx)
    mask = rho_yx >= (frac * rmax)
    if not np.any(mask):
        return np.nan

    w = rho_yx[mask]
    xm = np.sum(X[mask] * w) / np.sum(w)
    ym = np.sum(Y[mask] * w) / np.sum(w)
    dx = X[mask] - xm
    dy = Y[mask] - ym

    Ixx = np.sum(w * dx * dx) / np.sum(w)
    Iyy = np.sum(w * dy * dy) / np.sum(w)
    Ixy = np.sum(w * dx * dy) / np.sum(w)

    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = max(tr * tr - 4.0 * det, 0.0)
    lam1 = 0.5 * (tr + np.sqrt(disc))
    lam2 = 0.5 * (tr - np.sqrt(disc))
    if lam1 <= 0 or lam2 <= 0:
        return np.nan
    a = np.sqrt(lam1)
    b = np.sqrt(lam2)
    return b / a

# ============================
# Main
# ============================
def main():
    ap = argparse.ArgumentParser(description="HDF5 visualizer -> GIFs + diagnostics")
    ap.add_argument("h5file", help="HDF5 file (e.g. out.h5 or out2d.h5)")
    ap.add_argument("--outdir", default="gifs", help="output directory")
    ap.add_argument("--fields", default=None,
                    help="comma-separated dataset names to visualize (e.g. /rho,/P,/Mach). If omitted, auto-detect.")
    ap.add_argument("--tname", default="/t", help="time dataset name (default: /t)")
    ap.add_argument("--stride", type=int, default=2, help="frame stride (default: 2)")
    ap.add_argument("--fps", type=int, default=20, help="gif FPS (default: 20)")
    ap.add_argument("--cmap", default="viridis", help="colormap for 2D heatmaps")

    # diagnostics controls
    ap.add_argument("--diag", action="store_true", help="also write diagnostic PNGs")
    ap.add_argument("--gamma", type=float, default=1.4, help="gamma for derived fields (Mach, entropy) if not stored")
    ap.add_argument("--diag_nbins", type=int, default=128, help="radial bins for azimuthal profiles")
    ap.add_argument("--diag_times", default="0.0,0.5,1.0",
                    help="fractions of time range for snapshot diagnostics (ignored if --diag_ntimes>0)")
    ap.add_argument("--diag_ntimes", type=int, default=9,
                    help="number of snapshot times for overlay plots (default: 9). "
                         "If >0, overrides --diag_times with evenly spaced snapshots.")
    ap.add_argument("--shock_q", type=float, default=0.995, help="quantile for shock mask (default 0.995)")
    ap.add_argument("--min_bin_count", type=int, default=20,
                    help="mask radial bins with fewer than this many cells (default 20)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    diag_dir = os.path.join(args.outdir, "diagnostics")
    if args.diag:
        ensure_dir(diag_dir)

    with h5py.File(args.h5file, "r") as f:
        if args.tname not in f:
            raise SystemExit(f"Missing time dataset {args.tname}")
        t = f[args.tname][:].astype(float)

        dsets = list_datasets(f)
        if args.fields is None:
            skip = set([args.tname, "/t", "/time", "/r", "/x", "/y"])
            fields = [name for name in dsets if name not in skip]
        else:
            fields = [s.strip() for s in args.fields.split(",") if s.strip()]

        r = f["/r"][:] if "/r" in f else None
        x = f["/x"][:] if "/x" in f else None
        y = f["/y"][:] if "/y" in f else None

        print("Datasets found:", dsets)
        print("Fields selected:", fields)

        # -------------------------
        # GIFs for selected fields
        # -------------------------
        for name in fields:
            if name not in f:
                print(f"Skip missing dataset: {name}")
                continue
            arr = f[name][:]
            if not np.issubdtype(arr.dtype, np.number):
                print(f"Skip non-numeric dataset: {name}")
                continue

            arrT = pick_time_axis(arr, t)
            out_gif = os.path.join(args.outdir, name.strip("/").replace("/", "_") + ".gif")

            if arrT.ndim == 2:
                if r is None or arrT.shape[1] != len(r):
                    print(f"Skip {name}: looks 1D but /r missing or mismatched.")
                    continue
                print(f"[1D] {name} -> {out_gif}")
                make_line_gif(arrT, r, t, name, out_gif, stride=args.stride, fps=args.fps)

            elif arrT.ndim == 3:
                if x is None or y is None:
                    print(f"Skip {name}: looks 2D but /x or /y missing.")
                    continue
                Ny, Nx = arrT.shape[1], arrT.shape[2]
                if Ny != len(y) or Nx != len(x):
                    print(f"Skip {name}: shape {arrT.shape} != (Nt,len(y),len(x))")
                    continue
                print(f"[2D] {name} -> {out_gif}")
                make_heatmap_gif(arrT, x, y, t, name, out_gif, stride=args.stride, fps=args.fps, cmap=args.cmap)

            else:
                print(f"Skip {name}: unsupported ndim={arrT.ndim} (need 2 or 3)")

        # -------------------------
        # Shock GIF (uses /P)
        # -------------------------
        P = _get_field(f, "/P", t)
        if P is None:
            print("No /P found; skipping shock.gif")
        else:
            shock_gif = os.path.join(args.outdir, "shock.gif")
            base = _get_field(f, "/rho", t)
            if base is None:
                base = P

            baseT = pick_time_axis(base, t)
            PT = pick_time_axis(P, t)

            if baseT.ndim == 2 and r is not None and baseT.shape[1] == len(r) and PT.shape == baseT.shape:
                print(f"[shock 1D] -> {shock_gif}")
                make_shock_gif_1d(baseT, PT, r, t, shock_gif, stride=args.stride, fps=args.fps, q=args.shock_q)

            elif baseT.ndim == 3 and x is not None and y is not None and PT.shape == baseT.shape:
                Ny, Nx = baseT.shape[1], baseT.shape[2]
                if Ny == len(y) and Nx == len(x):
                    print(f"[shock 2D] -> {shock_gif}")
                    make_shock_gif_2d(baseT, PT, x, y, t, shock_gif, stride=args.stride,
                                      fps=args.fps, q=args.shock_q, cmap=args.cmap)
                else:
                    print("shock.gif: base/P shapes don't match x,y; skipping.")
            else:
                print("shock.gif: unsupported dimensionality or missing coords; skipping.")

        # -------------------------
        # Diagnostics
        # -------------------------
        if args.diag:
            print("[diag] computing diagnostics...")
            gamma = float(args.gamma)

            rho = _get_field(f, "/rho", t)
            u = v = None
            if rho is not None:
                u, v = get_velocity_fields(f, t, rho)

            # FIX: never use `or` with arrays; select explicitly
            Phi = get_first_field(f, t, ["/Phi", "/phi", "/potential"])

            is2d = (rho is not None and rho.ndim == 3 and x is not None and y is not None)
            if not is2d:
                print("[diag] 2D diagnostics require /rho with shape (Nt,Ny,Nx) and /x,/y. Skipping diagnostics.")
                print("Done.")
                return

            # ----- (A) Conservations & budgets
            dx, dy, dA = _cell_area_2d(x, y)
            if dA is None:
                print("[diag] non-uniform x,y spacing; skipping integral diagnostics.")
            else:
                M = integrate_2d_time(rho, x, y)
                Mfrac = (M / M[0] - 1.0) if M is not None else None

                Px = Py = None
                K = None
                if u is not None and v is not None:
                    Px = integrate_2d_time(rho * u, x, y)
                    Py = integrate_2d_time(rho * v, x, y)
                    K = integrate_2d_time(0.5 * rho * (u*u + v*v), x, y)

                Uth = None
                if P is not None:
                    Uth = integrate_2d_time(P / (gamma - 1.0), x, y)

                W = None
                if Phi is not None:
                    W = integrate_2d_time(0.5 * rho * Phi, x, y)

                Etot = None
                if (K is not None) and (Uth is not None) and (W is not None):
                    Etot = K + Uth + W

                plot_timeseries(
                    t,
                    {"M": M, "M/M0-1": Mfrac},
                    "Mass conservation",
                    "mass / fractional drift",
                    os.path.join(diag_dir, "mass_timeseries.png")
                )

                plot_timeseries(
                    t,
                    {"Px": Px, "Py": Py},
                    "Total momentum (periodic should be ~constant)",
                    "momentum",
                    os.path.join(diag_dir, "momentum_timeseries.png")
                )

                plot_timeseries(
                    t,
                    {"K": K, "U(th)": Uth, "W": W, "E=K+U+W": Etot},
                    "Energy budget (check gravity + shocks)",
                    "energy",
                    os.path.join(diag_dir, "energy_budget.png")
                )

            # ----- (B) Shock fraction vs time
            if P is not None:
                frac = compute_shock_fraction_2d(P, x, y, q=args.shock_q)
                plot_timeseries(
                    t,
                    {"shock_area_fraction": frac},
                    f"Shock area fraction (top q={args.shock_q})",
                    "fraction",
                    os.path.join(diag_dir, "shock_fraction.png")
                )

            # ----- (C) Entropy diagnostics
            if P is not None and rho is not None:
                s = compute_entropy_proxy(P, rho, gamma)  # (Nt,Ny,Nx)
                s_mean = np.sum(rho * s, axis=(1, 2)) / np.sum(rho, axis=(1, 2))
                try:
                    dS = integrate_2d_time(rho * (s - s[0]), x, y)
                except Exception:
                    dS = None
                plot_timeseries(
                    t,
                    {"<s>_mass_weighted": s_mean, "ΔS_proxy": dS},
                    "Entropy proxy (shock-driven increase)",
                    "entropy proxy",
                    os.path.join(diag_dir, "entropy_timeseries.png")
                )

            # ----- (D) Enstrophy & angular momentum
            if u is not None and v is not None:
                enst = np.zeros_like(t)
                Lz = np.zeros_like(t)
                X, Y = np.meshgrid(x, y)
                _, _, dA = _cell_area_2d(x, y)
                for it in range(len(t)):
                    w = vorticity_2d(u[it], v[it], x, y)
                    enst[it] = 0.5 * np.sum(w*w) * dA
                    Lz[it] = np.sum(rho[it] * (X * v[it] - Y * u[it])) * dA

                plot_timeseries(
                    t,
                    {"enstrophy": enst},
                    "Enstrophy (vorticity^2) vs time",
                    "enstrophy",
                    os.path.join(diag_dir, "enstrophy.png")
                )
                plot_timeseries(
                    t,
                    {"Lz": Lz},
                    "Total angular momentum Lz vs time",
                    "Lz",
                    os.path.join(diag_dir, "angular_momentum.png")
                )

            # ----- (E) Core axis ratio vs time
            axis_ratio = np.array([core_shape_axis_ratio(rho[it], x, y, frac=0.2) for it in range(len(t))])
            plot_timeseries(
                t,
                {"b/a (rho>0.2 rho_max)": axis_ratio},
                "Core axis ratio (shape diagnostic)",
                "b/a",
                os.path.join(diag_dir, "core_axis_ratio.png")
            )

            # ----- Snapshot indices: MORE TIMESTEPS
            if args.diag_ntimes and args.diag_ntimes > 0:
                idxs = np.linspace(0, len(t) - 1, args.diag_ntimes).round().astype(int).tolist()
            else:
                fracs = []
                for sfrac in args.diag_times.split(","):
                    sfrac = sfrac.strip()
                    if not sfrac:
                        continue
                    try:
                        fracs.append(float(sfrac))
                    except Exception:
                        pass
                if len(fracs) == 0:
                    fracs = [0.0, 0.5, 1.0]
                fracs = [min(max(frc, 0.0), 1.0) for frc in fracs]
                idxs = [int(round(frc * (len(t) - 1))) for frc in fracs]

            seen = set()
            idxs = [i for i in idxs if not (i in seen or seen.add(i))]
            snap_labels = [f"t={t[i]:.6g}" for i in idxs]

            # ----- (F) PDFs
            log_rho_snaps = [np.log10(np.maximum(rho[i].ravel(), 1e-300)) for i in idxs]
            plot_hist_overlay(
                log_rho_snaps, snap_labels,
                "PDF of log10(rho)",
                "log10(rho)",
                os.path.join(diag_dir, "pdf_logrho.png"),
                bins=140, density=True, logy=False
            )

            if P is not None and u is not None and v is not None:
                Mach_snaps = []
                for i in idxs:
                    a = sound_speed(P[i], rho[i], gamma)
                    Mfield = np.sqrt(u[i]*u[i] + v[i]*v[i]) / np.maximum(a, 1e-300)
                    Mach_snaps.append(Mfield.ravel())
                plot_hist_overlay(
                    Mach_snaps, snap_labels,
                    "PDF of Mach number",
                    "Mach",
                    os.path.join(diag_dir, "pdf_mach.png"),
                    bins=140, density=True, logy=True
                )

            # ----- (G) Joint scatter at final snapshot
            if P is not None:
                i = idxs[-1]
                sfield = compute_entropy_proxy(P[i], rho[i], gamma)
                plot_joint_scatter(
                    np.log10(np.maximum(rho[i].ravel(), 1e-300)),
                    sfield.ravel(),
                    f"log10(rho) vs entropy proxy ({snap_labels[-1]})",
                    "log10(rho)",
                    "s ~ ln(P/rho^gamma)",
                    os.path.join(diag_dir, "joint_logrho_entropy.png"),
                    s=2, alpha=0.2
                )

            # ----- (H) Power spectrum δrho
            try:
                kvals = None
                Pk_list = []
                for i in idxs:
                    dr = rho[i] - np.mean(rho[i])
                    k, Pk = isotropic_power_spectrum_2d(dr, x, y)
                    kvals = k
                    Pk_list.append(Pk)
                if kvals is not None:
                    plot_spectrum(
                        kvals, Pk_list, snap_labels,
                        "Isotropic power spectrum of δrho",
                        os.path.join(diag_dir, "spectrum_drho.png"),
                        loglog=True
                    )
            except Exception:
                pass

            # ----- (I) Radial profiles (COM center + mask low-count bins + vr radius floor)
            X, Y = np.meshgrid(x, y)
            i0 = idxs[-1]
            w = rho[i0]
            cx = float(np.sum(X * w) / np.sum(w))
            cy = float(np.sum(Y * w) / np.sum(w))

            RR = np.sqrt((X - cx)**2 + (Y - cy)**2)
            dxu = _uniform_spacing(x)
            dyu = _uniform_spacing(y)
            r_floor = 0.5 * min(dxu if dxu is not None else 1.0, dyu if dyu is not None else 1.0)

            rcent = None
            rho_profs, vr_profs, P_profs, s_profs = [], [], [], []

            for i in idxs:
                rcent, rho_prof, cnt = radial_profile_2d(rho[i], x, y, center=(cx, cy),
                                                         nbins=args.diag_nbins, weights=None)
                rho_prof[cnt < args.min_bin_count] = np.nan
                rho_profs.append(rho_prof)

                if u is not None and v is not None:
                    vr = (u[i] * (X - cx) + v[i] * (Y - cy)) / np.maximum(RR, r_floor)
                    _, vr_prof, cntv = radial_profile_2d(vr, x, y, center=(cx, cy),
                                                         nbins=args.diag_nbins, weights=rho[i])
                    vr_prof[cntv < args.min_bin_count] = np.nan
                    vr_profs.append(vr_prof)
                else:
                    vr_profs.append(None)

                if P is not None:
                    _, P_prof, cntp = radial_profile_2d(P[i], x, y, center=(cx, cy),
                                                        nbins=args.diag_nbins, weights=None)
                    P_prof[cntp < args.min_bin_count] = np.nan
                    P_profs.append(P_prof)

                    sfield = compute_entropy_proxy(P[i], rho[i], gamma)
                    _, s_prof, cnts = radial_profile_2d(sfield, x, y, center=(cx, cy),
                                                        nbins=args.diag_nbins, weights=rho[i])
                    s_prof[cnts < args.min_bin_count] = np.nan
                    s_profs.append(s_prof)
                else:
                    P_profs.append(None)
                    s_profs.append(None)

            if rcent is not None:
                plot_profile_set(
                    rcent, rho_profs, snap_labels,
                    "Azimuthal average density profile ⟨rho(r)⟩",
                    "rho",
                    os.path.join(diag_dir, "profile_rho.png")
                )
                plot_profile_set(
                    rcent, vr_profs, snap_labels,
                    "Mass-weighted radial velocity ⟨vr(r)⟩",
                    "vr",
                    os.path.join(diag_dir, "profile_vr.png")
                )
                plot_profile_set(
                    rcent, P_profs, snap_labels,
                    "Azimuthal average pressure profile ⟨P(r)⟩",
                    "P",
                    os.path.join(diag_dir, "profile_P.png")
                )
                plot_profile_set(
                    rcent, s_profs, snap_labels,
                    "Mass-weighted entropy proxy ⟨s(r)⟩",
                    "s ~ ln(P/rho^gamma)",
                    os.path.join(diag_dir, "profile_entropy.png")
                )

            # ----- (J) Shock radius vs time from max |dP/dr| of azimuthal avg
            if P is not None:
                r_sh = np.full_like(t, np.nan, dtype=float)
                for it in range(len(t)):
                    rcent2, Pprof, cntp = radial_profile_2d(P[it], x, y, center=(cx, cy),
                                                           nbins=args.diag_nbins, weights=None)
                    Pprof[cntp < args.min_bin_count] = np.nan
                    if np.all(np.isnan(Pprof)):
                        continue
                    Pfill = Pprof.copy()
                    good = np.isfinite(Pfill)
                    if np.any(good):
                        Pfill[~good] = np.interp(rcent2[~good], rcent2[good], Pfill[good])
                    dPdr = np.gradient(Pfill, rcent2)
                    k = int(np.argmax(np.abs(dPdr)))
                    r_sh[it] = rcent2[k]

                plot_timeseries(
                    t,
                    {"r_shock": r_sh},
                    "Shock radius estimate from max |dP/dr| of azimuthal avg",
                    "r_shock",
                    os.path.join(diag_dir, "shock_radius.png")
                )

            print("[diag] wrote diagnostics to:", diag_dir)

    print("Done.")

if __name__ == "__main__":
    main()
