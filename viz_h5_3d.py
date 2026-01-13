#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_time_axis(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    if arr.ndim < 4:
        raise ValueError("Field must have ndim>=4 to visualize 3D time series.")
    if arr.shape[0] == len(t):
        return arr
    for ax in range(arr.ndim):
        if arr.shape[ax] == len(t):
            return np.moveaxis(arr, ax, 0)
    raise ValueError(f"Could not find time axis matching len(t)={len(t)} in shape {arr.shape}")


def load_field(f: h5py.File, name: str, t: np.ndarray) -> np.ndarray:
    if name not in f:
        raise KeyError(f"Field {name} not found in file.")
    arr = f[name][:]
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Field {name} is not numeric.")
    return pick_time_axis(arr, t)


def downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr
    return arr[:, ::stride, ::stride, ::stride]


def voxel_colors(data: np.ndarray, cmap, vmin: float, vmax: float, alpha: float) -> np.ndarray:
    norm = np.clip((data - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    colors = cmap(norm)
    colors[..., 3] = alpha
    return colors


def render_frame(ax, data: np.ndarray, threshold: float, cmap, vmin: float, vmax: float, alpha: float) -> None:
    ax.clear()
    filled = data >= threshold
    if not np.any(filled):
        ax.set_title("No voxels above threshold")
        return
    colors = voxel_colors(data, cmap, vmin, vmax, alpha)
    ax.voxels(filled, facecolors=colors, edgecolor="k", linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D voxel visualization for 4D HDF5 fields (Nt,Nz,Ny,Nx).")
    parser.add_argument("h5file", help="Input HDF5 file")
    parser.add_argument("--field", default="/rho", help="Field path to visualize")
    parser.add_argument("--outdir", default="gifs/3d", help="Output directory for frames/gif")
    parser.add_argument("--stride", type=int, default=2, help="Downsample stride for each axis")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile threshold for voxels")
    parser.add_argument("--alpha", type=float, default=0.6, help="Voxel alpha")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames to render")
    parser.add_argument("--gif", action="store_true", help="Write animated gif")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")

    args = parser.parse_args()
    ensure_dir(args.outdir)

    with h5py.File(args.h5file, "r") as f:
        if "/t" not in f:
            raise KeyError("/t dataset not found in file.")
        t = f["/t"][:]
        data = load_field(f, args.field, t)

    data = downsample(data, args.stride)
    nt = data.shape[0]
    nframes = min(nt, args.max_frames)

    cmap = cm.get_cmap("viridis")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _frame(i):
        frame = data[i]
        threshold = np.percentile(frame, args.percentile)
        vmin = float(np.min(frame))
        vmax = float(np.max(frame))
        render_frame(ax, frame, threshold, cmap, vmin, vmax, args.alpha)
        ax.set_title(f"{args.field} t={t[i]:.4f}")
        return []

    if args.gif:
        anim = FuncAnimation(fig, _frame, frames=nframes, interval=1000 // args.fps)
        outpath = os.path.join(args.outdir, "volume.gif")
        anim.save(outpath, writer=PillowWriter(fps=args.fps))
        print(f"Wrote {outpath}")
    else:
        for i in range(nframes):
            _frame(i)
            outpath = os.path.join(args.outdir, f"frame_{i:04d}.png")
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Wrote {nframes} frames to {args.outdir}")


if __name__ == "__main__":
    main()
