#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
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


def to_log(field: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(field, eps))


def get_field_stack(f: h5py.File, name: str, t: np.ndarray) -> np.ndarray:
    if name == "/Bmag":
        bx = load_field(f, "/Bx", t)
        by = load_field(f, "/By", t)
        bz = load_field(f, "/Bz", t)
        return np.sqrt(bx * bx + by * by + bz * bz)
    return load_field(f, name, t)


def build_montage(frame: np.ndarray, log_scale: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    nz, ny, nx = frame.shape
    iz = nz // 2
    iy = ny // 2
    ix = nx // 2

    xy = frame[iz, :, :]
    xz = frame[:, iy, :]
    yz = frame[:, :, ix]
    proj = np.mean(frame, axis=0)

    if log_scale:
        xy = to_log(xy)
        xz = to_log(xz)
        yz = to_log(yz)
        proj = to_log(proj)

    return proj, {"xy": xy, "xz": xz, "yz": yz}


def render_frame(fig, axes, frame: np.ndarray, title: str, cmap: str, log_scale: bool) -> None:
    for ax in axes.flat:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

    proj, slices = build_montage(frame, log_scale)
    data_list = [slices["xy"], slices["xz"], slices["yz"], proj]
    vmin = min(float(np.min(d)) for d in data_list)
    vmax = max(float(np.max(d)) for d in data_list)

    im0 = axes[0, 0].imshow(slices["xy"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("XY mid-slice")
    axes[0, 1].imshow(slices["xz"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("XZ mid-slice")
    axes[1, 0].imshow(slices["yz"], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("YZ mid-slice")
    axes[1, 1].imshow(proj, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Column projection")

    fig.suptitle(title)
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("log10 value" if log_scale else "value")


def parse_fields(value: str) -> list[str]:
    return [field.strip() for field in value.split(",") if field.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Astro-style 3D visualization for 4D HDF5 fields (Nt,Nz,Ny,Nx).")
    parser.add_argument("h5file", help="Input HDF5 file")
    parser.add_argument("--fields", default="/rho", help="Comma-separated fields (supports /Bmag)")
    parser.add_argument("--outdir", default="gifs/3d", help="Output directory for frames/gif")
    parser.add_argument("--stride", type=int, default=2, help="Downsample stride for each axis")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames to render")
    parser.add_argument("--gif", action="store_true", help="Write animated gif")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap")
    parser.add_argument("--log", action="store_true", help="Use log10 scaling")

    args = parser.parse_args()
    ensure_dir(args.outdir)
    fields = parse_fields(args.fields)

    with h5py.File(args.h5file, "r") as f:
        if "/t" not in f:
            raise KeyError("/t dataset not found in file.")
        t = f["/t"][:]
        field_data = {name: downsample(get_field_stack(f, name, t), args.stride) for name in fields}

    nt = min(arr.shape[0] for arr in field_data.values())
    nframes = min(nt, args.max_frames)

    for name, data in field_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))

        def _frame(i):
            frame = data[i]
            render_frame(fig, axes, frame, f"{name} t={t[i]:.4f}", args.cmap, args.log)
            return []

        safe_name = name.strip("/").replace("/", "_") or "field"
        if args.gif:
            anim = FuncAnimation(fig, _frame, frames=nframes, interval=1000 // args.fps)
            outpath = os.path.join(args.outdir, f"{safe_name}.gif")
            anim.save(outpath, writer=PillowWriter(fps=args.fps))
            print(f"Wrote {outpath}")
        else:
            for i in range(nframes):
                _frame(i)
                outpath = os.path.join(args.outdir, f"{safe_name}_{i:04d}.png")
                fig.savefig(outpath, dpi=150, bbox_inches="tight")
            print(f"Wrote {nframes} frames for {name} to {args.outdir}")


if __name__ == "__main__":
    main()
