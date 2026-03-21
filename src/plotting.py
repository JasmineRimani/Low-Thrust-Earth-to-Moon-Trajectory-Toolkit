"""
Shared plotting helpers for the public example scripts.

These functions keep the examples focused on trajectory setup and reporting
instead of repeating matplotlib boilerplate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def save_orbital_history_plot(
    t_days: np.ndarray,
    coe: np.ndarray,
    mass: np.ndarray,
    save_path: str | Path,
    *,
    title: str,
    color: str = "steelblue",
    dpi: int = 180,
) -> Path:
    """Save the standard 2x3 Earth-phase history plot."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    labels = [
        "Semi-major axis [km]",
        "Eccentricity [-]",
        "Inclination [deg]",
        "Arg. of perigee [deg]",
        "RAAN [deg]",
        "Mass [kg]",
    ]
    series = [
        coe[:, 0] / 1e3,
        coe[:, 1],
        np.degrees(coe[:, 2]),
        np.degrees(coe[:, 3]),
        np.degrees(coe[:, 4]),
        mass,
    ]

    for axis, label, values in zip(axes.flat, labels, series):
        axis.plot(t_days, values, linewidth=1.4, color=color)
        axis.set_xlabel("Time [days]")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.45)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_trajectory_views(
    trajectory: np.ndarray,
    central_body_radius: float,
    save_path: str | Path,
    *,
    title: str,
    axis_unit_label: str,
    scale: float,
    trajectory_label: str,
    start_label: str = "Departure",
    end_label: str = "Arrival",
    body_label: str = "Central body",
    body_color: str = "steelblue",
    body_alpha: float = 0.25,
    reference_trajectory: np.ndarray | None = None,
    reference_label: str | None = None,
    dpi: int = 180,
) -> Path:
    """
    Save a trajectory figure with one 3-D view and two orthographic projections.

    Parameters
    ----------
    trajectory : (N, 3) Cartesian trajectory points.
    central_body_radius : radius of the central body in the same units as trajectory.
    save_path : output image path.
    axis_unit_label : label text inserted into axis units.
    scale : scale factor applied before plotting (for example 1e3 or 1e6).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    main_points = np.asarray(trajectory, dtype=float) / scale
    ref_points = None
    if reference_trajectory is not None:
        ref_points = np.asarray(reference_trajectory, dtype=float) / scale

    radius = central_body_radius / scale
    all_points = _combine_point_sets(main_points, ref_points, np.zeros((1, 3)))

    fig = plt.figure(figsize=(13, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])

    if ref_points is not None:
        ax3d.plot(
            ref_points[:, 0],
            ref_points[:, 1],
            ref_points[:, 2],
            color="0.45",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=reference_label,
        )

    ax3d.plot(
        main_points[:, 0],
        main_points[:, 1],
        main_points[:, 2],
        color="tab:blue",
        linewidth=1.6,
        label=trajectory_label,
    )
    ax3d.scatter(
        *main_points[0],
        s=48,
        color="tab:green",
        label=start_label,
        depthshade=False,
        zorder=5,
    )
    ax3d.scatter(
        *main_points[-1],
        s=48,
        color="tab:red",
        label=end_label,
        depthshade=False,
        zorder=5,
    )
    ax3d.scatter(0.0, 0.0, 0.0, s=28, color=body_color, label=body_label)
    _plot_body_sphere(ax3d, radius, body_color, body_alpha)
    _set_equal_limits_3d(ax3d, all_points, min_radius=radius)
    ax3d.set_xlabel(f"X [{axis_unit_label}]")
    ax3d.set_ylabel(f"Y [{axis_unit_label}]")
    ax3d.set_zlabel(f"Z [{axis_unit_label}]")
    ax3d.set_title("3-D view")
    ax3d.legend(fontsize=8, loc="upper left")

    for axis, title_text, idx_a, idx_b in (
        (ax_xy, "Top view (X-Y)", 0, 1),
        (ax_xz, "Side view (X-Z)", 0, 2),
    ):
        if ref_points is not None:
            axis.plot(
                ref_points[:, idx_a],
                ref_points[:, idx_b],
                color="0.45",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )

        axis.plot(
            main_points[:, idx_a],
            main_points[:, idx_b],
            color="tab:blue",
            linewidth=1.6,
        )
        axis.scatter(
            main_points[0, idx_a],
            main_points[0, idx_b],
            color="tab:green",
            s=35,
            zorder=5,
        )
        axis.scatter(
            main_points[-1, idx_a],
            main_points[-1, idx_b],
            color="tab:red",
            s=35,
            zorder=5,
        )
        axis.add_patch(Circle((0.0, 0.0), radius, color=body_color, alpha=body_alpha))
        _set_equal_limits_2d(axis, all_points[:, idx_a], all_points[:, idx_b], radius)
        axis.set_xlabel(f"{'XYZ'[idx_a]} [{axis_unit_label}]")
        axis.set_ylabel(f"{'XYZ'[idx_b]} [{axis_unit_label}]")
        axis.set_title(title_text)
        axis.grid(True, linestyle="--", alpha=0.45)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _combine_point_sets(*point_sets: np.ndarray | None) -> np.ndarray:
    valid_sets = []
    for points in point_sets:
        if points is None:
            continue
        array = np.asarray(points, dtype=float)
        if array.size == 0:
            continue
        valid_sets.append(array.reshape(-1, 3))

    if not valid_sets:
        return np.zeros((1, 3))
    return np.vstack(valid_sets)


def _set_equal_limits_3d(ax, points: np.ndarray, *, min_radius: float) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_span = max(0.5 * np.max(maxs - mins), min_radius)
    padding = max(0.08 * half_span, 1.0)
    half_span += padding

    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _set_equal_limits_2d(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    min_radius: float,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    x_center = 0.5 * (x_values.min() + x_values.max())
    y_center = 0.5 * (y_values.min() + y_values.max())
    half_span = max(
        0.5 * (x_values.max() - x_values.min()),
        0.5 * (y_values.max() - y_values.min()),
        min_radius,
    )
    padding = max(0.08 * half_span, 1.0)
    half_span += padding

    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_aspect("equal", adjustable="box")


def _plot_body_sphere(ax, radius: float, color: str, alpha: float) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
