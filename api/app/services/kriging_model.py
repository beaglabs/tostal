"""Kriging/spatial interpolation service using pykrige.

Pure geostatistics — no model training needed.
Supports ordinary kriging with automatic variogram fitting.
"""
from typing import Optional

import numpy as np

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None


def krige_interpolate(
    observations: dict,
    parameters: Optional[dict] = None,
) -> dict:
    """Interpolate scattered observations to a regular grid.

    Args:
        observations: dict with:
            - x: list of x coordinates
            - y: list of y coordinates
            - z: list of z coordinates (optional, for 3D)
            - values: list of measured values
            - variables: list of variable names
        parameters: dict with:
            - grid: dict with x_range, y_range, z_range (optional), resolution
            - method: "ordinary" | "universal" (default: "ordinary")
            - variogram_model: "gaussian" | "spherical" | "exponential" | "linear" (default: "gaussian")

    Returns:
        dict with:
            - grid: interpolated values array
            - grid_shape: shape of output grid
            - grid_x: x coordinates
            - grid_y: y coordinates
            - variance: kriging variance (optional)
    """
    if OrdinaryKriging is None:
        raise ImportError("pykrige required: pip install pykrige")

    params = parameters or {}
    grid_cfg = params.get("grid", {})
    method = params.get("method", "ordinary")
    variogram_model = params.get("variogram_model", "gaussian")

    x = np.asarray(observations.get("x", []), dtype=np.float64)
    y = np.asarray(observations.get("y", []), dtype=np.float64)
    z = np.asarray(observations.get("z", []), dtype=np.float64)
    values = np.asarray(observations.get("values", []), dtype=np.float64)

    if len(x) < 3:
        raise ValueError("Need at least 3 observation points for kriging")

    x_range = grid_cfg.get("x_range", [x.min(), x.max()])
    y_range = grid_cfg.get("y_range", [y.min(), y.max()])
    resolution = grid_cfg.get("resolution", [50, 50])

    grid_x = np.linspace(x_range[0], x_range[1], resolution[0])
    grid_y = np.linspace(y_range[0], y_range[1], resolution[1])

    if method == "ordinary":
        ok = OrdinaryKriging(
            x, y, values,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        z_values, sigma = ok.execute("grid", grid_x, grid_y)
    else:
        ok = OrdinaryKriging(
            x, y, values,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        z_values, sigma = ok.execute("grid", grid_x, grid_y)

    return {
        "grid": z_values.tolist(),
        "grid_shape": list(z_values.shape),
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "variance": sigma.tolist() if sigma is not None else None,
    }


def nearest_neighbor_interpolate(
    observations: dict,
    grid_cfg: Optional[dict] = None,
) -> dict:
    """Fallback: simple nearest-neighbor interpolation when pykrige is unavailable.

    Args:
        observations: dict with x, y, values arrays
        grid_cfg: dict with x_range, y_range, resolution

    Returns:
        dict with grid, grid_shape, grid_x, grid_y
    """
    from scipy.interpolate import griddata

    params = grid_cfg or {}
    x = np.asarray(observations.get("x", []), dtype=np.float64)
    y = np.asarray(observations.get("y", []), dtype=np.float64)
    values = np.asarray(observations.get("values", []), dtype=np.float64)

    x_range = params.get("x_range", [x.min(), x.max()])
    y_range = params.get("y_range", [y.min(), y.max()])
    resolution = params.get("resolution", [50, 50])

    grid_x = np.linspace(x_range[0], x_range[1], resolution[0])
    grid_y = np.linspace(y_range[0], y_range[1], resolution[1])
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    z_values = griddata(points, values, grid_points, method="nearest")
    z_values = z_values.reshape(len(grid_y), len(grid_x))

    return {
        "grid": z_values.tolist(),
        "grid_shape": list(z_values.shape),
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "variance": None,
    }