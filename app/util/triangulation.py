from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
@dataclass(frozen=True)
class TriangulationResult:
    lat_deg: float
    lon_deg: float
def estimate_emitter_latlon_enu_ls(
    n_active: int,
    doa_local_deg: Sequence[float],
    node_lat_deg: Sequence[float],
    node_lon_deg: Sequence[float],
    heading_deg: Sequence[float],
    ref_latlon_deg: Optional[Tuple[float, float]] = None,
    earth_radius_m: float = 6_371_000.0,) -> TriangulationResult:
    """
    Estimate emitter/drone latitude & longitude from multiple bearing (DOA) lines.
    Implements:
      1) Global bearing per node: alpha_i = (heading_i + doa_local_i) mod 360
      2) ENU approximation around reference (lat0, lon0):
           x_i = R cos(lat0) (lon_i - lon0)
           y_i = R (lat_i - lat0)
         (all lat/lon in radians)
      3) Bearing unit vectors in ENU (0°=North, 90°=East):
           u_i = [sin(alpha_i), cos(alpha_i)]^T
      4) Projection matrices:
           P_i = I - u_i u_i^T
      5) Least-squares intersection:
           (Σ P_i) x = Σ (P_i p_i),  where p_i = [x_i, y_i]^T, x=[x,y]^T
      6) Convert ENU back to lat/lon:
           lat = lat0 + y/R
           lon = lon0 + x/(R cos(lat0))
    Notes:
      - Requires at least 2 active nodes to triangulate.
      - If bearings are nearly parallel or geometry is poor, the system can be ill-conditioned.
        In that case we fall back to least-squares solve.
    Returns:
      TriangulationResult with (lat_deg, lon_deg) and some diagnostics.
    """
    # ----------------------------
    # Validate inputs
    # ----------------------------
    doa_local_deg = np.asarray(doa_local_deg, dtype=float).reshape(-1)
    node_lat_deg  = np.asarray(node_lat_deg, dtype=float).reshape(-1)
    node_lon_deg  = np.asarray(node_lon_deg, dtype=float).reshape(-1)
    heading_deg   = np.asarray(heading_deg, dtype=float).reshape(-1)
    if n_active < 2:
        raise ValueError("n_active must be >= 2 (need at least two bearing lines).")
    for name, arr in [
        ("doa_local_deg", doa_local_deg),
        ("node_lat_deg", node_lat_deg),
        ("node_lon_deg", node_lon_deg),
        ("heading_deg", heading_deg),
    ]:
        if arr.size != n_active:
            raise ValueError(f"{name} must have length n_active={n_active}. Got {arr.size}.")
    # ----------------------------
    # Reference point (lat0, lon0)
    # ----------------------------
    if ref_latlon_deg is None:
        lat0_deg = float(np.mean(node_lat_deg))
        lon0_deg = float(np.mean(node_lon_deg))
    else:
        lat0_deg, lon0_deg = ref_latlon_deg
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    lat0 = lat0_deg * deg2rad
    lon0 = lon0_deg * deg2rad
    lat = node_lat_deg * deg2rad
    lon = node_lon_deg * deg2rad
    # ----------------------------
    # Node positions in ENU (x=East, y=North)
    # ----------------------------
    cos_lat0 = np.cos(lat0)
    x_nodes = earth_radius_m * cos_lat0 * (lon - lon0)
    y_nodes = earth_radius_m * (lat - lat0)
    # p_i vectors stacked as (M, 2)
    p = np.column_stack([x_nodes, y_nodes])
    # ----------------------------
    # Global bearings and unit vectors u_i
    # alpha_i = heading_i + doa_local_i  (mod 360)
    # u_i = [sin(alpha), cos(alpha)]
    # ----------------------------
    alpha_deg = (heading_deg + doa_local_deg) % 360.0
    alpha = alpha_deg * deg2rad
    u = np.column_stack([np.sin(alpha), np.cos(alpha)])  # shape (M, 2)
    # ----------------------------
    # Build normal equations: A x = b
    # A = Σ P_i, b = Σ P_i p_i
    # with P_i = I - u_i u_i^T
    # ----------------------------
    I2 = np.eye(2)
    A = np.zeros((2, 2), dtype=float)
    b = np.zeros((2,), dtype=float)
    for i in range(n_active):
        ui = u[i].reshape(2, 1)               # (2,1)
        Pi = I2 - (ui @ ui.T)                 # (2,2)
        A += Pi
        b += (Pi @ p[i])
    try:
        xy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: least-squares (handles singular / near-singular)
        xy, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    x_est, y_est = float(xy[0]), float(xy[1])
    # ----------------------------
    # Convert ENU back to lat/lon (radians -> degrees)
    # ----------------------------
    lat_est = lat0 + (y_est / earth_radius_m)
    lon_est = lon0 + (x_est / (earth_radius_m * cos_lat0))
    lat_est_deg = float(lat_est * rad2deg)
    lon_est_deg = float(lon_est * rad2deg)
    return TriangulationResult(
        lat_deg=lat_est_deg,
        lon_deg=lon_est_deg,
    )
# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Suppose you have 3 active nodes
    n_active = 3
    node_lat = [6.2442, 6.2450, 6.2435]
    node_lon = [-75.5812, -75.5798, -75.5805]
    heading  = [10.0, 95.0, 210.0]          # degrees (0=N, 90=E)
    doa_local = [30.0, -15.0, 5.0]          # degrees (local array frame)
    res = estimate_emitter_latlon_enu_ls(
        n_active=n_active,
        doa_local_deg=doa_local,
        node_lat_deg=node_lat,
        node_lon_deg=node_lon,
        heading_deg=heading,
    )
    print("Estimated lat, lon:", res.lat_deg, res.lon_deg)