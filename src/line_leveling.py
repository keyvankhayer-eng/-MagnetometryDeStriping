import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep, griddata
from scipy.ndimage import gaussian_filter1d

# -----------------------------------------------------------
# Hooshang CORR4 – DeStriping Method (Khayer–Ansari 2023)
# -----------------------------------------------------------

def hooshang_corr4(
        df,
        x_col='x',
        y_col='y',
        tmi_col='tmi',
        cell=12.5,
        tie_spacing=150,
        lp_sigma=4,
        spline_smooth=0.001
):
    """
    Complete de-striping and line-leveling workflow.
    Returns input DataFrame with a new column 'CORR4'.
    """
    xi = np.arange(df[x_col].min(), df[x_col].max(), cell)
    yi = np.arange(df[y_col].min(), df[y_col].max(), cell)
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata(
        points=df[[x_col, y_col]].values,
        values=df[tmi_col].values,
        xi=(XI, YI),
        method='linear'
    )

    y_min, y_max = yi.min(), yi.max()
    ties_y = np.arange(y_min, y_max, tie_spacing)

    tie_profiles = []
    for yy in ties_y:
        idx = np.argmin(np.abs(yi - yy))
        tie_profiles.append(ZI[idx, :])

    xs = xi
    full_smooth_profiles = []
    for p in tie_profiles:
        mask = ~np.isnan(p)
        if mask.sum() < 10:
            full_smooth_profiles.append(np.full_like(p, np.nan))
            continue
        x_valid = xs[mask]
        p_valid = p[mask]
        p_lp = gaussian_filter1d(p_valid, sigma=lp_sigma)
        tck = splrep(x_valid, p_lp, s=spline_smooth)
        p_bs = splev(xs, tck)
        full_smooth_profiles.append(p_bs)

    ERR = np.full_like(ZI, np.nan)
    for i, yy in enumerate(ties_y):
        idx = np.argmin(np.abs(yi - yy))
        ERR[idx, :] = full_smooth_profiles[i]

    ERR_interp = np.full_like(ZI, np.nan)
    for j in range(len(xi)):
        y_valid, e_valid = [], []
        for i, yy in enumerate(ties_y):
            idx = np.argmin(np.abs(yi - yy))
            if not np.isnan(ERR[idx, j]):
                y_valid.append(yy)
                e_valid.append(ERR[idx, j])
        if len(y_valid) >= 2:
            tck_y = splrep(y_valid, e_valid, s=spline_smooth)
            ERR_interp[:, j] = splev(yi, tck_y)

    ZI_corr = ZI - ERR_interp

    points_corr = []
    for i in range(len(yi)):
        for j in range(len(xi)):
            if not np.isnan(ZI_corr[i, j]):
                points_corr.append([xi[j], yi[i], ZI_corr[i, j]])

    df_corr = pd.DataFrame(points_corr, columns=[x_col, y_col, 'CORR4'])
    return pd.merge(df, df_corr, on=[x_col, y_col], how='left')
