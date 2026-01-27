import numpy as np
from scipy.stats import binned_statistic_dd

from corrections import features

def fill_scale_factors(sf, max_k=5, fallback=1.0):
    filled = sf.copy()

    for i_eta in range(sf.shape[-1]):
        filled[:, :, i_eta] = fill_nan_adaptive_2d(sf[:, :, i_eta], max_k=max_k, fallback=fallback)

    return filled


def fill_nan_adaptive_2d(arr, max_k=3, fallback=1.0, threshold=30):    # threshold is min num bins
    # assert max_k % 2 == 1, "max_k must be odd"
    filled = arr.copy()
    nx, ny = arr.shape

    nan_idx = np.argwhere( (np.isnan(arr) | (arr < threshold)) )    # find indicies of nan bins
    for ix, iy in nan_idx:    # loop over every nan bin
        found = False
        for r in range(1, (max_k // 2) + 1, 1):    # 1, 2, 3, ..., INCLUDES MAX_K
            # handle boundaries
            x0 = max(ix - r, 0)
            x1 = min(ix + r + 1, nx)
            y0 = max(iy - r, 0)
            y1 = min(iy + r + 1, ny)

            window = arr[x0:x1, y0:y1]
            valid = window[~np.isnan(window)]    # find any non nans in window

            if valid.size > 0:    # if is any non nans then calculate the mean for that window
                filled[ix, iy] = valid.mean()
                found = True
                break    # stop the loop using smallest window size
            # else next iter of loop

        if not found:    # if all windows only contained nans, use fallback
            filled[ix, iy] = fallback

    return filled


def histogram(df, nBins=100, nans=1, how="median", max_k=5):
    pt_response = df["pt_response"].to_numpy()
    mass_response = df["mass_response"].to_numpy()    

    bin_edges = []
    vals = []

    for v in features:
        if (v == "pt") or (v == "mass"):    # if pt or mass then quantile bin
            values = df[v].to_numpy()
            vals.append(values)
            bin_edges.append( np.unique( np.quantile(values, np.linspace(0, 1, nBins + 1)) ) )
        elif v == "eta":    # if eta just use fixed bins
            values = df["eta"].to_numpy()
            vals.append(values)
            bin_edges.append(np.linspace(-2.5, 2.5, 11))
        else:
            raise Exception("Invalid variable in l1_vars!")

    train_numpy = np.column_stack(vals)
    
    pt_resp = binned_statistic_dd(
        train_numpy,
        pt_response,
        statistic=how,
        bins=bin_edges
    )[0]

    mass_resp = binned_statistic_dd(
        train_numpy,
        mass_response,
        statistic=how,
        bins=bin_edges
    )[0]

    counts = binned_statistic_dd(
        train_numpy,
        np.ones(len(train_numpy)),
        statistic="count",
        bins=bin_edges
    )[0]

    pt_sf_raw = 1.0 / pt_resp
    mass_sf_raw = 1.0 / mass_resp

    if len(features) == 3:
        pt_sf = fill_scale_factors(pt_sf_raw, max_k=max_k, fallback=nans)
        mass_sf = fill_scale_factors(mass_sf_raw, max_k=max_k, fallback=nans)
    else:
        pt_sf = 1.0 / np.nan_to_num(pt_resp, nan=nans)
        mass_sf = 1.0 / np.nan_to_num(mass_resp, nan=nans)

    return pt_sf, mass_sf, bin_edges, counts


def get_scale_factors(df, pt_sf, mass_sf, bin_edges):
    bin_idx = [ np.digitize(df[var].to_numpy(), bin_edges[i]) - 1 for i, var in enumerate(features) ]
    bin_idx = np.stack(bin_idx, axis=-1)

    # check that jet falls into a valid, predefined bin
    valid = np.all( [(0 <= bin_idx[:, i]) & (bin_idx[:, i] < len(bin_edges[i]) - 1) for i in range(len(features))], axis=0 )

    # initialize output and assign values from mean_values
    pt_output, mass_output = np.ones( len(df) ), np.ones( len(df) )
    pt_output[valid] = pt_sf[ tuple(bin_idx[valid].T) ]    # where jet falls into a valid bin, assign the scale factor
    mass_output[valid] = mass_sf[ tuple(bin_idx[valid].T) ]    # where jet falls into a valid bin, assign the scale factor

    return pt_output, mass_output