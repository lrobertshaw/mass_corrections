from collections import namedtuple
import awkward as ak
import numpy as np
import uproot as up
from scipy.stats import binned_statistic_dd

ScaleFactors = namedtuple('ScaleFactors', ['pt_sfs', 'mass_sfs', 'bin_edges', 'counts'])
Thresholds = namedtuple('Thresholds', ['l1_pt', 'l1_mass', 'gen_pt', 'gen_mass'])

def fill_scale_factors(sf, max_k=5, fallback=1.0):
    """
    Fill NaNs in scale factors using pt-mass neighbours only.

    Parameters
    ----------
    sf : np.ndarray
        Scale factor array with shape (..., eta_bins)
    """
    filled = sf.copy()

    for i_eta in range(sf.shape[-1]):
        filled[:, :, i_eta] = fill_nan_adaptive_2d(sf[:, :, i_eta], max_k=max_k, fallback=fallback)

    return filled


def fill_nan_adaptive_2d(arr, max_k=5, fallback=1.0, threshold=30):    # threshold is min num bins
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


class Jet:

    required_variables = ["pt", "eta", "mass", "genpt", "genmass", "event"]    # variables required for the class to function
    # l1_vars = ["pt", "mass", "eta"]    # variables to bin on
    l1_vars = ["pt", "eta"]    # variables to bin on
    # l1_vars = ["mass", "eta"]    # variables to bin on
    
    def __init__(self, path: str, branch: str = "outnano/Jets",
                 keys: dict = {"jet_pt": "pt", "jet_eta_phys": "eta", "jet_mass": "mass", "jet_genmatch_pt": "genpt", "jet_genmatch_mass": "genmass", "event": "event"}):
        for v in Jet.required_variables:
            if v not in list(keys.values()): raise Exception("A required variable is not defined in keys!")
        self.data = self.load(path, branch, keys)


    @staticmethod
    def load(path, branch, keys):
        print("Loading data...")
        with up.open(path)[branch] as file:
            data = file.arrays( filter_name=list( keys.keys() ) )
            reject_mask = file["jet_reject"].array() == False if "jet_reject" in file.keys() else ak.ones_like(data["event"], dtype=bool)
        
        data = ak.Array({keys[field]: data[field] for field in data.fields})[reject_mask]
        print("Data loaded!\n")
        return data


    @staticmethod
    def preprocess(data, eta_limit: int = 5.0, l1_pt_range: tuple[float, float] = (0., 0.), l1_mass_range: tuple[float, float] = (0., 0.),
                   gen_pt_range: tuple[float, float] = (0., 1000.), gen_mass_range: tuple[float, float] = (0., 182.)
                   ) -> ak.highlevel.Array:
        
        # filter out gen masses outside of range
        mask = (data["genmass"] > gen_mass_range[0]) & (data["genmass"] < gen_mass_range[1])
        mask = mask & ( data["genpt"] > gen_pt_range[0]) & (data["genpt"] < gen_pt_range[1] )
        
        mask = mask & ( data["mass"] > l1_mass_range[0]) & (data["mass"] < l1_mass_range[1] )
        mask = mask & ( data["pt"] > l1_pt_range[0]) & (data["pt"] < l1_pt_range[1] )

        mask = mask & (abs(data["eta"]) < eta_limit)
        return data[mask]
    

    @staticmethod
    def test_train_split(data, train_ratio = 0.75):
        data = data[ np.random.permutation(len(data)) ]
        splitIdx = int(len(data) * train_ratio)
        data_train, data_test = data[:splitIdx], data[splitIdx:]
        return data_train, data_test
    

    @staticmethod
    def response(data_train, eps = 1e-3):
        pt_response = (data_train["pt"] + eps) / (data_train["genpt"] + eps)
        mass_response = (data_train["mass"] + eps) / (data_train["genmass"] + eps)
        return pt_response, mass_response
    

    @staticmethod
    def histogram( data_train: ak.highlevel.Array, pt_response: ak.highlevel.Array, mass_response: ak.highlevel.Array,
        nBins: int = 100, nans: int = 1, how: str = "mean" ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        bin_edges = []
        vals = []
        for v in Jet.l1_vars:
            if (v == "pt") | (v=="mass"):
                values = ak.to_numpy(data_train[v])
                vals.append( values )
                bin_edges.append( np.unique( np.quantile(values, np.linspace(0, 1, nBins + 1)) ) )
            elif v == "eta":
                vals.append( ak.to_numpy(data_train["eta"]) )
                bin_edges.append( np.linspace(-2.5, 2.5, 11) )
            else:
                raise Exception("Invalid variable in l1_vars!")

        train_numpy = np.column_stack(vals)

        pt_resp = binned_statistic_dd( train_numpy, ak.to_numpy(pt_response), statistic = how, bins = bin_edges )[0]
        mass_resp = binned_statistic_dd( train_numpy, ak.to_numpy(mass_response), statistic = how, bins = bin_edges )[0]

        counts = binned_statistic_dd(
            train_numpy, np.ones(len(train_numpy)),
            statistic='count', bins=bin_edges
        )[0]
        

        pt_sf_raw = 1.0 / pt_resp
        mass_sf_raw = 1.0 / mass_resp

        if len(Jet.l1_vars) == 3:
            pt_sf = fill_scale_factors(pt_sf_raw, max_k=5, fallback=1.0)
            mass_sf = fill_scale_factors(mass_sf_raw, max_k=5, fallback=1.0)
        else:
            pt_sf = 1 / np.nan_to_num(pt_resp, nan=nans)
            mass_sf = 1 / np.nan_to_num(mass_resp, nan=nans)

        return ScaleFactors(pt_sf, mass_sf, bin_edges, counts)
    

    @staticmethod
    def apply_scale_factors(data_test, scale_factors: ScaleFactors):
        pt_scalefactors, mass_scalefactors, bin_edges, _ = scale_factors

        bin_idx = [ np.digitize(ak.to_numpy(data_test[var]), bin_edges[i]) - 1 for i, var in enumerate(Jet.l1_vars) ]
        bin_idx = np.stack(bin_idx, axis=-1)

        # check that jet falls into a valid, predefined bin
        valid = np.all( [(0 <= bin_idx[:, i]) & (bin_idx[:, i] < len(bin_edges[i]) - 1) for i in range(len(Jet.l1_vars))], axis=0 )

        # initialize output and assign values from mean_values
        pt_output, mass_output = np.ones( len(data_test) ), np.ones( len(data_test) )
        pt_output[valid] = pt_scalefactors[ tuple(bin_idx[valid].T) ]    # where jet falls into a valid bin, assign the scale factor
        mass_output[valid] = mass_scalefactors[ tuple(bin_idx[valid].T) ]    # where jet falls into a valid bin, assign the scale factor
        
        pt_corr, mass_corr = data_test["pt"] * pt_output, data_test["mass"] * mass_output
        data_test["pt_corr"], data_test["mass_corr"] = pt_corr, mass_corr
        return data_test, pt_corr, mass_corr


    def get_scale_factors(self, **params):
        data = self.data
        self.params = params

        eta_limit = params["eta_limit"]
        gen_pt_range = params["gen_pt_range"]
        gen_mass_range = params["gen_mass_range"]
        l1_pt_range = params["l1_pt_range"]
        l1_mass_range = params["l1_mass_range"]
        nBins = params["nBins"]
        nans = params["nans"]
        train_ratio = params["train_ratio"]
        how = params["how"]
        eps = params["eps"]

        print("Shuffling jets and splitting into test and train...")
        train, test = self.test_train_split(data, train_ratio=train_ratio)
        print("Jets shuffled and split!\n")

        print("Preprocessing data...")
        train = self.preprocess(train, eta_limit=eta_limit, 
                               l1_pt_range=l1_pt_range, gen_pt_range=gen_pt_range,
                               l1_mass_range=l1_mass_range, gen_mass_range=gen_mass_range)
        print("Data preprocessed!\n")

        print("Calculating response of each jet from training data...")
        pt_response, mass_response = self.response(train, eps=eps)
        print("Responses calculated!\n")

        print("Histogramming responses, calculating mean of each bin, and determining scale factor as inverse of mean response...")
        scale_factors = self.histogram(train, pt_response, mass_response, nBins=nBins, nans=nans, how=how)
        print("Scale factors calculated!\n")

        return test, scale_factors