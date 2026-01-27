import uproot as up
from jet_mass_corrections.histo_corrections import Jet
import numpy as np
import awkward as ak

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import median_absolute_error

class BDT_Jet(Jet):

    required_variables = ["pt", "eta", "mass", "genpt", "genmass", "event"]    # variables required for the class to function
    # l1_vars = ["pt", "mass", "eta"]    # variables to bin on
    l1_vars = ["pt", "eta"]    # variables to bin on
    # l1_vars = ["mass", "eta"]    # variables to bin on
    
    def __init__(self, path: str, branch: str = "outnano/Jets",
                 keys: dict = {"jet_pt": "pt", "jet_eta_phys": "eta", "jet_mass": "mass", "jet_genmatch_pt": "genpt", "jet_genmatch_mass": "genmass", "event": "event"}):
        super().__init__(path, branch, keys)


    # @staticmethod
    # def load(path, branch, keys):
    #     print("Loading data...")
    #     with up.open(path)[branch] as file:
    #         data = file.arrays( filter_name=list( keys.keys() ) )
    #         reject_mask = file["jet_reject"].array() == False if "jet_reject" in file.keys() else ak.ones_like(data["event"], dtype=bool)
        
    #     data = ak.Array({keys[field]: data[field] for field in data.fields})[reject_mask]
    #     print("Data loaded!\n")
    #     return data


    # @staticmethod
    # def preprocess(data, eta_limit: int = 5.0, l1_pt_range: tuple[float, float] = (0., 0.), l1_mass_range: tuple[float, float] = (0., 0.),
    #                gen_pt_range: tuple[float, float] = (0., 1000.), gen_mass_range: tuple[float, float] = (0., 182.)
    #                ) -> ak.highlevel.Array:
        
    #     # filter out gen masses outside of range
    #     mask = (data["genmass"] > gen_mass_range[0]) & (data["genmass"] < gen_mass_range[1])
    #     mask = mask & ( data["genpt"] > gen_pt_range[0]) & (data["genpt"] < gen_pt_range[1] )
        
    #     mask = mask & ( data["mass"] > l1_mass_range[0]) & (data["mass"] < l1_mass_range[1] )
    #     mask = mask & ( data["pt"] > l1_pt_range[0]) & (data["pt"] < l1_pt_range[1] )

    #     mask = mask & (abs(data["eta"]) < eta_limit)
    #     return data[mask]
    

    # @staticmethod
    # def test_train_split(data, train_ratio = 0.75):
    #     data = data[ np.random.permutation(len(data)) ]
    #     splitIdx = int(len(data) * train_ratio)
    #     data_train, data_test = data[:splitIdx], data[splitIdx:]
    #     return data_train, data_test
    

    # @staticmethod
    # def response(data_train, eps = 1e-3):
    #     pt_response = (data_train["pt"] + eps) / (data_train["genpt"] + eps)
    #     mass_response = (data_train["mass"] + eps) / (data_train["genmass"] + eps)
    #     return ak.to_numpy(pt_response), ak.to_numpy(mass_response)

    @staticmethod
    def features(data_train):
        pt, mass, eta = ak.to_numpy(data_train["pt"]), ak.to_numpy(data_train["mass"]), ak.to_numpy(data_train["eta"])
        return np.column_stack([pt, mass, eta])

    @staticmethod
    def train(features, targets: tuple):
        """
        Train BDT models for pt and mass
        """

        bdt_params = dict(
            n_estimators=1,
            learning_rate=0.8,
            max_depth=6,
            min_samples_leaf=100,
            subsample=0.7,
            loss="huber",
            random_state=42
        )

        model_pt = GradientBoostingRegressor(**bdt_params)
        model_mass = GradientBoostingRegressor(**bdt_params)

        model_pt.fit(features, targets[0])
        model_mass.fit(features, targets[1])

        return model_pt, model_mass

    @staticmethod
    def apply(data_train, models: tuple):
        """
        Apply BDT-based corrections to jets
        """
        # if self.model_pt is None or self.model_mass is None:
        #     raise RuntimeError("Models not trained!")

        X = BDT_Jet.features(data_train)

        pt_sf = models[0].predict(X)
        m_sf  = models[1].predict(X)
        return pt_sf, m_sf
        # pt_sf = np.clip(pt_sf, self.clip[0], self.clip[1])
        # m_sf  = np.clip(m_sf,  self.clip[0], self.clip[1])

        # pt_corr = ak.to_numpy(data["pt"])   * pt_sf
        # m_corr  = ak.to_numpy(data["mass"]) * m_sf

        # data["pt_corr"]   = pt_corr
        # data["mass_corr"] = m_corr

        # return data

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

        print("Getting features from data")
        features = self.features(train)
        print("Got features!")

        print("Training BDT")
        model_pt, model_mass = self.train(features, (1/pt_response, 1/mass_response))
        print("BDT trained!")

        print("Applying BDT")
        pt_sf, mass_sf = self.apply(train, (model_pt, model_mass))
        print("DONE!")

        return test, train, pt_sf, mass_sf