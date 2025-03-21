import awkward as ak
import numpy as np
import xgboost as xgb

def load_model(model_path):
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model


def preprocess(jets):
    shape = ak.num(jets)    # get shape for ak.unflatten( flat, ak.num(sig_jets) )
    jets_flat = ak.flatten(jets)
    return ak.to_numpy( np.column_stack( (jets_flat.pt, jets_flat.eta, jets_flat.mass, jets_flat.nDau) ) ), shape


def predict_sf(jets, model, shape):
    sf_pred = model.predict(jets)
    return ak.unflatten(sf_pred, shape)


def apply_corrections(jets, model): 

    if model == None:
        jets = ak.with_field(jets, 1, where="sf_pred")
    else:
        inputs, shape = preprocess(jets)
        scale_factors = predict_sf(inputs, model, shape) 
        jets = ak.with_field(jets, scale_factors, where="sf_pred")
        
    jets = ak.with_field(jets, jets.sf_pred * jets.mass, where="mass_corr")
    jets = ak.with_field(jets, jets.genmass / jets.mass, where="sf_true")
    
    return jets
    