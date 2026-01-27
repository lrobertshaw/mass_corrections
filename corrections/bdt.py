import numpy as np
import xgboost as xgb

def get_weights(df, target, eps=1e-6):

    if target == "pt":
        bins = np.linspace(0, 2048, 257)  # 4 GeV bins
    elif target == "mass":  # mass weights
        bins = np.linspace(0, 182, 92)  # 2 GeV bins

    df[f"{target}_bin"] = np.digitize(df["pt"], bins)

    norm = ( df.groupby(f"{target}_bin").apply( lambda g: np.median(np.abs(1.0 - (g["genpt"] / g["pt"]))) ) )
    counts = df[f"{target}_bin"].value_counts()

    df[f"{target}_weight"] = ( 1.0 / (df[f"{target}_bin"].map(counts) + eps) / (df[f"{target}_bin"].map(norm) + eps) )
    df[f"{target}_weight"] /= df[f"{target}_weight"].mean()

    return df


def train(df, target, features = ["pt", "mass", "eta"], loss="custom", params=None, weights=None):

    def _custom_pseudohuber_wrapper(delta=1.0):
        def l1_loss(preds, dtrain):
            w = dtrain.get_weight()
            y = dtrain.get_label()
            r = (preds - y) / y

            denom = np.sqrt(1.0 + (r / delta)**2)

            grad = r / denom
            hess = delta**2 / ((delta**2 + r**2)**1.5)
            if len(w) > 0:
                grad *= w
                hess *= w

            return grad, hess
        return l1_loss

    if loss == "custom":
        print("Using custom loss function!")
        loss = _custom_pseudohuber_wrapper()
    else: loss = "reg:squarederror"

    dtrain = xgb.DMatrix(
        data = df[features],
        label = 1 / df[f"{target}_response"],
        feature_names = features
        )
    
    if weights is not None:
        dtrain.set_weight(weights)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=12,
        obj=loss
    )

    return model