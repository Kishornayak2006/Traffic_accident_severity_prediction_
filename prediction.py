import xgboost as xgb
import numpy as np
from config import *

def get_prediction(model, data):

    dmatrix = xgb.DMatrix(data)

    pred = model.predict(dmatrix)

    pred_class = int(np.argmax(pred))

    # Safe dictionary lookup
    return Accident_severity_dict.get(pred_class, "Prediction Error")