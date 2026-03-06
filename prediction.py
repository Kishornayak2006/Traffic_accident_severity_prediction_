import xgboost as xgb
import numpy as np
from config import *

def get_prediction(model, data):

    # convert input to DMatrix
    dmatrix = xgb.DMatrix(data)

    # model prediction
    pred = model.predict(dmatrix)

    # get class index
    pred_class = int(np.argmax(pred))

    return Accident_severity_dict[pred_class]