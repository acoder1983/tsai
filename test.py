from datetime import datetime as dt
from tsai.all import *
import numpy as np

x=np.random.random((100,1,5))
y=np.random.random((100,3))

splits = TimeSplitter(y.shape[0]//5,show_plot=False)(y) 
batch_tfms = TSStandardize()
fcst = TSForecaster(x, y, splits=splits, path='models', batch_tfms=batch_tfms, bs=128, arch=Informer, metrics=mse)
fcst.fit_one_cycle(1, 1e-3)

raw_preds, target, preds = fcst.get_X_preds(x[splits[1]], y[splits[1]])
print(raw_preds[0])