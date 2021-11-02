#from brain_plot import brainViz
import importlib
import lib.adni.visualizer3d
importlib.reload(lib.adni.visualizer3d)
from lib.adni.visualizer3d import Visualizer as vis3d
import pickle
import numpy as np
import scipy.stats
import os

import importlib


def normalize(data):
    for t in range(data.shape[0]):
        data_ = data[t]
        data[t] = (data_ - data_.min()) / (data_.max() - data_.min())
    return data


data_path = "/home/nazarovs/projects/panel_me_ode/results/adni_extrap_test/000_29.pickle"
res_path = "/".join(
    data_path.split("/")[:-3]) + "/plots/adni_extrap_test_summary/"
os.makedirs(res_path, exist_ok=True)
#stat_path = "/".join(data_path.split("/")[:-3]) + "/stats/adni_toy"
#os.makedirs(stat_path, exist_ok=True)

# Load data
sample = 0
with open(data_path, 'rb') as f:
    x, pred, best_pred, best_me = pickle.load(f)
x, pred, best_pred, best_me = \
    x[sample], pred[:, sample], best_pred[sample], best_me[sample]
#x = normalize(x)
#pred = normalize(pred)
#best_pred = normalize(best_pred)

frame_size = np.array((x.shape[-3], x.shape[-1]))  # size of image

# ========================================
# (1) plot 3 slices of 1 stat for 3 steps
# ========================================
slices = (50, 60, 50)
# Plot x
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = x
frame.make_plot(data_plot)
vizpath = res_path + "real_%d.png" % sample
frame.saveIt(path=vizpath)

# Plot best_pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = best_pred
frame.make_plot(data_plot)
vizpath = res_path + "best_pred_%d.png" % sample
frame.saveIt(path=vizpath)

# Plot diff between readl and best_pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = np.abs(x - best_pred)
frame.make_plot(data_plot)
vizpath = res_path + "diff_real_best_pred_%d.png" % sample
frame.saveIt(path=vizpath)
print("MSE x, best_pred: %f" % (((x[-1] - best_pred[-1])**2).mean()))

# Plot mean of pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = pred.mean(axis=0)
frame.make_plot(data_plot)
vizpath = res_path + "mean_pred_%d.png" % sample
frame.saveIt(path=vizpath)

# Plot diff between real and mean of pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = np.abs(x - pred.mean(axis=0))
frame.make_plot(data_plot)
vizpath = res_path + "diff_real_mean_pred_%d.png" % sample
frame.saveIt(path=vizpath)

# Plot diff between best_pred and mean of pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = np.abs(best_pred - pred.mean(axis=0))
frame.make_plot(data_plot)
vizpath = res_path + "diff_best_mean_pred_%d.png" % sample
frame.saveIt(path=vizpath)

# Plot std of pred
frame = vis3d(frame_size=frame_size, slices=slices, ncols=x.shape[0])
data_plot = pred.std(axis=0)
frame.make_plot(data_plot)
vizpath = res_path + "std_pred_%d.png" % sample
frame.saveIt(path=vizpath)

## Stats summary
mean_pred = pred.mean(axis=0)
print("MSE x, best_pred: %f" % (((x - best_pred)**2).mean()))
print("MSE x, mean_pred: %f" % (((x - mean_pred)**2).mean()))

print("MSE_extrap x, best_pred: %f" % (((x[-1] - best_pred[-1])**2).mean()))
print("MSE_extrap x, mean_pred: %f" % (((x[-1] - mean_pred[-1])**2).mean()))
