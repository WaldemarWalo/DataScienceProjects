version = '0.1'
print(f'framework loaded, version: {version}')

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

def get_ax(size_ratio = 1, w_aspect = 1):
    return get_axes(1, 1, size_ratio, w_aspect)[0,0]

def get_axes(row_count=1, col_count=1, size_ratio = 1, w_aspect = 1) -> np.ndarray:
    # max default kaggle's width resolution
    max_width = 1080
    fig_w = (max_width / 72 * size_ratio)
    fig_h = ((row_count/col_count) * 1080 / 72 * size_ratio / w_aspect)
    fig, axs = plt.subplots(row_count, col_count, figsize=(fig_w, fig_h), layout='constrained');
    
    # ensure we are returning an ndarray with 2d shape
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    if len(axs.shape) == 1:
        axs = axs.reshape(len(axs), 1)
        
    return axs