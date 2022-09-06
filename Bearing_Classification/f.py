version = '0.1'
print(f'framework loaded, version: {version}')

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

def get_ax(height=5, aspect=1):
    return get_axes(1, 1, height=height, aspect=aspect)[0,0]

def get_axes(row_count=1, col_count=1, height=5, aspect=1) -> np.ndarray:
    figsize = (col_count * height * aspect, row_count * height)
    fig, axs = plt.subplots(row_count, col_count, figsize=figsize, layout='constrained');
    
    # ensure we are returning an ndarray with 2d shape
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    if len(axs.shape) == 1:
        axs = axs.reshape(len(axs), 1)
        
    return axs