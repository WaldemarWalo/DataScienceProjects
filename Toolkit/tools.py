import numpy as np
import matplotlib.pyplot as plt

def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

def get_ax(h_aspect = 1, size_ratio = 1):
    return get_axes(1, 1, h_aspect, size_ratio)[0,0]

def get_axes(row_count=1, col_count=1, h_aspect = 1, size_ratio = 1) -> np.ndarray:
    # max default kaggle's width resolution
    max_width = 1080
    matplot_default_dpi = 72
    fig_w = (max_width / matplot_default_dpi * size_ratio)
    fig_h = ((row_count/col_count) * max_width / matplot_default_dpi * size_ratio * h_aspect)
    # print((fig_w, fig_h))
    fig, axs = plt.subplots(row_count, col_count, figsize=(fig_w, fig_h), layout='constrained');
    
    # ensure we are returning an ndarray with 2d shape
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    if len(axs.shape) == 1:
        axs = axs.reshape(len(axs), 1)
        
    return axs


import pickle
from pathlib import Path

def to_pkl(obj:any, file_name: str, overwrite:bool = False) -> None:
    file_path = Path(file_name)
    if file_path.exists() and not overwrite:
        raise Exception(f'file already exits, set overwrite=True. File path {file_path}')
    file_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)
        
def from_pkl(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


import itertools
# def grid_exec1(func, *args):
#     return [func(*element) for element in itertools.product(*args)]

def grid_exec(func, *args, on1Completed=None):
    all_results = []
    for param_product in itertools.product(*args):
        result = func(*param_product) 
        all_results.append(result)
        if on1Completed:
            on1Completed(all_results)
    return all_results