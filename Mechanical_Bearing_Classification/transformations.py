version = '0.1'
print(f'transformations loaded, version: {version}')

import pandas as pd
import numpy as np
import seaborn as sns

def get_iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

def drop_rpm_outliers(df: pd.DataFrame):
    # removes 5073 items
    return df.loc[ (df['rpm'] >= 0) & (df['rpm'] < 6000)]

def drop_w_outliers(df: pd.DataFrame):
    # removes 17 items
    return df.loc[df['w'] < 3.5]

def drop_idle_rpm(df: pd.DataFrame):
    return df.loc[df['rpm'] != 0]

def plt_8_params_for_experiment(df, axs, exp_id, s=0.01): 
    df = df[['rpm', 'a1_x', 'a1_y', 'a1_z', 'w', 'a2_x', 'a2_y', 'a2_z', 'timestamp']]
    y_min_max_col = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
    

    
    for ax_i, (ax, col) in enumerate(zip(axs, df.columns[:len(axs)])):
        
        _ = ax.scatter(df['timestamp'], df[col], s=s)
        _ = ax.set_title(f'{exp_id} - {col}')
        
        if col in y_min_max_col:
            y_min = df[y_min_max_col].min().min()
            y_max = df[y_min_max_col].max().max()
        else:
            y_min = df[col].min()
            y_max = df[col].max()
            # ax.get_xaxis().set_ticks([])
            # ax.get_yaxis().set_ticks([])

        buffer= 0.1
        y_max = y_max + buffer * np.abs(y_max)
        y_min = y_min - buffer * np.abs(y_min)
        _ = ax.set_ylim(y_min, y_max)
        
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
            
def plt_8_params_for_experiment_clustered(df, axs, exp_id, s=0.01):
    df = df[['rpm', 'a1_x', 'a1_y', 'a1_z', 'w', 'a2_x', 'a2_y', 'a2_z', 'timestamp', 'rpm_clusters']]
    y_min_max_col = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
    
    for ax_i, (ax, col) in enumerate(zip(axs, df.columns[:len(axs)])):
        
        # _ = ax.scatter(df['timestamp'], df[col], s=s)
        _ = sns.scatterplot(ax=ax, data = df, x='timestamp', y=col, hue='rpm_clusters', palette=['r', 'g', 'b'], s=10)
        _ = ax.set_title(f'{exp_id} - {col}')
        
        if col in y_min_max_col:
            y_min = df[y_min_max_col].min().min()
            y_max = df[y_min_max_col].max().max()
        else:
            y_min = df[col].min()
            y_max = df[col].max()
            # ax.get_xaxis().set_ticks([])
            # ax.get_yaxis().set_ticks([])

        buffer= 0.1
        y_max = y_max + buffer * np.abs(y_max)
        y_min = y_min - buffer * np.abs(y_min)
        _ = ax.set_ylim(y_min, y_max)