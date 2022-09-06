version = '0.1'
print(f'transformations loaded, version: {version}')

import pandas as pd

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

def plt_8_params_for_experiment(df, axs, exp_id, s=0.05): 
    df = df[['rpm', 'w', 'a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'timestamp']]
    y_min_max_col = ['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z']
    
    for ax_i, (ax, col) in enumerate(zip(axs, df.columns[:len(axs)])):
        y_min = df[y_min_max_col].min().min() * 0.95
        y_max = df[y_min_max_col].max().max() * 1.05
        
        _ = ax.scatter(df['timestamp'], df[col], s=s)
        _ = ax.set_title(f'{exp_id} - {col}')
        if col in y_min_max_col:
            _ = ax.set_ylim(y_min, y_max)
        else:
            y_min = df[col].min() * 0.95
            y_max = df[col].max() * 1.05
            _ = ax.set_ylim(y_min, y_max)

# def plt_8_params(df, axs, s=0.25): 
#     for ax_i, (ax_, col) in enumerate(zip(axs, df.columns[:len(axs)])):
#         for cluster in sorted(df['rpm_clusters'].unique()):
#             x = df.loc[df['rpm_clusters'] == cluster]
#             _ = ax_.scatter(x['timestamp'], x[col], s=s)
#         _ = ax_.set_title(f'{col}')