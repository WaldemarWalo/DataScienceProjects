import numpy as np
import pandas as pd
import time 
from IPython.display import clear_output
import matplotlib.pyplot as plt

def cv_classification(model_and_params, cv, X, y, eval_metric, model_ypred_return_list = None, RS=35566):
    get_score = lambda model, X_test : model.predict_proba(X_test)[:, 1]
    return cv_base(model_and_params, cv, X, y, eval_metric, model_ypred_return_list, RS=35566, get_score=get_score)
                                                     
def cv_regression(model_and_params, cv, X, y, eval_metric, model_ypred_return_list = None, RS=35566):
    get_score = lambda model, X_test : model.predict(X_test)
    return cv_base(model_and_params, cv, X, y, eval_metric, model_ypred_return_list, RS=35566, get_score=get_score)

def cv_base(model_and_params, cv, X, y, eval_metric, model_ypred_return_list = None, RS=35566, get_score=None):
    np.random.seed(RS)
    get_random = lambda  : np.random.randint(1, 2**16)
    
    trained_models_and_y_pred = []
    
    cv_scores_dict = {}
    for metric in eval_metric:
        cv_scores_dict[metric.__name__] = []
            
    n_folds_completed = 0
    
    start_time = time.perf_counter()

    for i_fold, (idx_train, idx_test) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train].values
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test].values

        constructor, params_dic = model_and_params

        if 'random_state' in params_dic.keys():
            params_dic['random_state'] = get_random()

        model = constructor(**params_dic)
        

        model.fit(X_train, y_train)
        y_pred = get_score(model, X_test)
        [ cv_scores_dict[metric.__name__].append(metric(y_test, y_pred)) for metric in eval_metric ]
        
        y_pred_ser = pd.Series(index = X_test.index, data=y_pred, name=f'fold_{i_fold}')
        # trained_models_and_y_pred.append((model, y_pred_ser))
        trained_models_and_y_pred.append(model)
        n_folds_completed += 1
        
    model_name = model.__class__.__name__
    model_params = params_dic.copy()
        
    total_elapsed_time = time.perf_counter() - start_time
    cv_results = model_name, model_params, n_folds_completed, total_elapsed_time, cv_scores_dict
    
    if model_ypred_return_list is not None:
        print('model_ypred_return_list')
        model_ypred_return_list.append(trained_models_and_y_pred)
        
    return get_stats(cv_results)

metric_name_map = {
    'mean_absolute_error': 'MAE',
    'root_mean_squared_error': 'RMSE',
    'r2_score': 'R2',
    'roc_auc_score': 'ROC_AUC'
}
    
def get_stats(result):
    params_to_ignore = ['random_state', 'silent', 'allow_writing_files']
    
    model_params = result[1]
    for param in params_to_ignore:
        model_params.pop(param, None)
    
    result_dict = {}
    result_dict['model'] = result[0]
    result_dict['params'] = str(model_params).strip('{').strip('}')
    result_dict['n_folds'] = result[2]

    metrics_and_scores = result[4]
    for k, v in metrics_and_scores.items():
        if k in metric_name_map:
            k = metric_name_map[k]

        result_dict[f'{k}_mean'] = np.mean(v)
        result_dict[f'{k}_std'] =  np.std(v)

    result_dict['time'] = result[3]
    # result_dict['models_and_pred'] = result[-1]
    return result_dict
    

def display_stats(df_stats, clear=True, reverse_rank_idx=[]):
    # df_stats = pd.DataFrame(stats).drop(columns = 'models_and_pred', errors='ignore')
    
    metrics_start_col_idx = 3
    metrics_menstd_cols = df_stats.columns[metrics_start_col_idx: df_stats.columns.get_loc('time')]
    rank_cols = []

    for i_metric, metric_col in enumerate(metrics_menstd_cols[::2]):
        metric_col_idx = df_stats.columns.get_loc(metric_col)

        rank_col_idx = metric_col_idx + 2
        rank_col_name = f'#{i_metric + 1}'
        rank_cols.append(rank_col_name)
        
        if i_metric in reverse_rank_idx:
            df_stats.insert(rank_col_idx, rank_col_name, df_stats[metric_col].rank(ascending=False).astype(int))
        else:
            df_stats.insert(rank_col_idx, rank_col_name, df_stats[metric_col].rank().astype(int))
    
    styler = df_stats.style

    for c in metrics_menstd_cols:
        gmap = df_stats[c].rank()
        vmin = -0.1 * gmap.max()
        _ = styler.background_gradient(cmap='Blues', subset=c, gmap=gmap, vmin=vmin)

        _range = df_stats[c].max() - df_stats[c].min()
        _range
        styler.format('{:.3f}', c)
        # if _range < 10:
        #     styler.format('{:.3f}', c)
        # elif _range < 100:
        #     styler.format('{:.2f}', c)
        # else:
        #     styler.format('{:.0f}', c)

    for c in rank_cols:
        _ = styler.background_gradient(cmap='Oranges', subset=c, vmin=-2)
        
    styler.format('{:,.1f}', 'time').bar(subset='time')
    
    if clear:
        clear_output(wait=True)
    display(styler)
    
    
## Preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
def ordinalEncode(df, cols):
    for col in cols:
        df.loc[:, col] = OrdinalEncoder(dtype=int).fit_transform(df[col].values.reshape(-1, 1))
        
def oh_encode(df, cols, drop_encoded=True):
    for col in cols:
        for val in df[col].unique():
            new_col = f'{col}__{val}'
            df[new_col] = 0
            df.loc[df[col] == val, new_col] = 1 
        if drop_encoded:
            df.drop(col, axis=1, inplace=True)
            
def mad(s):
    return (s - s.mean() ).abs().mean()

from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.preprocessing import PolynomialFeatures
def polynomialFeatures(X, degree):
    pf = PolynomialFeatures(degree = degree)
    X_poly = pf.fit_transform(X)
    return pd.DataFrame(
        columns = pf.get_feature_names_out(), 
        index = X.index, 
        data = X_poly
    )

def get_fe_df(list_of_list_of_models):
    df_all_fe = []
    
    for i, models_to_analyse in enumerate(list_of_list_of_models):
        feature_imp_for_model = []

        for j, m in enumerate(models_to_analyse):
            model_short_name = ''.join([l for l in m.__class__.__name__ if l.isupper()])

            if hasattr(m, 'feature_names_in_'):
                feature_names = m.feature_names_in_
            elif hasattr(m, 'feature_name_'):
                feature_names = m.feature_name_
            else:
                feature_names = m.feature_names_

            feature_imp_for_model.append(
                pd.Series(index = feature_names, data = m.feature_importances_, name=f'{model_short_name}_{i}_{j}')
            )

        df_f_imp = pd.concat(feature_imp_for_model, axis=1)
        df_f_imp[f'{model_short_name}_{i}_sum'] = df_f_imp.sum(axis=1)
        df_f_imp[f'{model_short_name}_{i}_rank'] = df_f_imp[f'{model_short_name}_{i}_sum'].rank().astype(int)

        df_all_fe.append(df_f_imp)
        df_final_fe = pd.concat(df_all_fe, axis=1)
        
    sum_cols = [col for col in df_final_fe.columns if col.endswith('_rank')]
    # df_final_fe['sum'] = df_final_fe[sum_cols].sum(axis=1)
    df_final_fe['sum_rank'] = df_final_fe[sum_cols].sum(axis=1).astype(int)
    df_final_fe = df_final_fe.sort_values(by='sum_rank', ascending=False)


    rank_cols = [col for col in df_final_fe.columns if col.endswith('_rank')]
    styler = df_final_fe[rank_cols].style
    styler.background_gradient(subset=rank_cols, cmap=plt.cm.Oranges, vmin=-5)
    return styler

