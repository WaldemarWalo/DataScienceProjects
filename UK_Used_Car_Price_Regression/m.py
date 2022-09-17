# import numpy as np
# import time 
# import pandas as pd
# from IPython.display import clear_output

# import itertools
# def grid_exec1(func, *args):
#     return [func(*element) for element in itertools.product(*args)]

# def grid_exec(func, *args, on1Completed=None):
#     all_results = []
#     for param_product in itertools.product(*args):
#         result = func(*param_product) 
#         all_results.append(result)
#         if on1Completed:
#             on1Completed(all_results)
#     return all_results

# def grid_exec_callback(func, callback, *args):
#     return [ callback(func(*element)) for element in itertools.product(*args)]



# from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
# def cross_val(model_and_params, X, y, eval_metric, n_folds = 5, RS=35566):
#     np.random.seed(RS)
#     get_random = lambda  : np.random.randint(1, 2**16)
    
#     cv_scores_dict = {}
#     for metric in eval_metric:
#         cv_scores_dict[metric.__name__] = []
            
#     n_folds_completed = 0
    
#     start_time = time.perf_counter()
    
#     # cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_fold_repeats, random_state=get_random())
#     cv = KFold(n_splits=n_folds)
#     for i_fold, (idx_train, idx_test) in enumerate(cv.split(X)):
#         X_train, y_train = X.iloc[idx_train], y.iloc[idx_train].values
#         X_test, y_test = X.iloc[idx_test], y.iloc[idx_test].values

#         constructor, params_dic = model_and_params

#         if 'random_state' in params_dic.keys():
#             params_dic['random_state'] = get_random()

#         model = constructor(**params_dic)
#         # all_trained_models.append(model)

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         # scores_dict = [ (metric.__name__, metric(y_test, y_pred)) for metric in eval_metric ]

#         [ cv_scores_dict[metric.__name__].append(metric(y_test, y_pred)) for metric in eval_metric ]
#         n_folds_completed += 1
        
#     model_name = model.__class__.__name__
#     model_params = params_dic.copy()
        
#     total_elapsed_time = time.perf_counter() - start_time
#     return model_name, model_params, n_folds_completed, total_elapsed_time, cv_scores_dict


# def get_stats_df(cv_results):
#     ret_list = []
    
#     for result in cv_results:
        
#         model_params = result[1]
#         model_params.pop('random_state', None)
#         model_params.pop('silent', None)
        
#         result_dict = {}
#         result_dict['model'] = result[0]
#         result_dict['params'] = str(model_params).strip('{').strip('}')
#         result_dict['n_folds'] = result[2]

#         for k, v in result[4].items():
#             result_dict[f'{k}_mean'] = np.mean(v)
#             result_dict[f'{k}_std'] =  np.std(v)
        
#         result_dict['time'] = result[3]

#         ret_list.append(result_dict)
        
#     return pd.DataFrame(ret_list)
    

# def display_stats(df_stats, clear=True):
#     styler = df_stats.style
#     styler.format('{:,.1f}', 'time')\
#           .bar(subset='time')

#     for c in df_stats.columns[3: -1]:
#         gmap = df_stats[c].rank()
#         vmin = -0.1 * gmap.max()
#         styler.background_gradient(cmap='Blues', subset=c, gmap=gmap, vmin=vmin)
        
#         _range = df_stats[c].max() - df_stats[c].min()
#         if _range < 1:
#             styler.format('{:.3f}', c)
#         elif _range < 10:
#             styler.format('{:.1f}', c)
#         # elif _range > 10:
#         #     styler.format('{:.0f}', c)
#         else:
#             styler.format('{:.0f}', c)
            
#     if clear:
#         clear_output(wait=True)
#     display(styler)