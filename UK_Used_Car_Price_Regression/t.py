# import pandas as pd

# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# def ordinalEncode(df, cols):
#     for col in cols:
#         df.loc[:, col] = OrdinalEncoder(dtype=int).fit_transform(df[col].values.reshape(-1, 1))
        
# def oh_encode(df, cols, drop_encoded=True):
#     for col in cols:
#         for val in df[col].unique():
#             new_col = f'{col}__{val}'
#             df[new_col] = 0
#             df.loc[df[col] == val, new_col] = 1 
#         if drop_encoded:
#             df.drop(col, axis=1, inplace=True)
            
        
    