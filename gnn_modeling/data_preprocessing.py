import pandas as pd
from calc import calc_ddg, calc_ee
from sklearn.preprocessing import MinMaxScaler

smiles_dic = {
    'List_ACIE_THF': ['smiles', 'Ar'],
    'List_ACIE_THP': ['smiles', 'Ar', 'R'],
    'Sunoj_PNAS': ['smiles'],
    'HongAckermann_NatSyn': ['Biaryl', 'Olefin', 'Catalyst', 'TDG'],
    'SigmanToste_Science': ['cat_smiles', 'sub_smiles'],
}

def preprocess_data(dataset, tc):

    df = pd.read_csv(f'datasets_230520/{dataset}.csv')

    # Normalize 'ee' and 'ddG' columns to range [0, 1]
    scaler = MinMaxScaler()
    df[tc] = scaler.fit_transform(df[[tc]])

    smiles_list = smiles_dic[dataset]
    return df, smiles_list, scaler

def load_both_datasets(dataset, tc, with_T):

    # original data
    df_orig = pd.read_csv(f'datasets_230520/{dataset}.csv')

    # predicted data
    if with_T:
        with_T = 'withT'
    else:
        with_T = 'withoutT'

    df_pred = pd.read_csv(f'nfold_det/{dataset}-GNN_{tc}_{with_T}.csv', delimiter=',')
    
    return df_orig, df_pred

def load_both_datasets_v2(dataset, prediction_dataset):

    # original data
    df_orig = pd.read_csv(f'datasets_230520/{dataset}.csv')
    df_pred = pd.read_csv(prediction_dataset, delimiter=',')
    df_pred = df_pred.dropna(axis=1, how='any')

    n_columns_after_drop = df_pred.shape[1]
    #print("Number of columns after dropping:", n_columns_after_drop)
    return df_orig, df_pred